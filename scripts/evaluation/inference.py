import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config

from lvdm.data.dataset_merged import _get_plucker_embedding2
import json
from lvdm.geometry.projection import get_world_rays
from lvdm.geometry.epipolar_lines import project_rays
from lvdm.visualization.drawing.lines import draw_attn

from collections import defaultdict
import csv
import random


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


VIDEO_ID = 'P36_102'
# with open(os.path.join(data_dir, 'all_pose.json')) as f:       # very big file
with open(os.path.join('/workspace/DynamiCrafter/Epic', f'{VIDEO_ID}_ex.json')) as f:       # very big file
    frame_to_ex = json.load(f)

def index_to_keystring(index):
    return f'frame_{str(index).zfill(10)}.jpg'


def get_all_camera_from_pose_list(extrinsics_list, intrinsic):
    camera_embeddings = []
    for i, extrinsics in enumerate(extrinsics_list):
        if i == 0:
            base_pose = torch.inverse(extrinsics)
        camera_embeddings.append((base_pose @ extrinsics)[:3, :].flatten())
    camera_embeddings = torch.stack(camera_embeddings, dim = 0)
    extrinsics = torch.stack(extrinsics_list, dim = 0)
    plucker_embedding = _get_plucker_embedding2(
        intrinsic=intrinsic,
        extrinsic_lst=list(map(lambda x: x.numpy(), extrinsics_list)),
        t=len(extrinsics_list),
    )
    intrinsics = torch.tensor([
        intrinsic[0] / (2 * intrinsic[2]),
        intrinsic[1] / (2 * intrinsic[3]),
        0.5, 0.5, 0, 0
    ], dtype=torch.float32)
    
    intrinsics = torch.unsqueeze(intrinsics, 0).to(0)
    extrinsics = torch.unsqueeze(extrinsics, 0).to(0)

    epipolar_masks = []
    b, t, h, w = 1, 16, 40, 64
    for s in range(3):
        epipolar_masks.append(_calculate_attn_mask(intrinsics, extrinsics, b,t,h// 2 ** (s + 1),w// 2 ** (s + 1),None,lw =4// 2 ** s))

    return {
        'intrinsics': intrinsics,       # 6,
        'extrinsics': extrinsics,       # T, 4, 4
        'plucker_embedding': torch.unsqueeze(plucker_embedding, 0).to(0),     # T, 6, H, W
        'camera_embeddings': torch.unsqueeze(camera_embeddings, 0).to(0),     # T, 12
        'epipolar_masks': epipolar_masks,
    }


def get_frame_pixel_repeated(frame: int, num_frames=16, data_dir='/workspace/DynamiCrafter/Epic'):
    transformer = transforms.Compose([
            transforms.Resize([320, 512]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    filename = index_to_keystring(frame)
    with Image.open(os.path.join(data_dir, 'epic', VIDEO_ID, filename)) as img:
        pixels = transformer(img)
    pixels = torch.stack([pixels] * num_frames, dim=0)
    pixels = rearrange(pixels, 't c h w -> c t h w')
    return torch.unsqueeze(pixels, 0).to(0)     # B, C, T, H, W


def get_BLIP(frame: int, data_dir='/workspace/DynamiCrafter/Epic'):
    with open(os.path.join(data_dir, 'caption_merged', VIDEO_ID + '.json')) as f:
        data = json.load(f)
        # round frames to existing keys
        shift = 0
        while (index_to_keystring(frame + shift) not in data) and (index_to_keystring(frame - shift) not in data):
            shift += 1
        if index_to_keystring(frame + shift) in data:
            return data[index_to_keystring(frame + shift)]
        else:
            return data[index_to_keystring(frame - shift)]


def get_list_by_stride(start, stride, length):
    return list(range(start, start + stride * length, stride))

def get_epic_data(data_dir='/workspace/DynamiCrafter/Epic', video_id='P35_101',
    start_frame=3706, num_frames=16):

    # 706 - 2400
    # {pixel 706, caption 706 + some final frame with action label, camera pose/plucker: T=16 t0=706 hand pick}
    # 2. get pixels
    transformer = transforms.Compose([
        transforms.Resize([320, 512]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    filename = index_to_keystring(start_frame)
    with Image.open(os.path.join(data_dir, 'epic', video_id, filename)) as img:
        ori_w, ori_h = img.size
        print(ori_h, ori_w)
        # pixels.append(transformer(img))
        pixels = transformer(img)
    pixels = torch.stack([pixels] * num_frames, dim=0)

    with open(os.path.join(data_dir, 'intrinsics.json')) as f:
        data = json.load(f)
        intrinsic = tuple(data[video_id])

    # 706 750 790 830
    extrinsics_lst = []
    camera_embeddings = []
    # frame_indices = [706, 750, 790, 830, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450]
    
    stride = random.choice([2, 3, 5])
    frame_indices = get_list_by_stride(start_frame, stride, num_frames)
    assert len(frame_indices) == num_frames
    captions = []
    for i, index in enumerate(frame_indices):
        filename = index_to_keystring(index)

        # 1. get captions
        if i == 0 or i == num_frames - 1:
            # BLIP
            with open(os.path.join(data_dir, 'caption_merged', video_id + '.json')) as f:
                data = json.load(f)
                # round frames to existing keys
                shift = 0
                while (index_to_keystring(index + shift) not in data) and (index_to_keystring(index - shift) not in data):
                    shift += 1
                if index_to_keystring(index + shift) in data:
                    captions.append(data[index_to_keystring(index + shift)])
                else:
                    captions.append(data[index_to_keystring(index - shift)])

        # 2. get extrinsics
        extrinsics_lst.append(torch.tensor(frame_to_ex[video_id][f'{video_id}/{filename}']).float())
        if i == 0:
            base_pose = torch.inverse(extrinsics_lst[0])
        camera_embeddings.append((base_pose @ extrinsics_lst[i])[:3, :].flatten())
    
    extrinsics = torch.stack(extrinsics_lst, dim = 0)
    camera_embeddings = torch.stack(camera_embeddings, dim = 0)

    plucker_embedding = _get_plucker_embedding2(
        intrinsic=intrinsic,
        extrinsic_lst=list(map(lambda x: x.numpy(), extrinsics_lst)),
        t=num_frames,
    )
    # IMPORTANT!
    intrinsics = torch.tensor([
        intrinsic[0] / (2 * intrinsic[2]),
        intrinsic[1] / (2 * intrinsic[3]),
        0.5, 0.5, 0, 0
    ], dtype=torch.float32)

    text = captions[0] + ',' + captions[1]

    pixels = rearrange(pixels, 't c h w -> c t h w')
    return {
        'pixel_values': torch.unsqueeze(pixels, 0).to(0),     # C, T, H, W
        'text': text,     # str
        'intrinsics': torch.unsqueeze(intrinsics, 0).to(0),       # 6,
        'extrinsics': torch.unsqueeze(extrinsics, 0).to(0),       # T, 4, 4
        'plucker_embedding': torch.unsqueeze(plucker_embedding, 0).to(0),     # T, 6, H, W
        'camera_embeddings': torch.unsqueeze(camera_embeddings, 0).to(0),     # T, 12
        'stride': stride,
    }
    
   
def load_data_prompts(data_dir, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2*idx]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            image2 = Image.open(file_list[2*idx+1]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            _, filename = os.path.split(file_list[idx*2])
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)
        
    return filename_list, data_list, prompt_list


def save_results(prompt, samples, filename, fakedir, fps=8, loop=False):
    filename = filename.split('.')[0]+'.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        if loop:
            video = video[:-1,...]
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'}) ## crf indicates the quality


def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            # path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{filename.split(".")[0]}_sample{i}.mp4')
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), filename)
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    # if not text_input:
    #     prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        # if loop or interp:
        #     img_cat_cond = torch.zeros_like(z)
        #     img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
        #     img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        # else:
        #     img_cat_cond = z[:,:,:1,:,:]
        #     img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

        img_cat_cond = z[:,:,:1,:,:]
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])


        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def _calculate_attn_mask(intrinsics, extrinsics, b, t,h,w,x=None,lw=4):
    xs = torch.linspace(0, 1, steps=w)
    ys = torch.linspace(0, 1, steps=h)
    grid = torch.stack(
        torch.meshgrid(xs, ys, indexing='xy'), dim=-1).float().to(
        0)

    grid = rearrange(grid, "h w c  -> (h w) c")
    grid = grid.repeat(t, 1)
    attn_mask = []
    for b in range(b):
        k = torch.eye(3).float().to(
            0)
        k[0, 0] = intrinsics[b][0]
        k[1, 1] = intrinsics[b][1]
        k[0, 2] = 0.5
        k[1, 2] = 0.5
        source_intrinsics = k
        source_intrinsics = source_intrinsics[None].repeat_interleave(t * w * h, 0)

        source_extrinsics_all = []
        target_extrinsics_all = []
        for t1 in range(t):
            source_extrinsics = torch.inverse(extrinsics[b][t1].to(
                0))
            source_extrinsics_all.append(source_extrinsics[None].repeat_interleave(w * h, 0))
            tmp_seq = []
            for t2 in range(t):
                target_extrinsics = torch.inverse(extrinsics[b][t2].to(
                    0))
                tmp_seq.append(target_extrinsics[None])
            target_extrinsics_all.append(torch.cat(tmp_seq).repeat(w * h, 1, 1))

        source_extrinsics_all = torch.cat(source_extrinsics_all)
        target_extrinsics_all = torch.cat(target_extrinsics_all)
        origin, direction = get_world_rays(grid.float(), source_extrinsics_all.float(), source_intrinsics.float())
        origin = origin.repeat_interleave(t, 0)
        direction = direction.repeat_interleave(t, 0)
        source_intrinsics = source_intrinsics.repeat_interleave(t, 0)
        projection = project_rays(
            origin.float(), direction.float(), target_extrinsics_all.float(), source_intrinsics.float()
        )

        attn_image = torch.zeros((3, h, w)).to(
            0).float()

        attn_image = draw_attn(
            attn_image,
            projection["xy_min"],
            projection["xy_max"],
            (1, 1, 1),
            lw,
            x_range=(0, 1),
            y_range=(0, 1), )
        attn_image = attn_image
        attn_image = rearrange(attn_image, '(t1 a t2) b-> (t1 a) (t2 b)', t1=t, t2=t)
        attn_mask.append(attn_image)
    attn_mask = torch.stack(attn_mask).half()
    return  attn_mask


def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list = load_data_prompts(args.prompt_dir, video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))

    print(f'running inference on epic, video id {VIDEO_ID}')

    with open('/workspace/DynamiCrafter/Epic/EPIC_100_train.csv', "r") as f:
        epic_meta_file = csv.reader(f)
        epic_meta_file = list(epic_meta_file)[1:] # drop the head line
    for line in epic_meta_file:
        video_id = line[2]

        # if video_id != VIDEO_ID: continue
        # skip if the video is missing
        frame_dir = os.path.join('/workspace/DynamiCrafter/Epic/epic', video_id)
        if not os.path.exists(frame_dir): continue
        start_frame = int(line[6])
        # end_frame = int(line[7])
        narration = line[8]


        # print(f'getting data from {video_id}, {start_frame}')
        # try:
        #     data_info = get_epic_data(
        #         data_dir='/workspace/DynamiCrafter/Epic',
        #         video_id=video_id,
        #         start_frame=start_frame,
        #         num_frames=16
        #     )
        # except Exception:
        #     print(f'extrinsic missing on {video_id}, {start_frame}')
        #     exit(0)
        #     continue

        # # calculate epipolar mask
        # epipolar_masks = []
        # b, t, h, w = 1, 16, 40, 64
        # for s in range(3):
        #     epipolar_masks.append(_calculate_attn_mask(data_info['intrinsics'], data_info['extrinsics'], b,t,h// 2 ** (s + 1),w// 2 ** (s + 1),None,lw =4// 2 ** s))
        # data_info['epipolar_masks'] = epipolar_masks

        # reading data from pt file
        video_id = VIDEO_ID

        stride = 5
        # indices_to_read = [456, 460, 470] + get_list_by_stride(start=473, stride=3, length=13)
        # indices_to_read = [480] + get_list_by_stride(start=485, stride=5, length=15)
        # indices_to_read = [635, 765, 900] + get_list_by_stride(start=980, stride=3, length=13)
        # indices_to_read = get_list_by_stride(start=470, stride=3, length=16)
        indices_to_read = [470] + get_list_by_stride(4880, 3, 15)
        print(f'indices: {indices_to_read}')
        read_pt_mask = [True] * 0 + [False] * 16
        extrinsics_list = []
        pt_data = None
        for index, flag in zip(indices_to_read, read_pt_mask):
            filename = index_to_keystring(index)
            if flag:
                pt_data = torch.load(f'/workspace/DynamiCrafter/camera_data/{VIDEO_ID}_{filename}.pth')
                extrinsics_list.append(pt_data['extrinsics'])
            else:
                extrinsics_list.append(torch.tensor(frame_to_ex[video_id][f'{video_id}/{filename}']).float())

        # interpolate
        extrinsics_list = [torch.eye(4) for _ in range(16)]
        for i in range(16):
            # extrinsics_list[i][2, 3] = -0.2 * i
            extrinsics_list[i][2, 3] = -0.2 * i

        print(extrinsics_list)
        # assert pt_data is not None
        with open(os.path.join('/workspace/DynamiCrafter/Epic', 'intrinsics.json')) as f:
            data = json.load(f)
            intrinsic = tuple(data[video_id])

        # videos = get_frame_pixel_repeated(indices_to_read[0])       # repeated using 1st frame
        videos = get_frame_pixel_repeated(470)
        camera_data = get_all_camera_from_pose_list(extrinsics_list=extrinsics_list, intrinsic=intrinsic)
        # prompts = pt_data['text']
        prompts = f'{get_BLIP(indices_to_read[0])},{get_BLIP(indices_to_read[-1])},moving'
        print(camera_data['extrinsics'])
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            name_flag = 'selfmade'
            print(f'prompts {prompts}')
            batch_samples = image_guided_synthesis(
                model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                args.unconditional_guidance_scale, args.cfg_img, 30 // 5, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale,
                plucker_embedding = camera_data['plucker_embedding'],
                extrinsics = camera_data['extrinsics'],
                intrinsics = camera_data['intrinsics'],
                camera_embeddings = camera_data['camera_embeddings'],
                epipolar_masks = camera_data['epipolar_masks'],
            )

            # save each example individually
            for nn, samples in enumerate(batch_samples):
                narration = 'NA'
                print(f"saving {video_id}_{start_frame}_{narration}_{name_flag}.mp4")
                save_results_seperate(
                    prompts, samples,
                    f'{video_id}_{start_frame}_{narration}_{name_flag}.mp4', fakedir, fps=8, loop=args.loop
                )
        print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")

        exit(0)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=10, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--video_id")

    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)