import os
import random
import csv
import json
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from packaging import version as pver
import scipy.io
import einops


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


class EpicAndRealestate(Dataset):
    def __init__(self,
        epic_root = '/workspace/Epic',
        epic_image_subfolder = 'epic',
        epic_posefile_subfolder = 'pose',
        epic_meta_file = "EPIC_100_train.csv",
        epic_caption_subfolder = 'caption_merged',
        h = 320,
        w = 512,
        num_frames=16,       # t
        epic_sample_stride=2,       # 2-3
        is_image=False,         # set to true to return C, H, W instead of T, C, H, W
        sample_by_narration=False,       # true: use narration as key
        is_valid = False,

        realestate_root = '/workspace/RealEstate',
        realestate_image_subfolder = './train',
        realestate_image_json = 'train.json',  # video_id -> [image_file_names]
        realestate_caption_json = 'train_captions.json',  # video_id -> [video_caption]
        realestate_pose_mat = 'train.mat',
        realestate_sample_stride = 4,       # 4-6
        relative_pose=True,
        zero_t_first_frame=True,
        rescale_fxy=False,
        use_flip=True,
    ) -> None:
        # RealEstate
        self.realestate_root = realestate_root
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.realestate_sample_stride = realestate_sample_stride
        self.t = num_frames

        self.realestate_image_subfolder = realestate_image_subfolder
        self.realestate_dataset = json.load(open(os.path.join(realestate_root, realestate_image_json), 'r'))
        self.caption_dict = json.load(open(os.path.join(realestate_root, realestate_caption_json), 'r'))
        self.mat = scipy.io.loadmat(os.path.join(realestate_root, realestate_pose_mat))

        self.w, self.h, self.sample_size = w, h, (h, w)
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = w / h
        self.use_flip = use_flip


        # EpicKitchen
        with open(os.path.join(epic_root, epic_meta_file), "r") as f:
            self.epic_meta_file = csv.reader(f)
            self.epic_meta_file = list(self.epic_meta_file)[1:] # drop the head line
        self.epic_root = epic_root
        self.epic_image_subfolder = epic_image_subfolder
        self.epic_posefile_subfolder = epic_posefile_subfolder
        self.epic_caption_subfolder = epic_caption_subfolder

        self.epic_sample_stride = epic_sample_stride
        self.datalist = []
        self.datalist_no_narration = []
        self.narration_to_videos = defaultdict(list)
        self.is_image = is_image
        self.is_valid = is_valid
        self.sample_by_narration = sample_by_narration

        self.transformer = transforms.Compose([
            transforms.Resize([self.h, self.w]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        videoid_to_timestamps = defaultdict(list)
        for line in self.epic_meta_file:
            video_id = line[2]
            # skip if the video is missing
            frame_dir = os.path.join(self.epic_root, self.epic_image_subfolder, video_id)
            if not os.path.exists(frame_dir): continue
            start_frame = int(line[6])
            end_frame = int(line[7])
            narration = line[8]
            videoid_to_timestamps[video_id].append((start_frame, end_frame))
            if end_frame < start_frame + 20: continue
            self.datalist.append((video_id, start_frame, end_frame, narration))
            self.narration_to_videos[narration].append((video_id, start_frame, end_frame,))

        # fill the gap for missing narration clips
        for video_id, timestampes in videoid_to_timestamps.items():
            timestampes = sorted(timestampes)
            
            # interval: time 0 to first snippet
            if timestampes[0][0] - 1 > 20:
                self.datalist_no_narration.append((video_id, 1, timestampes[0][0] - 1, 'moving'))
            
            for i in range(len(timestampes) - 1):
                start_frame = timestampes[i][1] + 1
                end_frame = timestampes[i + 1][0] - 1
                if end_frame < start_frame + 20: continue
                self.datalist_no_narration.append((video_id, start_frame, end_frame, 'moving'))
        
        print(f'EpicKitchen: there are {len(self.datalist)} videos with narrations, and {len(self.datalist_no_narration)} videos without narrations')
        # draw cases with and without narrations both
        self.datalist.extend(self.datalist_no_narration)
        print(f'dataset_epic: after merging data, total size is {len(self.datalist)}')
        print(f'dataset_realestate: total size is {len(self.realestate_dataset)}')
        self.narrations = [*self.narration_to_videos]
        with open(os.path.join(epic_root, 'all_pose.json')) as f:       # very big file
            self.frame_to_ex = json.load(f)
        with open(os.path.join(epic_root, 'intrinsics.json')) as f:
            self.videoid_to_ex = json.load(f)
    

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_cameras(self, idx):
        video_dict = self.realestate_dataset[idx]
        pose_file = os.path.join(self.realestate_root, video_dict['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params
    
    def realestate_get_batch(self, idx):
        video_id = list(self.realestate_dataset.keys())[idx]
        plist = self.mat[video_id]
        flist = self.realestate_dataset[video_id]
        flist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        video_caption = self.caption_dict[video_id + '.mp4'][0]
        total_frames = len(flist)

        current_sample_stride = random.choice([1, 4, 5, 6])
        if total_frames < self.t * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.t)
            current_sample_stride = random.randint(4, maximum_sample_stride)

        cropped_length = self.t * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.t
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.t, dtype=int)
        if self.use_flip and random.randint(0, 1) == 1:
            frame_indices = np.flip(frame_indices)

        # stack images into a tensor
        pixel_values = []
        for frame_idx in frame_indices:
            with Image.open(os.path.join(self.realestate_root, self.realestate_image_subfolder, video_id, flist[frame_idx])) as img:
                pixel_values.append(self.transformer(img).float())
        pixel_values = torch.stack(pixel_values, dim=0)  # t, c, h, w

        # load poses
        cam_params = [plist[indice] for indice in frame_indices]
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:  # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:  # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1]/8.0,
                                  cam_param.fy * self.sample_size[0]/8.0,
                                  cam_param.cx * self.sample_size[1]/8.0,
                                  cam_param.cy * self.sample_size[0]/8.0]
                                 for cam_param in cam_params], dtype=np.float32)
        # print(f'real origin {intrinsics}')
        intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]  # [1, n_frame, 4, 4]

        flip_flag = torch.zeros(self.t, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0]//8, self.sample_size[1]//8, device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
        # [V, H, W, 6] --> [V, 6, H, W]
        extrinsics = torch.empty((self.t, 4, 4))
        camera_embeddings = torch.empty((self.t, 12))
        intrinsics = plist[0][1:7]
        for i in range(self.t):
            index = frame_indices[i]
            pose = np.identity(4)
            tmp = plist[index][7:].reshape((3, 4))
            pose[:3, :4] = tmp
            if i == 0:
                base_pose = np.linalg.inv(np.array(pose))

            extrinsics[i] = torch.tensor(pose)
            camera_embeddings[i] = torch.tensor(np.matmul(base_pose, pose)[:3, :].flatten())
        
        # print(f'real normal {intrinsics}')

        return {
            'pixel_values': pixel_values,     # T, C, H, W
            'text': video_caption,     # str
            'intrinsics': plist[0][1: 7],       # 6,
            'extrinsics': extrinsics,       # T, 4, 4
            'plucker_embedding': plucker_embedding,     # T, 6, H, W
            'camera_embeddings': camera_embeddings,     # T, 12
            'frame_stride': current_sample_stride,
        }

    def epic_get_batch(self, index):
        if not self.sample_by_narration:
            video_id, start_frame, end_frame, narration = self.datalist[index]
        else:
            narration = self.narrations[index]
            if self.is_valid:
                # for validation, use fixed first video
                video_id, start_frame, end_frame = self.narration_to_videos[narration][0]
            else:
                video_id, start_frame, end_frame = random.sample(self.narration_to_videos[narration], 1)[0]

        epic_sample_stride = random.randint(2, 5)
        sample_frame_width = (self.t - 1) * epic_sample_stride + 1
        if start_frame + sample_frame_width - 1 <= end_frame:
            random_shift = random.randint(0, end_frame - start_frame + 1 - sample_frame_width)
            indices = [x + random_shift for x in range(start_frame, end_frame + 1, epic_sample_stride)][:self.t]
        else:
            indices = random.sample(list(range(start_frame, end_frame + 1)), self.t)
        assert len(indices) == self.t
        
        pixels = []
        extrinsics_lst = []
        intrinsic = tuple(self.videoid_to_ex[video_id])
        captions = []
        camera_embeddings = []

        for i, index in enumerate(indices):
            def index_to_keystring(index):
                return f'frame_{str(index).zfill(10)}.jpg'
            
            filename = f'frame_{str(index).zfill(10)}'

            # 1. get captions
            if i == 0 or i == self.t - 1:
                with open(os.path.join(self.epic_root, self.epic_caption_subfolder, video_id + '.json')) as f:
                    data = json.load(f)
                    # round frames to existing keys
                    shift = 0
                    while (index_to_keystring(index + shift) not in data) and (index_to_keystring(index - shift) not in data):
                        shift += 1
                    if index_to_keystring(index + shift) in data:
                        captions.append(data[index_to_keystring(index + shift)])
                    else:
                        captions.append(data[index_to_keystring(index - shift)])

            # 2. get pixels
            with Image.open(os.path.join(self.epic_root, self.epic_image_subfolder, video_id, filename + '.jpg')) as img:
                ori_h, ori_w = img.size
                pixels.append(self.transformer(img))
            
            # 3.a get camera poses, method 1: from official json
            # with open(os.path.join(self.epic_root, self.epic_posefile_subfolder, video_id + '.json')) as f:
            #     data = json.load(f)
            #     extrinsics_lst.append(_get_extrinsic_matrix(*data['images'][filename + '.jpg']))
            # 3.a get camera poses, method 2: from personal json                    
            extrinsics_lst.append(torch.tensor(self.frame_to_ex[video_id][f'{video_id}/{filename}.jpg']).float())
            # extrinsics_lst.append(torch.tensor(self.frame_to_ex[video_id][f'{filename}.jpg']).float())
            # 4. get camera embeddings
            if i == 0:
                base_pose = torch.inverse(extrinsics_lst[0])
            camera_embeddings.append((base_pose @ extrinsics_lst[i])[:3, :].flatten())

        pixels = torch.stack(pixels, dim = 0)
        extrinsics = torch.stack(extrinsics_lst, dim = 0)
        camera_embeddings = torch.stack(camera_embeddings, dim = 0)

        plucker_embedding = _get_plucker_embedding2(
            intrinsic=intrinsic,
            extrinsic_lst=list(map(lambda x: x.numpy(), extrinsics_lst)),
            h=self.h, w=self.w, t=self.t,
            ori_h=ori_h, ori_w=ori_w
        )
        # IMPORTANT!
        intrinsics = torch.tensor([
            intrinsic[0] / (2 * intrinsic[2]),
            intrinsic[1] / (2 * intrinsic[3]),
            0.5, 0.5, 0, 0
        ], dtype=torch.float32)
        # print(f'epic normal {intrinsics}')
        if self.is_image:
            pixels = pixels[0]
            text = captions[0] + ',' + captions[0] + ',' + narration
        else:
            text = captions[0] + ',' + captions[1] + ',' + narration

        return {
            'pixel_values': pixels,     # T, C, H, W
            'text': text,     # str
            'intrinsics': intrinsics,       # 6,
            'extrinsics': extrinsics,       # T, 4, 4
            'plucker_embedding': plucker_embedding,     # T, 6, H, W
            'camera_embeddings': camera_embeddings,     # T, 12
            'frame_stride': epic_sample_stride,
        }

    def __len__(self):    
        if self.sample_by_narration:
            return len(self.realestate_dataset) + len(self.narration_to_videos.keys())
        else:
            return len(self.realestate_dataset) + len(self.datalist)

    def __getitem__(self, idx):
        while True:
            try:
                if idx < len(self.realestate_dataset):
                    # res = self.realestate_get_batch(idx)
                    idx = random.randint(len(self.realestate_dataset), self.__len__() - 1)
                    continue
                else:
                    res = self.epic_get_batch(idx - len(self.realestate_dataset))
                break
            except Exception as err:
                idx = random.randint(0, self.__len__() - 1)
                # import traceback, sys
                # print(traceback.format_exc(), file=sys.stderr)
                # print(f'err: {err}', file=sys.stderr)

        # pixel_values: t, c, h, w
        # plucker_embedding: t, 6, h, w
        # intrinsics: 6,
        # extrinsics: t, 4, 4
        # camera_embeddings: t, 12
        res['video'] = einops.rearrange(res['pixel_values'], 't c h w -> c t h w')
        res['caption'] = res.pop('text')
        res['path'] = ''
        res['fps'] = 30.0 / res['frame_stride']
        return res

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    def custom_meshgrid(*args):
        # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
        if pver.parse(torch.__version__) < pver.parse('1.10'):
            return torch.meshgrid(*args)
        else:
            return torch.meshgrid(*args, indexing='ij')
    B, V = K.shape[:2]
    assert B == 1

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    K_ = K[0][0]
    fx, fy, cx, cy = K_[0], K_[1], K_[2], K_[3]

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def get_relative_pose(extrinsic_lst, zero_t_first_frame=True):
        """ extrinsic_lst: [4 * 4 extrinsics numpy array]"""
        abs_w2cs = extrinsic_lst
        abs_c2ws = [np.linalg.inv(mat) for mat in extrinsic_lst]
        source_cam_c2w = abs_c2ws[0]
        if zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses


def _get_plucker_embedding2(intrinsic, extrinsic_lst, h, w, ori_h, ori_w, t):
    """
    intrinsic: (fx, fy, cx, cy)
    extrinsic_lst: list of extrinsic 4 * 4 numpy matrices
    ori_h, ori_w: the original size of pixels before resizing
    """

    fx, fy, cx, cy = intrinsic
    intrinsics = np.array([fx/228*32, fy/128.0*20, cx/228*32, cy/128.0*20], dtype=np.float32)
    # print(f'epic origin {intrinsics}')
    intrinsics = torch.tensor(intrinsics).repeat(t, 1)
    intrinsics = torch.unsqueeze(intrinsics, dim=0).numpy()     # [1, t, 4]

    c2w_poses = get_relative_pose(extrinsic_lst)
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, t, 4, 4]

    return ray_condition(
            intrinsics,
            c2w,
            40,
            64,
            device='cpu'
    )[0].permute(0, 3, 1, 2).contiguous()
