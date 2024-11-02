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


class Epic(Dataset):
    def __init__(self,
        root = '/root/Epic',
        image_subfolder = 'frame-extracted',
        caption_subfolder = 'caption',
        extrinsic_file = 'all_pose.json',
        # extrinsic_file = 'sub_ex.json',
        h = 320,
        w = 512,
        num_frames = 16,       # t
        is_image = False,         # set to true to return C, H, W instead of T, C, H, W
    ) -> None:
        self.root = root
        self.image_subfolder = image_subfolder
        self.caption_subfolder = caption_subfolder

        self.video_ids = sorted(os.listdir(os.path.join(root, image_subfolder)))
        self.video_frame_lengths = [len(os.listdir(os.path.join(root, image_subfolder, video_id))) for video_id in self.video_ids]

        self.is_image = is_image
        self.t = num_frames
        self.h = h
        self.w = w

        self.transformer = transforms.Compose([
            transforms.Resize([self.h, self.w]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.caption_by_videoid = {}
        for video_id in self.video_ids:
            with open(os.path.join(root, caption_subfolder, f'{video_id}.json')) as f:
                self.caption_by_videoid[video_id] = json.load(f)

        with open(os.path.join(root, extrinsic_file)) as f:
            self.frame_to_ex = json.load(f)
        with open(os.path.join(root, 'intrinsics.json')) as f:
            self.videoid_to_ex = json.load(f)
        with open(os.path.join(root, 'high.json')) as f:
            self.high_data = json.load(f)
    

    def get_batch(self, may_repeat=True):
        def to_key(index):
            return f'frame_{str(index).zfill(10)}.jpg'
    
        idx = random.choices(list(range(len(self.video_ids))), weights=[float(x) / sum(self.video_frame_lengths) for x in self.video_frame_lengths], k=1)[0]
        video_id = self.video_ids[idx]
        
        strides = [random.randint(3, 5) for _ in range(self.t - 1)]
        video_length = self.video_frame_lengths[idx]
        start_frame = random.randint(0, self.video_frame_lengths[idx] - sum(strides) - 1)
        indices = [start_frame]
        for stride in strides:
            indices.append(stride + indices[-1])
        # in case 0 high, 50% to resample
        high_data = self.high_data[video_id]
        num_high = sum(list(map(lambda x: to_key(x) in high_data, indices)))
        if num_high == 0 and may_repeat:
            return self.get_batch(False)
        assert len(indices) == self.t

        pixels = []
        extrinsics_lst = []
        intrinsic = tuple(self.videoid_to_ex[video_id])
        captions = []
        camera_embeddings = []

        for i, index in enumerate(indices):            
            keyname = to_key(index)

            # 1. get captions
            def closest_in_sequence(x):     # round to 1 + 8x
                n = round((x - 1) / 8) + 1
                closest_value = 1 + 8 * (n - 1)
                return closest_value
            rounded_keyname = to_key(closest_in_sequence(index))        
            captions.append(self.caption_by_videoid[video_id][rounded_keyname])

            # 2. get pixels
            with Image.open(os.path.join(self.root, self.image_subfolder, video_id, keyname)) as img:
                pixels.append(self.transformer(img))
            
            # 3. get camera poses                    
            extrinsics_lst.append(torch.tensor(self.frame_to_ex[video_id][f'{video_id}/{keyname}']).float())

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
            t=self.t,
        )
        # IMPORTANT!
        intrinsics = torch.tensor([
            intrinsic[0] / (2 * intrinsic[2]),
            intrinsic[1] / (2 * intrinsic[3]),
            0.5, 0.5, 0, 0
        ], dtype=torch.float32)
        
        if self.is_image:
            pixels = pixels[0]
            text = captions[0] + ',' + captions[0]
        else:
            text = captions[0] + ',' + captions[-1]

        return {
            'pixel_values': pixels,     # T, C, H, W
            'text': text,     # str
            'intrinsics': intrinsics,       # 6,
            'extrinsics': extrinsics,       # T, 4, 4
            'plucker_embedding': plucker_embedding,     # T, 6, H, W
            'camera_embeddings': camera_embeddings,     # T, 12
            'frame_stride': 6,
        }

    def __len__(self):    
        return 48000

    def __getitem__(self, idx):
        while True:
            try:
                res = self.get_batch(True)
                break
            except Exception as err:
                pass
                # import traceback
                # print(traceback.format_exc())
                # exit(0)

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


# epic
def _get_plucker_embedding2(intrinsic, extrinsic_lst, t):
    """
    intrinsic: (fx, fy, cx, cy)
    extrinsic_lst: list of extrinsic 4 * 4 numpy matrices
    """

    fx, fy, cx, cy = intrinsic
    intrinsics = np.array([fx / 228.0 * 32, fy / 128.0 * 20, cx / 228.0 * 32, cy / 128.0 * 20], dtype=np.float32)
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


# # realestate
# def _get_plucker_embedding2(intrinsic, extrinsic_lst, t):
#     """
#     intrinsic: (fx, fy, cx, cy)
#     extrinsic_lst: list of extrinsic 4 * 4 numpy matrices
#     """

#     fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
#     intrinsics = np.array([fx*64, fy*40, cx*64, cy*40], dtype=np.float32)
#     intrinsics = torch.tensor(intrinsics).repeat(t, 1)
#     intrinsics = torch.unsqueeze(intrinsics, dim=0).numpy()     # [1, t, 4]

#     c2w_poses = get_relative_pose(extrinsic_lst)
#     c2w = torch.as_tensor(c2w_poses)[None]                          # [1, t, 4, 4]

#     return ray_condition(
#             intrinsics,
#             c2w,
#             40,
#             64,
#             device='cpu'
#     )[0].permute(0, 3, 1, 2).contiguous()