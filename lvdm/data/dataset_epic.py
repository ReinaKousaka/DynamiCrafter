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
from torch.nn import functional as F

from packaging import version as pver


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def _get_extrinsic_matrix(qw, qx, qy, qz, tx, ty, tz) -> torch.Tensor:
    """
    Args: quaternion and translation
    """
    extrinsic = np.eye(4)
    r = qvec2rotmat([qx, qy, qz, qw])
    extrinsic[:3, :3] = r
    extrinsic[:3, 3] = [tx, ty, tz]

    return torch.from_numpy(extrinsic).float()


def _get_rays(H, W, intrinsics, c2w):
    """
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    """
    u, v = torch.meshgrid(
        torch.arange(W, device=c2w.device),
        torch.arange(H, device=c2w.device),
        indexing="ij",
    )
    B = c2w.shape[0]
    u, v = u.reshape(-1), v.reshape(-1)
    u_noise = v_noise = 0.5
    u, v = u + u_noise, v + v_noise  # add half pixel
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H*W)
    pixels = pixels.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)
    if intrinsics.sum() == 0:
        inv_intrinsics = torch.eye(3, device=c2w.device).tile(B, 1, 1)
    else:
        inv_intrinsics = torch.linalg.inv(intrinsics)
    rays_d = inv_intrinsics @ pixels  # (B, 3, H*W)
    rays_d = c2w[:, :3, :3] @ rays_d
    rays_d = rays_d.transpose(-1, -2)  # (B, H*W, 3)
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = c2w[:, :3, 3].reshape((-1, 3))  # (B, 3)
    rays_o = rays_o.unsqueeze(1).repeat(1, H * W, 1)  # (B, H*W, 3)

    return rays_o, rays_d


# cite from: https://github.com/echen01/ray-conditioning/blob/8e1d5ae76d4747c771d770d1f042af77af4b9b5d/training/plucker.py#L9
def _get_plucker_embedding(H, W, intrinsics, c2w):
    """Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    c2w: (B, 4, 4)
    intrinsics: (B, 3, 3)
    """    
    cam_pos, ray_dirs = _get_rays(H, W, intrinsics, c2w)
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)

    plucker = plucker.view(-1, H, W, 6).permute(0, 3, 1, 2)
    return plucker  # B, 6, H, W


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
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
    intrinsics = torch.tensor(intrinsics).repeat(t, 1)
    intrinsics = torch.unsqueeze(intrinsics, dim=0).numpy()     # [1, t, 4]

    c2w_poses = get_relative_pose(extrinsic_lst)
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, t, 4, 4]

    return ray_condition(
            intrinsics,
            c2w,
            h,
            w,
            device='cpu'
    )[0].permute(0, 3, 1, 2).contiguous()


class EpicKitchen(Dataset):
    def __init__(self,
        root = '/workspace/CameraCtrl/Epic',
        image_subfolder = 'epic',
        posefile_subfolder = 'pose',
        meta_file = "EPIC_100_train.csv",
        caption_subfolder = 'caption_merged',
        h = 320,
        w = 512,
        sample_size = [320, 512],
        num_frames=16,
        sample_stride=4,
        is_image=False,         # set to true to return C, H, W instead of T, C, H, W
        sample_by_narration=False,       # true: use narration as key
        is_valid = False,
    ):
        """Define EpicKiten Dataset
        Args:
            root (str): path of images
            image_subfolder (str): relative path to the folder of video frames
            meta_file (str): relative path to the meta file, e.g. EPIC_100_train.csv
            num_frames (int): frames number of input images sequences
        """
        with open(os.path.join(root, meta_file), "r") as f:
            self.meta_file = csv.reader(f)
            self.meta_file = list(self.meta_file)[1:] # drop the head line
        self.root = root
        self.image_subfolder = image_subfolder
        self.posefile_subfolder = posefile_subfolder
        self.caption_subfolder = caption_subfolder
        self.h, self.w = h, w
        self.num_frames = num_frames
        self.sample_stride = sample_stride
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
        
        self._check_from_data()
        
        print(f'EpicKitchen: there are {len(self.datalist)} videos with narrations, and {len(self.datalist_no_narration)} videos without narrations')
        # draw cases with and without narrations both
        # self.ori_indices_for_valid = [len(self.datalist), len(self.datalist) + 2000, len(self.datalist) + 3000, 500, 1500, 2500, 3500]
        # self.ori_indices_for_valid = list(range(len(self.datalist), len(self.datalist) + 3500, 500))
        self.datalist.extend(self.datalist_no_narration)
        # self.datalist.extend(self.datalist_no_narration)
        # self.datalist.extend(self.datalist_no_narration)
        # self.datalist.extend(self.datalist_no_narration)
        print(f'dataset_epic: after merging data, total size is {len(self.datalist)}')
        if self.is_valid:
            random.shuffle(self.datalist)
        self.narrations = [*self.narration_to_videos]

        # The following snippet is to save frame_to_ex.json
        # self.frame_to_ex = {}
        # with open(os.path.join(root, 'epic_cam-pre.json')) as f:
        #     data = json.load(f)
        #     for chunk in data:
        #         narration = chunk[0]   
        #         for piece in chunk[1]:
        #             path = piece[0]
        #             path = os.path.join(root, image_subfolder, path[26:])
        #             # extrinsic = torch.tensor(piece[1]).float()
        #             extrinsic = piece[1]
        #             # self.datalist2.append((narration, path, extrinsic))
        #             splits = path.split('/')
        #             if splits[-2] not in self.frame_to_ex:
        #                 self.frame_to_ex[splits[-2]] = {}
        #             self.frame_to_ex[splits[-2]][splits[-1]] = extrinsic
        # with open(os.path.join(root, 'frame_to_ex.json'), 'w') as f:
        #     # self.frame_to_ex = json.load(f)
        #     json.dump(self.frame_to_ex, f)
        # exit(0)

        # with open(os.path.join(root, 'all_pose.json')) as f:
        #     self.frame_to_ex = json.load(f)
        # self.frame_to_ex = {}
        with open(os.path.join(root, 'intrinsics.json')) as f:
            self.videoid_to_ex = json.load(f)

    def _check_from_data(self):
        videoid_to_timestamps = defaultdict(list)
        for line in self.meta_file:
            video_id = line[2]
            # skip if the video is missing
            frame_dir = os.path.join(self.root, self.image_subfolder, video_id)
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
            for i in range(len(timestampes) - 1):
                start_frame = timestampes[i][1] + 1
                end_frame = timestampes[i + 1][0] - 1
                if end_frame < start_frame + 20: continue
                self.datalist_no_narration.append((video_id, start_frame, end_frame, 'moving'))

    def __len__(self):
        if self.sample_by_narration:
            return len(self.narration_to_videos.keys())
        else:
            return len(self.datalist)
    
    def __getitem__(self, index):
        while True:
            try:
                if not self.sample_by_narration:
                    # if self.is_valid:
                    #     index = self.ori_indices_for_valid[index % len(self.ori_indices_for_valid)]
                    video_id, start_frame, end_frame, narration = self.datalist[index]
                else:
                    narration = self.narrations[index]
                    if self.is_valid:
                        # for validation, use fixed first video
                        video_id, start_frame, end_frame = self.narration_to_videos[narration][0]
                    else:
                        video_id, start_frame, end_frame = random.sample(self.narration_to_videos[narration], 1)[0]
                sample_stride = self.sample_stride
                sample_frame_width = (self.num_frames - 1) * sample_stride + 1
                if start_frame + sample_frame_width - 1 <= end_frame:
                    random_shift = random.randint(0, end_frame - start_frame + 1 - sample_frame_width)
                    indices = [x + random_shift for x in range(start_frame, end_frame + 1, sample_stride)][:self.num_frames]
                else:
                    indices = random.sample(list(range(start_frame, end_frame + 1)), self.num_frames)
                assert len(indices) == self.num_frames
                
                pixels = []
                extrinsics_lst = []
                intrinsic = tuple(self.videoid_to_ex[video_id])
                captions = []
                camera_embeddings = []

                for i, index in enumerate(indices):
                    filename = f'frame_{str(index).zfill(10)}'

                    # 1. get captions
                    if i == 0 or i == self.num_frames - 1:
                        with open(os.path.join(self.root, self.caption_subfolder, video_id + '.json')) as f:
                            data = json.load(f)
                            # round frames to existing keys, current ends with: 1, 3, 6, 8
                            round_mapping = {
                                0: 1,
                                1: 0,
                                2: -1,
                                3: 0,
                                4: -1,
                                5: 1,
                                6: 0,
                                7: -1,
                                8: 0,
                                9: -1,
                            }
                            index_ = index + round_mapping[index % 10]
                            captions.append(data[f'frame_{str(index_).zfill(10)}.jpg'])

                    # 2. get pixels
                    with Image.open(os.path.join(self.root, self.image_subfolder, video_id, filename + '.jpg')) as img:
                        ori_h, ori_w = img.size
                        pixels.append(self.transformer(img))
                    
                    # 3.a get camera poses, method 1: from official json
                    # with open(os.path.join(self.root, self.posefile_subfolder, video_id + '.json')) as f:
                    #     data = json.load(f)
                    #     extrinsics_lst.append(_get_extrinsic_matrix(*data['images'][filename + '.jpg']))
                    # 3.a get camera poses, method 2: from personal json
                    extrinsics_lst.append(torch.tensor(self.frame_to_ex[video_id][f'{video_id}/{filename}.jpg']).float())
                    # extrinsics_lst.append(torch.tensor(self.frame_to_ex[video_id][f'{filename}.jpg']).float())
                    # 4. get camera embeddings
                    if i == 0:
                        base_pose = torch.inverse(extrinsics_lst[0])
                    camera_embeddings.append((base_pose @ extrinsics_lst[i])[:3, :].flatten())
                
                break
            except Exception as err:
                # skip if the corresponding camera pose is missing, get a random idx instead
                index = random.randint(0, self.__len__() - 1)
                # import traceback
                # print(traceback.format_exc())
                # print(f'err: {err}')

        pixels = torch.stack(pixels, dim = 0)
        extrinsics = torch.stack(extrinsics_lst, dim = 0)
        camera_embeddings = torch.stack(camera_embeddings, dim = 0)

        # plucker_embedding = _get_plucker_embedding(
        #     self.h, self.w,
        #     torch.unsqueeze(intrinsic_mat, 0).repeat(self.num_frames, 1, 1),      # 3, 3 --> T, 3, 3
        #     extrinsics
        # )
        #
        # plucker_embedding = _get_plucker_embedding2(
        #     intrinsic=intrinsic,
        #     extrinsic_lst=list(map(lambda x: x.numpy(), extrinsics_lst)),
        #     h=self.h, w=self.w, t=self.num_frames,
        #     ori_h=ori_h, ori_w=ori_w
        # )
        intrinsics = torch.tensor([intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], 0, 0], dtype=torch.float32)

        if self.is_image:
            pixels = pixels[0]
            text = captions[0] + ',' + captions[0] + ',' + narration
        else:
            text = captions[0] + ',' + captions[1] + ',' + narration
        import einops
        pixels = einops.rearrange(pixels, 't c h w -> c t h w')
        return {
            'video': pixels,     # T, C, H, W
            'caption': text,     # str
            'frame_stride': self.sample_stride,
            'fps': 30.0 / self.sample_stride,
            'path': './' ,
            'intrinsics': intrinsics,       # 6,
            'extrinsics': extrinsics,       # T, 4, 4
            # 'plucker_embedding': plucker_embedding,     # T, 6, H, W
            'camera_embeddings': camera_embeddings,     # T, 12
        }

    

if __name__== "__main__":
    dataset = EpicKitchen(
        root="/workspace/CameraCtrl/Epic",
        h=320,
        w=512,
        sample_stride=4,
        num_frames=16,
    )
    
    dataloader = DataLoader(dataset,
                    batch_size=2,
                    num_workers=0,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/','_')
        tensor_to_mp4(video, save_dir+'/'+name, fps=8)
