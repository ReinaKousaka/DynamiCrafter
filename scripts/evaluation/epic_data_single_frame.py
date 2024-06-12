import os
import json
import torch


def index_to_keystring(index):
    return f'frame_{str(index).zfill(10)}.jpg'

def get_epic_data_single_frame(video_id, frame: int, data_dir='/workspace/DynamiCrafter/Epic'):
    with open(os.path.join(data_dir, 'intrinsics.json')) as f:
        data = json.load(f)
        intrinsic = tuple(data[video_id])

    with open(os.path.join('/workspace/DynamiCrafter/Epic', f'{video_id}_ex.json')) as f:       # very big file
        frame_to_ex = json.load(f)
        filename = index_to_keystring(frame)
        extrinsics = torch.tensor(frame_to_ex[video_id][f'{video_id}/{filename}']).float()
    
    with open(os.path.join(data_dir, 'caption_merged', video_id + '.json')) as f:
        data = json.load(f)
         # round frames to existing keys
        shift = 0
        while (index_to_keystring(frame + shift) not in data) and (index_to_keystring(frame - shift) not in data):
            shift += 1
        if index_to_keystring(frame + shift) in data:
            text = data[index_to_keystring(frame + shift)]
        else:
            text = data[index_to_keystring(frame - shift)]

    print(intrinsic, extrinsics.shape, text)

    return {
        'intrinsics': intrinsic,       # 6
        'extrinsics': extrinsics,       # 4, 4
        'text': text,
    }




if __name__ == '__main__':
    start_frame = 980
    video_id = 'P36_102'
    res = get_epic_data_single_frame(video_id, start_frame)
    print(res)
    torch.save(res, f'camera_data/{video_id}_{index_to_keystring(start_frame)}.pth')
    print(f'camera_data/{video_id}_{index_to_keystring(start_frame)}.pth saved')
