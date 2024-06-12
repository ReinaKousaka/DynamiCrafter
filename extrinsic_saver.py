import json
import os


video_id='P36_102'

with open('/workspace/DynamiCrafter/Epic/all_pose.json', 'r') as f:
    data = json.load(f)
    res = {
        video_id: data[video_id]
    }

with open(f'/workspace/DynamiCrafter/Epic/{video_id}_ex.json', 'w+') as f:
    json.dump(res, f)
