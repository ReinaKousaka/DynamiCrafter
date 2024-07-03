import os
import json

with open('/workspace/DynamiCrafter/Epic/P36_102_ex.json') as f:
    data = json.load(f)
    print(data.keys())