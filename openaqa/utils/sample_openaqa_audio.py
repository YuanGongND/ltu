# -*- coding: utf-8 -*-
# @Time    : 10/4/23 2:36 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : sample_openasqa_audio_feat.py

# copy audio and corresponding whisper features for release sample purpose

import json
import os

with open('/data/sls/scratch/yuangong/ltu/openaqa/data/openaqa_toy.json', 'r') as file:
    data = json.load(file)

# copy audio
for entry in data:
    cur_audio_id = entry['audio_id']
    #os.system('cp ' + cur_audio_id + ' /data/sls/scratch/yuangong/ltu/openaqa/data/sample_audio/audio/' + cur_audio_id.split('/')[-1])
    cur_audio_id = '../../../openaqa/data/sample_audio/audio/' + cur_audio_id.split('/')[-1]
    entry['audio_id'] = cur_audio_id

with open('/data/sls/scratch/yuangong/ltu/openaqa/data/openaqa_toy_relative.json', 'w') as outfile:
    json.dump(data, outfile, indent=1)
