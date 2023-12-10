# -*- coding: utf-8 -*-
# @Time    : 10/4/23 2:36 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : sample_openasqa_audio_feat.py

# copy audio and corresponding whisper features for release sample purpose

import json
import os

with open('/data/sls/scratch/yuangong/ltu/openasqa/data/openasqa_toy.json', 'r') as file:
    data = json.load(file)

# # copy audio
# for entry in data:
#     cur_audio_id = entry['audio_id']
#     os.system('cp ' + cur_audio_id + ' /data/sls/scratch/yuangong/ltu/openasqa/data/sample_audio_feat/audio/' + cur_audio_id.split('/')[-1])

# # copy whisper feature extracted by nvidia titan gpus
# for entry in data:
#     filename = entry['audio_id']
#     ext = filename.split('.')[-1]
#     filename = '_'.join(filename.split('/')[-4:-1]) + '_' + filename.split('/')[-1][:-len(ext) - 1]
#     filename = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/whisper_feat/openasqa/whisper_large-v1/' + filename + '.npz'
#     # return filename
#     os.system('cp ' + filename + ' /data/sls/scratch/yuangong/ltu/openasqa/data/sample_audio_feat/whisper_feat_titan/' + filename.split('/')[-1])

# copy whisper feature extracted by nvidia a6 gpus
for entry in data:
    filename = entry['audio_id']
    ext = filename.split('.')[-1]
    filename = '_'.join(filename.split('/')[-4:-1]) + '_' + filename.split('/')[-1][:-len(ext) - 1]
    filename = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/whisper_feat/openasqa_a6/whisper_large-v1/' + filename + '.npz'
    # return filename
    os.system('cp ' + filename + ' /data/sls/scratch/yuangong/ltu/openasqa/data/sample_audio_feat/whisper_feat_a6/' + filename.split('/')[-1])