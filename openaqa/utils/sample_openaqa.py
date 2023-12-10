# -*- coding: utf-8 -*-
# @Time    : 10/4/23 2:36 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : sample_openasqa.py

# sample a mini toy subset from the openasqa (1000 audios) for sample purpose

import json
import random

with open('/data/sls/scratch/yuangong/ltu/openaqa/data/openaqa_5.6M.json', 'r') as file:
    data = json.load(file)

all_audio_id = [x['audio_id'] for x in data]
all_audio_id = list(set(all_audio_id))

print('total {:d} audios'.format(len(all_audio_id)))

sample_cnt = 1000
sampled_list = random.sample(all_audio_id, sample_cnt)

sample_data = [entry for entry in data if entry["audio_id"] in sampled_list]

print('sampled {:d} QAs'.format(len(sample_data)))

with open('/data/sls/scratch/yuangong/ltu/openaqa/data/openaqa_toy.json', 'w') as outfile:
    json.dump(sample_data, outfile, indent=1)

# terminal print:
# total 1089204 audios
# sampled 8769 QAs