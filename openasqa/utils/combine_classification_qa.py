# -*- coding: utf-8 -*-
# @Time    : 10/4/23 1:19 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_closed_ended_qa.py

# combine all closed ended qa, prepare for stage 1&2 training

import json
import random
from collections import Counter

datafile_list = ['combine_paralinguistic_whisper.json', 'openaqa_classification.json', 'fma_genre.json']

all_data = []
for datafile in datafile_list:
    with open('../data/closed_ended/' + datafile, 'r') as fp:
        data_json = json.load(fp)
    all_data= all_data + data_json

random.shuffle(all_data)

for i in range(len(all_data)):
    if 'yuangong/voxceleb' in all_data[i]['audio_id']:
        all_data[i]['audio_id'] = all_data[i]['audio_id'].replace('yuangong/voxceleb', 'yuangong/dataset/voxceleb')

output = all_data
print(len(output))
with open('../data/openasqa_classification.json', 'w') as f:
    json.dump(output, f, indent=1)

for attribute in ['dataset', 'task']:
    dataset_distribution = [x[attribute] for x in all_data]
    dataset_distribution = {key: {'count': value, 'percentage': '{:.1f}%'.format((value / len(dataset_distribution)) * 100)} for key, value in Counter(dataset_distribution).items()}
    print(dataset_distribution)

# terminal print
# 2072481
# {'voxceleb_tr': {'count': 536040, 'percentage': '25.9%'}, 'fsd50k_tr': {'count': 73592, 'percentage': '3.6%'}, 'as_2m': {'count': 500139, 'percentage': '24.1%'}, 'mosei_tr': {'count': 130616, 'percentage': '6.3%'}, 'vggsound_train': {'count': 367454, 'percentage': '17.7%'}, 'as_20k': {'count': 37382, 'percentage': '1.8%'}, 'fma': {'count': 92526, 'percentage': '4.5%'}, 'as_strong_train': {'count': 203582, 'percentage': '9.8%'}, 'libritts_tr': {'count': 86392, 'percentage': '4.2%'}, 'iemocap_tr': {'count': 21450, 'percentage': '1.0%'}, 'fsd50k_val': {'count': 8340, 'percentage': '0.4%'}, 'mosei_dev': {'count': 14968, 'percentage': '0.7%'}}
# {'energy_cla': {'count': 151294, 'percentage': '7.3%'}, 'gender_cla': {'count': 150449, 'percentage': '7.3%'}, 'cla_label': {'count': 345175, 'percentage': '16.7%'}, 'cla_label_des': {'count': 845314, 'percentage': '40.8%'}, 'age_cla': {'count': 107208, 'percentage': '5.2%'}, 'speed_cla': {'count': 151294, 'percentage': '7.3%'}, 'genre_cla': {'count': 92526, 'percentage': '4.5%'}, 'pitch_cla': {'count': 151294, 'percentage': '7.3%'}, 'emotion_detail': {'count': 18198, 'percentage': '0.9%'}, 'emotion_score': {'count': 18198, 'percentage': '0.9%'}, 'emotion_cla': {'count': 22488, 'percentage': '1.1%'}, 'emotion_bin': {'count': 18198, 'percentage': '0.9%'}, 'gender_emotion_cla': {'count': 845, 'percentage': '0.0%'}}