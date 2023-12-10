# -*- coding: utf-8 -*-
# @Time    : 10/1/23 5:48 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_all_caption.py

# combine closed-ended captioning qa for all datasets

import json
import random
from collections import Counter

def sample_dataset(json_data, ratio):
    if ratio < 1:
        total_entries = len(json_data)
        num_entries_to_sample = int(total_entries * ratio)
        random_indices = random.sample(range(total_entries), num_entries_to_sample)
        sampled_entries = [json_data[i] for i in random_indices]
        return sampled_entries
    else:
        num_duplicates = int(len(json_data) * (ratio - 1))
        duplicates = random.choices(json_data, k=num_duplicates)
        json_data = json_data + duplicates
        return json_data

datafile_list = ['audio_caps_train_label_caption.json', 'audio_caps_val_label_caption.json', 'clotho_caption_validation.json', 'clotho_caption_development.json',
                 'freesound_10s_caption.json', 'sound_bible_caption.json', 'as_strong_label_caption_train.json']

all_data = []
for datafile in datafile_list:
    with open('../data/closed_ended/caption/' + datafile, 'r') as fp:
        data_json = json.load(fp)
    if datafile == 'clotho_caption_development.json' or datafile == 'clotho_caption_validation.json':
        data_json = sample_dataset(data_json, 2)
    if datafile == 'sound_bible_caption.json':
        data_json = sample_dataset(data_json, 10)
    all_data= all_data + data_json

random.shuffle(all_data)

for entry in all_data:
    assert len(entry) == 6

output = all_data
print(len(output))
with open('../data/closed_ended/combine_caption.json', 'w') as f:
    json.dump(output, f, indent=1)

for attribute in ['dataset', 'task']:
    dataset_distribution = [x[attribute] for x in all_data]
    dataset_distribution = {key: {'count': value, 'percentage': '{:.1f}%'.format((value / len(dataset_distribution)) * 100)} for key, value in Counter(dataset_distribution).items()}
    print(dataset_distribution)