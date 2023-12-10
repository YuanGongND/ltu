# -*- coding: utf-8 -*-
# @Time    : 10/1/23 5:43 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_all_classification.py

# combine closed-ended classification and classification + acoustic feature description qa for all datasets

import json
import random
from collections import Counter

def sample_dataset(json_data, ratio):
    total_entries = len(json_data)
    num_entries_to_sample = int(total_entries * ratio)
    random_indices = random.sample(range(total_entries), num_entries_to_sample)
    sampled_entries = [json_data[i] for i in random_indices]
    return sampled_entries

datafile_list = ['as_20k.json', 'as_strong_train.json', 'fsd50k_tr.json', 'fsd50k_val.json', 'vggsound_train.json', 'as_500k.json']

all_data = []
for datafile in datafile_list:
    with open('../data/closed_ended/classification/' + datafile, 'r') as fp:
        data_json = json.load(fp)
    # if datafile == 'vggsound_train.json':
    #     data_json = sample_dataset(data_json, 0.5) # to roughly balance
    all_data= all_data + data_json

random.shuffle(all_data)

output = all_data
print(len(output))
with open('../data/closed_ended/combine_cla.json', 'w') as f:
    json.dump(output, f, indent=1)

for attribute in ['dataset', 'task']:
    dataset_distribution = [x[attribute] for x in all_data]
    dataset_distribution = {key: {'count': value, 'percentage': '{:.1f}%'.format((value / len(dataset_distribution)) * 100)} for key, value in Counter(dataset_distribution).items()}
    print(dataset_distribution)