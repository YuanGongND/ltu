# -*- coding: utf-8 -*-
# @Time    : 10/1/23 5:58 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_temporal.py

# combine closed-ended temporal qa for all datasets

import json
import random
from collections import Counter


datafile_list = ['as_strong_temporal_order_train.json', 'as_strong_temporal_single_train.json', 'as_strong_temporal_train.json']

all_data = []
for datafile in datafile_list:
    with open('../data/closed_ended/temporal/' + datafile, 'r') as fp:
        data_json = json.load(fp)
    all_data= all_data + data_json

random.shuffle(all_data)

output = all_data
print(len(output))
with open('../data/closed_ended/combine_temporal.json', 'w') as f:
    json.dump(output, f, indent=1)

for attribute in ['dataset', 'task']:
    dataset_distribution = [x[attribute] for x in all_data]
    dataset_distribution = {key: {'count': value, 'percentage': '{:.1f}%'.format((value / len(dataset_distribution)) * 100)} for key, value in Counter(dataset_distribution).items()}
    print(dataset_distribution)