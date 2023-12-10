# -*- coding: utf-8 -*-
# @Time    : 10/2/23 2:03 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_all_close.py

# combine all closed-ended qas, including caption, classfication, and temporal

import json
import random
from collections import Counter


datafile_list = ['../data/closed_ended/combine_caption.json',
                 '../data/closed_ended/combine_cla.json',
                 '../data/closed_ended/combine_temporal.json']

all_data = []
for datafile in datafile_list:
    with open(datafile, 'r') as fp:
        data_json = json.load(fp)
    all_data= all_data + data_json

random.shuffle(all_data)

output = all_data
print(len(output))
with open('../data/closed_ended/all_closed_qa.json', 'w') as f:
    json.dump(output, f, indent=1)

for attribute in ['dataset', 'task']:
    dataset_distribution = [x[attribute] for x in all_data]
    dataset_distribution = {key: {'count': value, 'percentage': '{:.1f}%'.format((value / len(dataset_distribution)) * 100)} for key, value in Counter(dataset_distribution).items()}
    print(dataset_distribution)