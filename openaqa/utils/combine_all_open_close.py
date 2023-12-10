# -*- coding: utf-8 -*-
# @Time    : 10/2/23 2:22 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_all_open_close.py

# combine everything of the openaqa-5.6M dataset

import json
import random

json_file = '../data/closed_ended/all_closed_qa.json'
with open(json_file, 'r') as f:
    close_data = json.load(f)
# Get the length of the Python object
length = len(close_data)
print(f"The Closed JSON file '{json_file}' has a length of {length}.")

for i in range(len(close_data)):
    if random.random() < 0.5:
        close_data[i]['instruction'] = 'Closed-ended question: ' + close_data[i]['instruction']

json_file = '../data/open_ended/all_open_qa.json'
with open(json_file, 'r') as f:
    open_data = json.load(f)
length = len(open_data)
print(f"The Open JSON file '{json_file}' has a length of {length}.")

all_data = close_data + open_data
random.shuffle(all_data)
print(len(all_data))

with open('../data/openaqa_5.6M.json', 'w') as outfile:
    json.dump(all_data, outfile, indent=1)

# The Open JSON file '../data/open_ended/all_open_qa.json' has a length of 3763814.
# 5681708