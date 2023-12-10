# -*- coding: utf-8 -*-
# @Time    : 10/2/23 3:58 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_stats.py

import json
from collections import Counter

datafile = '/data/sls/scratch/yuangong/ltu/openaqa/data/openaqa_5.6M.json'
with open(datafile, 'r') as f:
    data = json.load(f)

# Verify that the data is a list
if isinstance(data, list):
    print(f"Number of dictionaries in the list: {len(data)}")

    # Use Counter to count the number of entries in each dictionary
    entry_counts = Counter(len(item) for item in data if isinstance(item, dict))

    for entries, count in entry_counts.items():
        print(f"{count} dictionaries have {entries} entries.")
else:
    print("The loaded data is not a list.")

if isinstance(data, list):

    # Gather all keys from dictionaries
    all_keys = [key for d in data if len(d)==6 for key in d.keys()]

    all_keys = list(set(all_keys))

    # Print the results
    print("Non-repeated keys:", all_keys)

else:
    print("The loaded data is not a list.")