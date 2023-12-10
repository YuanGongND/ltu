# -*- coding: utf-8 -*-
# @Time    : 7/19/23 1:11 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : filter_no.py

# filter out qas that the answer is "based on the given information, it cannot be determined ..." etc

import json

def filter_no(file_path, output_path):

    with open(file_path, 'r') as infile:
        data = json.load(infile)

    no_word_list = ['cannot determine', 'not provided', 'cannot be determined', 'sorry', 'i cannot',
                    'without more information', 'enough information',
                    'not possible', 'more context', 'enough', 'impossible', 'cannot be determined',
                    'without additional information',
                    'unclear', 'cannot', 'not clear', 'do not provide sufficient', 'does not provide',
                    'difficult to determine', 'no information provided',
                    "can't infer", "difficult to infer", "not specified", "no specific", "no information",
                    "without additional", 'it is difficult to',
                    "no indication"]

    certain_data = []
    for x in data:
        if any(word in x['output'].lower() for word in no_word_list):
            pass
        else:
            certain_data.append(x)

    print('{:d} QAs are kept, which is {:.2f}% of the original data.'.format(len(certain_data), len(certain_data)*100/len(data)))
    with open(output_path, 'w') as outfile:
        json.dump(certain_data, outfile, indent=1)

filter_no('/data/sls/scratch/yuangong/ltu/openasqa/data/openasqa_10.3M_v2.json', '/data/sls/scratch/yuangong/ltu/openasqa/data/openasqa_9.4M_v2_noqa.json')