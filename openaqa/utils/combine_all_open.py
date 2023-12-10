# -*- coding: utf-8 -*-
# @Time    : 10/1/23 5:19 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_open.py

# combine all open-ended qa data

import json
import random

def combine_json(dir_list, output_path, shuffle=False):
    data = []
    for file_path in dir_list:
        with open(file_path, 'r') as infile:
            cur_data = json.load(infile)
            print(file_path, len(cur_data), 'samples')
            data = data + cur_data

    if shuffle == True:
        print('shuffle the data')
        random.shuffle(data)

    # Write the combined JSON data to the output file
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=1)
    print('Successfully concatenated {:d} JSON files into {:s} with {:d} qa pairs.'.format(len(dir_list), output_path, len(data)))

dir_list = ['/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/combine_as_strong_train_qa.json',  # audioset strong
            '/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/combine_as20k_fsd_qa.json', # audioset 20k + FSD50K
            '/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/combine_vggsound_qa.json', # VGGSound
            '/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/combine_clotho_qa.json', # Clotho
            '/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/combine_fs_sb_cap_all.json', # Freesound and Sound Bible
            '/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/combine_audiocaps_qa.json'] # AudioCaps

combine_json(dir_list=dir_list, output_path='/data/sls/scratch/yuangong/ltu/openaqa/data/open_ended/all_open_qa.json')


# dir_list = ['/data/sls/scratch/yuangong/audiollm/src/data/prep_data/all_info/datafiles/combine_open_formal.json',
#             '/data/sls/scratch/yuangong/audiollm/src/data/prep_data/temporal/datafiles/combine_cla_cap_temp_formal.json']
#
# # Path to the JSON file
# json_file = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data/temporal/datafiles/combine_cla_cap_temp_formal.json'
# with open(json_file, 'r') as f:
#     close_data = json.load(f)
# # Get the length of the Python object
# length = len(close_data)
# print(f"The Closed JSON file '{json_file}' has a length of {length}.")
#
# for i in range(len(close_data)):
#     if random.random() < 0.5:
#         close_data[i]['instruction'] = 'Closed-ended question: ' + close_data[i]['instruction']
#
# json_file = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data/all_info/datafiles/combine_open_formal.json'
# with open(json_file, 'r') as f:
#     open_data = json.load(f)
# length = len(open_data)
# print(f"The Open JSON file '{json_file}' has a length of {length}.")
#
# all_data = close_data + open_data
# random.shuffle(all_data)
# print(len(all_data))
#
# with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data/all_info/datafiles/combine_open_close_formal.json', 'w') as outfile:
#     json.dump(all_data, outfile, indent=1)
