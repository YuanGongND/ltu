# -*- coding: utf-8 -*-
# @Time    : 10/5/23 7:26 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : convert_pretrained_weights.py
#
# import sys
# argument = sys.argv[1]
# if argument=='4':
#     argument='0,1,2,3'
# import os
# if argument != '-1':
#     os.environ["CUDA_VISIBLE_DEVICES"]=argument

import os
import torch

def process_file(file_path):
    state_dict = torch.load(file_path)

    filtered_state_dict = {name: param for name, param in state_dict.items() if 'lora' in name or 'audio' in name}

    print(f"Parameters in file {file_path}:")
    for name in filtered_state_dict.keys():
        print(name)

    torch.save(filtered_state_dict, file_path[:-4] + '_trainable.bin')
    print(file_path[:-4] + '_trainable.bin')
    print('----------------------------------')

# Walk through the current directory and its subdirectories
count = 0
for dirpath, dirnames, filenames in os.walk('/data/sls/scratch/yuangong/audiollm/src/llm/alpaca-lora-main/'):
    for file in filenames:
        if file == "pytorch_model.bin":
            cur_target = os.path.join(dirpath, file)
            if os.path.exists(cur_target[:-4] + '_trainable.bin') == False:
                print(os.path.join(dirpath, file))
                process_file(os.path.join(dirpath, file))
                count +=1
print(count)