# -*- coding: utf-8 -*-
# @Time    : 5/1/23 2:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : combine_json.py

import json
import os
import random
import re

def remove_thanks_for_watching(text):
    # Define the variations of the phrase to be removed
    variations = [
        "thanks for watching", "Thanks for watching", "THANKS FOR WATCHING",
        "thanks for watching.", "Thanks for watching.", "THANKS FOR WATCHING.",
        "thanks for watching!", "Thanks for watching!", "THANKS FOR WATCHING!",
        "thank you for watching", "Thank you for watching", "THANK YOU FOR WATCHING",
        "thank you for watching.", "Thank you for watching.", "THANK YOU FOR WATCHING.",
        "thank you for watching!", "Thank you for watching!", "THANK YOU FOR WATCHING!"
    ]

    variations = sorted(variations, key=len, reverse=True)

    # Create a regular expression pattern to match the variations
    pattern = "|".join(re.escape(var) for var in variations)

    # Remove the variations from the text
    result = re.sub(pattern, "", text)

    return result

def remove_tfw(data):
    empty_cnt = 0
    for i in range(len(data)):
        rm_tfw_input = remove_thanks_for_watching(data[i]['input'])
        if rm_tfw_input != data[i]['input']:
            # print(rm_tfw_input)
            # print(data[i]['input'])
            data[i]['input'] = rm_tfw_input
            empty_cnt += 1
    return empty_cnt, data

def check_no_text(data):
    empty_cnt = 0
    for entry in data:
        cur_input = entry['input']
        if cur_input == '':
            empty_cnt += 1
    return empty_cnt

def combine_json_list(dir_list, output_path):
    data = []
    for file_path in dir_list:
        print(file_path)
        with open(file_path, 'r') as infile:
            new_data = json.load(infile)
            empty_cnt = check_no_text(new_data)
            tfw_cnt, new_data = remove_tfw(new_data)
            print(len(new_data), empty_cnt/(len(new_data)), tfw_cnt/(len(new_data)))
            data = data + new_data
    print(len(data))
    random.shuffle(data)
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=1)
    print('Successfully concatenated {:d} JSON files into {:s} with {:d} qa pairs.'.format(len(dir_list), output_path, len(data)))

combine_json_list(['../data/openaqa_5.6M.json', # audio all, audio open and close 5.6m 5681708
                   '../data/closed_ended/combine_paralinguistic_whisper.json', # speech close 789466
                   '../data/closed_ended/fma_genre.json', # music close 92526  ## above 3 together = all_cla + audio open
                   '../data/open_ended/fma_train_val_whisper.json', # music open 395750
                   '../data/open_ended/combine_speech_open_whisper.json', # speech open 1784049
                   '../data/closed_ended/all_asr_old.json', # speech asr 151294
                   '../data/open_ended/joint_as_audio_speech_filter_no.json'], # audioset, audio and speech joint understanding 746947
                   '../data/openasqa_9.6M_v1.json') # mix speech open/close and original 8255223

# check leading " " space in text entry
datafile_list = ['../data/openasqa_9.6M_v1.json']
for datafile in datafile_list:
    print(datafile)
    with open(datafile, 'r') as infile:
        full_mix = json.load(infile)
    print(len(full_mix))

    empty_cnt, no_text_cnt = 0, 0
    for i in range(len(full_mix)):
        cur_input = full_mix[i]['input']
        if cur_input != '':
            if cur_input[0] == ' ':
                empty_cnt += 1
                full_mix[i]['input'] = full_mix[i]['input'].lstrip()
        else:
            no_text_cnt += 1

print(len(full_mix), empty_cnt/len(full_mix), no_text_cnt/len(full_mix))
with open('../data/openasqa_9.6M_v1.json', 'w') as outfile:
    json.dump(full_mix, outfile, indent=1)

# terminal print:
# ../data/openasqa_9.6M_v1.json
# 9641740
# 9641740 0.5940178847386468 0.10343091599649026