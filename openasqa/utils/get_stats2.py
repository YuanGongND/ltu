# -*- coding: utf-8 -*-
# @Time    : 10/2/23 3:58 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_stats.py

import json
from collections import Counter

combine_json_list = ['../data/openaqa_5.6M.json', # audio all, audio open and close 5.6m 5681708
                   '../data/closed_ended/combine_paralinguistic_whisper.json', # speech close 789466
                   '../data/closed_ended/fma_genre.json', # music close 92526  ## above 3 together = all_cla + audio open
                   '../data/open_ended/fma_train_val_whisper.json', # music open 395750
                   '../data/open_ended/combine_speech_open_whisper.json', # speech open 1784049
                   '../data/closed_ended/all_asr.json', # speech asr 151294, renewed question after submission
                   '../data/open_ended/joint_as_audio_speech_filter_no.json',# audioset, audio and speech joint understanding 746947
                   '../data/open_ended/joint_as_audio_speech_after_sub.json']


for datafile in combine_json_list:
    print(datafile)
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
        all_keys = [key for d in data if isinstance(d, dict) for key in d.keys()]

        all_keys = list(set(all_keys))

        # Print the results
        print("Non-repeated keys:", all_keys)

    else:
        print("The loaded data is not a list.")