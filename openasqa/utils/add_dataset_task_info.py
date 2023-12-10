# -*- coding: utf-8 -*-
# @Time    : 10/2/23 6:13 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : add_dataset_task_info.py

# add information of dataset and task before release

import json

def add_info(json_path, task=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for d in data:
        filename = d['audio_id']

        if 'mosei' in filename:
            dataset = 'mosei'
        elif 'IEMOCAP' in filename:
            dataset = 'IEMOCAP'
        elif 'libritts' in filename:
            dataset = 'libritts'
        elif 'FMA/audio_16k' in filename:
            dataset = 'FMA'
        elif 'voxceleb' in filename:
            dataset = 'voxceleb'
        elif 'audioset' in filename:
            dataset = 'audioset'
        else:
            print('error')
            exit()

        d["dataset"] = dataset
        d["task"] = task

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=1)

add_info('../data/open_ended/fma_train_val_whisper.json', task='open-ended question')
add_info('../data/open_ended/combine_speech_open_whisper.json', task='open-ended question')
add_info('../data/closed_ended/all_asr.json', task='ASR')
add_info('../data/closed_ended/all_asr_old.json', task='ASR')
add_info('../data/open_ended/joint_as_audio_speech_filter_no.json', task='open-ended question')
add_info('../data/open_ended/joint_as_audio_speech_after_sub.json', task='open-ended question') # , 'audioset', 'open-ended question'