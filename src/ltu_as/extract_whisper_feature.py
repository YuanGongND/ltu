# -*- coding: utf-8 -*-
# @Time    : 7/9/23 3:56 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_whisper_feature.py

# Please extract whisper features of all audios on SAME type of gpus.
# Different types of GPUs will generate slightly different results, which does impact the LTU-AS performance.
# The extraction GPU should also be the same type as the server/inference GPU.

# extract representation for all layers for whisper model, pool by 20, not include the input mel.
# save as npz to save space
# extract first 10s

# conda ltue is fine
import sys
argument = sys.argv[1]
if argument=='-1':
    gpu_argument='0,1,2,3'
else:
    gpu_argument = str(int(argument) % 4)
import os
if argument != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_argument

import json
import torch
os.environ["XDG_CACHE_HOME"] = './'
import numpy as np
from whisper.model import Whisper, ModelDimensions
import skimage.measure
import math

def gen_filename(filename):
    ext = filename.split('.')[-1]
    filename = '_'.join(filename.split('/')[-4:-1]) + '_' + filename.split('/')[-1][:-len(ext) - 1]
    return filename

def extract_audio(dataset_json_file, mdl, tar_path, total_split=16):
    if os.path.exists(tar_path) == False:
        os.makedirs((tar_path))
    with open(dataset_json_file, 'r') as fp:
        data = json.load(fp)
        num_sample = len(data)
        num_each_split = math.ceil(num_sample / total_split)
        cur_start = int(argument) * num_each_split
        data = data[cur_start:cur_start + num_each_split]
        print(cur_start, len(data))

        for idx, entry in enumerate(data):
            wav = entry

            if os.path.exists(tar_path + '/' + gen_filename(wav) + '.npz') == False:
                try:
                    # for new GPUs, e.g., a5000, a6000, etc
                    _, audio_rep = mdl.transcribe_audio(wav)
                    audio_rep = audio_rep[0]

                    # # for old GPUs, e.g., titan
                    # _, audio_rep = mdl.transcribe_audio(wav, fp16=False)
                    # audio_rep = audio_rep[0].half()

                    audio_rep = torch.permute(audio_rep, (2, 0, 1)).detach().cpu().numpy()
                    audio_rep = skimage.measure.block_reduce(audio_rep, (1, 20, 1), np.mean)
                    audio_rep = audio_rep[1:] # skip the first layer
                except Exception as e:
                    print(f"Error loading file {e}")
                    audio_rep = torch.zeros((32, 25, 1280))

                np.savez_compressed(tar_path + '/' + gen_filename(wav) + '.npz', audio_rep)
            if idx % 50 == 0:
                print(idx)

mdl_size_list = ['large-v1']
for mdl_size in mdl_size_list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path = '/data/sls/scratch/yuangong/whisper-a/src/{:s}.pt'.format(mdl_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    tar_path = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/whisper_feat/openasqa/' + 'whisper_' + mdl_size + '/'
    esc_train1 = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/all_mix_disjoint_audio_list.json'
    extract_audio(esc_train1, model, tar_path)
    del model