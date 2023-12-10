# -*- coding: utf-8 -*-
# @Time    : 6/30/23 3:43 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : eval_iemocap_cla.py

import os.path
import datetime
current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y%m%d%H%M%S")
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score

def get_immediate_files_with_name(directory, search_string):
    files = []
    for file_path in os.listdir(directory):
        if os.path.isfile(directory+'/'+file_path) and search_string in file_path:
            files.append(file_path)
    return files

def gender_cla(ref, pred):
    ref, pred = ref.lower(), pred.lower()
    if 'female' in ref and 'female' in pred:
        return True
    elif ('female' not in ref) and ('female' not in pred):
        return True
    else:
        return False

def gender_cla2(ref, pred):
    ref, pred = ref.lower(), pred.lower()
    if 'female' in ref:
        ref = 0
    else:
        ref = 1
    if 'female' in pred:
        pred = 0
    else:
        pred = 1
    return ref, pred

dataset = 'voxceleb'
llm_task_list = ['gender']

base_path = '/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/raw/'

all_res = []
for task in llm_task_list:
    eval_file_list = get_immediate_files_with_name(base_path, 'voxceleb_' + task)
    eval_file_list = [base_path + x for x in eval_file_list]

    eval_file_list.sort()

    eval_file_list = ['/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/raw/voxceleb_gender_formal_audio_lora_mix_from_close_from_e2e_cla_from_proj_checkpoint-20000_gender_cla_0.10_0.95_500.json',
                      '/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/raw/voxceleb_gender_formal_speech_all_open_close_final_checkpoint-35000_gender_cla_fp16_joint_3.json']

    for eval_file in eval_file_list:
        all_pred, all_ref = [], []
        print(eval_file)
        with open(eval_file) as json_file:
            data = json.load(json_file)

        num_sample = len(data)
        num_correct = 0

        for item in data:
            if task == 'gender':
                ref, pred = gender_cla2(item['ref'], item['pred'])
                all_pred.append(pred)
                all_ref.append(ref)

        sk_acc = accuracy_score(all_ref, all_pred)
        print('gender accuracy: ', sk_acc)

        f1 = f1_score(all_ref, all_pred, average='macro')
        print('gender f1: ', f1)

        print('{:s}, {:s}, {:.3f}'.format(eval_file.split('/')[-1].split('.')[0], task, sk_acc, f1, len(all_pred)))
        all_res.append([eval_file.split('/')[-1].split('.')[0], task, sk_acc, f1, len(all_pred)])
        print(classification_report(all_ref, all_pred))

np.savetxt('/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/summary/summary_voxceleb_gender_{:s}.csv'.format(time_string), all_res, delimiter=',', fmt='%s')