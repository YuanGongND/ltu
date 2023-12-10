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
def get_immediate_files_with_name(directory, search_string):
    files = []
    for file_path in os.listdir(directory):
        if os.path.isfile(directory+'/'+file_path) and search_string in file_path:
            files.append(file_path)
    return files

import re
import statistics

def extract_and_process_number(string):
    pattern = r'\b(\d+(?:-\d+)?)\b'
    matches = re.findall(pattern, string)

    def process_number_range(number):
        if '-' in number:
            start, end = map(int, number.split('-'))
            return statistics.mean([start, end])
        else:
            return int(number)

    processed_result = [process_number_range(n) for n in matches]
    return processed_result

def age_reg(ref, pred):
    ref = int(ref.split(' ')[-1][:-1])
    if len(extract_and_process_number(pred)) != 0:
        pred = extract_and_process_number(pred)[0]
        return ref, pred
    else:
        return None, None

dataset = 'voxceleb'
llm_task_list = ['age']

base_path = '/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/raw/'

all_res = []
for task in llm_task_list:
    eval_file_list = get_immediate_files_with_name(base_path, 'voxceleb_' + task)
    eval_file_list = [base_path + x for x in eval_file_list]

    eval_file_list.sort()

    for eval_file in eval_file_list:
        all_pred, all_ref = [], []
        sq_err_list = []
        print(eval_file)
        with open(eval_file) as json_file:
            data = json.load(json_file)

        num_sample = len(data)
        num_correct = 0

        for item in data:
            if task == 'age':
                ref, pred = age_reg(item['ref'], item['pred'])
                if ref != None:
                    cur_sq_err = abs(ref-pred)
                    print(ref, pred, cur_sq_err)
                    sq_err_list.append(cur_sq_err)
                    all_ref.append(ref)
                    all_pred.append(pred)

        corr = np.corrcoef(all_ref, all_pred)[0, 1]
        print('{:d}/{:d} predicted, mse={:.3f}'.format(len(sq_err_list), len(all_ref), np.mean(sq_err_list), corr))
        all_res.append([eval_file.split('/')[-1].split('.')[0], len(sq_err_list), len(all_ref), np.mean(sq_err_list), corr])

np.savetxt('/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/summary/summary_voxceleb_age_{:s}.csv'.format(time_string), all_res, delimiter=',', fmt='%s')