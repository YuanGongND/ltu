# -*- coding: utf-8 -*-
# @Time    : 4/10/23 3:09 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : eval_llm.py

# install this package: https://github.com/audio-captioning/caption-evaluation-tools and place this script in the same dir
# evaluate the llm output

import json
import os.path
import string
import numpy as np
from eval_metrics import evaluate_metrics
import datetime
current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y%m%d%H%M%S")

def remove_punctuation_and_lowercase(text):
    """
    This function takes a string as input, removes all the punctuations,
    and converts the string to lowercase.
    """
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

directory = '/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/raw/'
prefix = 'audiocaps'
files = os.listdir(directory)
eval_file_list = [os.path.join(directory, file) for file in files if file.startswith(prefix)]
eval_file_list.sort()
print(eval_file_list)

result_summary = []
for eval_file in eval_file_list:
    try:
        with open(eval_file, 'r') as fp:
            eval_data = json.load(fp)

        print(eval_data[0]['audio_id'])
        print(eval_data[0]['pred'].split(':')[-1][1:])
        print(eval_data[0]['ref'].split(':')[-1][1:])

        num_sample = len(eval_data)
        print('number of samples {:d}'.format(num_sample))
        pred_dict = {}
        truth_dict = {}
        for i in range(num_sample):
            cur_audio_id = eval_data[i]['audio_id'].split('/')[-1]
            cur_pred = remove_punctuation_and_lowercase(eval_data[i]['pred'].split(':')[-1][1:])
            cur_truth = remove_punctuation_and_lowercase(eval_data[i]['ref'].split(':')[-1][1:])
            if cur_audio_id in pred_dict.keys():
                pred_dict[cur_audio_id].append(cur_pred)
                truth_dict[cur_audio_id].append(cur_truth)
            else:
                pred_dict[cur_audio_id] = [cur_pred]
                truth_dict[cur_audio_id] = [cur_truth]

        save_fold = "/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/{:s}_report".format('.'.join(eval_file.split('/')[-1].split('.')[:-1]))
        if os.path.exists(save_fold) == False:
            os.mkdir(save_fold)

        ciders, spices = [], []
        for trial in range(5):
            all_pred = [['file_name','caption_predicted']]
            all_truth = [['file_name','caption_1','caption_2','caption_3','caption_4','caption_5']]
            for key in pred_dict.keys():
                cur_audio_id = key
                cur_pred = pred_dict[key][trial]
                cur_truth = truth_dict[key]
                all_pred.append([cur_audio_id, cur_pred])
                all_truth.append([cur_audio_id] + cur_truth)

            np.savetxt(save_fold + '/sample_pred_audiocaps_{:d}.csv'.format(trial), all_pred, fmt='%s', delimiter=',')
            np.savetxt(save_fold + '/sample_truth_audiocaps_{:d}.csv'.format(trial), all_truth, fmt='%s', delimiter=',')

            res = evaluate_metrics(save_fold + '/sample_pred_audiocaps_{:d}.csv'.format(trial), save_fold + '/sample_truth_audiocaps_{:d}.csv'.format(trial), 5)
            ciders.append(res['cider']['score'])
            spices.append(res['spice']['score'])

            with open(save_fold + "/res_summary_{:d}.json".format(trial), "w") as f:
                json.dump(res, f)

        result_summary.append([eval_file, np.mean(ciders), np.mean(spices)])
        np.savetxt('./all_acaps_summary_ablation.csv', result_summary, delimiter=',', fmt='%s')
        ciders = ciders + [np.mean(ciders), np.std(ciders)]
        spices = spices + [np.mean(spices), np.std(spices)]
        np.savetxt(save_fold + "/ciders_summary.csv", ciders, delimiter=',')
        np.savetxt(save_fold + "/spices_summary.csv", spices, delimiter=',')

        np.savetxt('/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/summary/summary_audiocaps_{:s}.csv'.format(time_string), result_summary, delimiter=',', fmt='%s')
    except:
        pass
