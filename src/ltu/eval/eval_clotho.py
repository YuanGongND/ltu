# -*- coding: utf-8 -*-
# @Time    : 4/10/23 3:09 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : eval_llm.py

# install this package: https://github.com/audio-captioning/caption-evaluation-tools and place this script in the same dir
# evaluate the llm output

import os
import json
import string
import numpy as np
from eval_metrics import evaluate_metrics

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

eval_file_list = [
'clotho_formal_audio_lora_mix_from_proj_fz_audio_encoder_checkpoint-22000_clotho_cap_0.10_0.95_500_repp110.json',
'clotho_formal_audio_lora_close_from_e2e_cla_from_proj_checkpoint-6000_clotho_cap_0.10_0.95_500_repp110.json',
'clotho_formal_audio_lora_mix_no_corr_cont_checkpoint-22000_clotho_cap_0.10_0.95_500_repp110.json',
'clotho_formal_audio_lora_mix_from_proj_fz_no_adapter_checkpoint-22000_clotho_cap_0.10_0.95_500_repp110.json']

eval_file_list = ['/data/sls/scratch/yuangong/audiollm/src/llm/alpaca-lora-main/eval_res/' + x for x in eval_file_list]

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

        save_fold = eval_file[:-5]
        if os.path.exists(save_fold) == False:
            os.mkdir(save_fold)

        ciders, spices = [], []
        for trial in range(5):
            all_pred = [['file_name', 'caption_predicted']]
            all_truth = [['file_name', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']]
            for key in pred_dict.keys():
                cur_audio_id = key
                cur_pred = pred_dict[key][trial]
                cur_truth = truth_dict[key]
                all_pred.append([cur_audio_id, cur_pred])
                all_truth.append([cur_audio_id] + cur_truth)

            print(len(all_pred), len(all_truth))
            np.savetxt(save_fold + '/clotho_pred_{:d}.csv'.format(trial), all_pred, fmt='%s', delimiter=',')
            np.savetxt(save_fold + '/clotho_truth_{:d}.csv'.format(trial), all_truth, fmt='%s', delimiter=',')

            res = evaluate_metrics(save_fold + '/clotho_pred_{:d}.csv'.format(trial), save_fold + '/clotho_truth_{:d}.csv'.format(trial), 5)
            ciders.append(res['cider']['score'])
            spices.append(res['spice']['score'])

            with open(save_fold + "/res_summary_{:d}.json".format(trial), "w") as f:
                json.dump(res, f)

        result_summary.append([eval_file, np.mean(ciders), np.mean(spices)])
        np.savetxt('./all_clotho_summary_ablation.csv', result_summary, delimiter=',', fmt='%s')
        ciders = ciders + [np.mean(ciders), np.std(ciders)]
        spices = spices + [np.mean(spices), np.std(spices)]

        np.savetxt(save_fold + "/ciders_summary.csv", ciders, delimiter=',')
        np.savetxt(save_fold + "/spices_summary.csv", spices, delimiter=',')

    except:
        pass