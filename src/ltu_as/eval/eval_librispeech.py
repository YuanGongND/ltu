# -*- coding: utf-8 -*-
# @Time    : 7/14/23 3:47 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : eval_librispeech.py

import os.path
import datetime
current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y%m%d%H%M%S")
import json

import os
import editdistance
import jiwer
import numpy as np
import re

def remove_parentheses_content(s):
    # This regular expression finds content within parentheses
    return re.sub(r'\s?\(.*?\)', '', s)

def gen_filename(filename):
    ext = filename.split('.')[-1]
    filename = filename.split('/')[-1][:-len(ext)-1]
    return filename

def load_transcript(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return data['text'].lstrip()

def get_immediate_files_with_name(directory, search_string):
    files = []
    for file_path in os.listdir(directory):
        if os.path.isfile(directory+'/'+file_path) and search_string in file_path:
            files.append(file_path)
    return files

def process_text(text):
    text = text.lower()
    text = text.split('spoken text: ')[-1].replace('spoken text:','').lstrip()
    return text

def calculate_wer(seqs_hat, seqs_true):
    """Calculate sentence-level WER score.
    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level WER score
    :rtype float
    """
    word_eds, word_ref_lens = [], []
    for i in range(len(seqs_true)):
        seq_true_text = seqs_true[i]
        seq_hat_text = seqs_hat[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)

def preprocess_text(cur_trans):
    cur_trans = jiwer.ToUpperCase()(cur_trans)
    cur_trans = jiwer.RemovePunctuation()(cur_trans)
    return cur_trans

base_path = '/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/raw/'
eval_file_list = get_immediate_files_with_name(base_path, 'librispeech_formal_speech_all_open_close_final_checkpoint-35000_asr_fp16_joint_3.json')
eval_file_list = [base_path + x for x in eval_file_list]
eval_file_list.sort()
print(eval_file_list)

all_res = []
for eval_file in eval_file_list:
    gt_list, whisper_list, ltue_list, ori_ltue_list = [], [], [], []
    print(eval_file)
    with open(eval_file) as json_file:
        data = json.load(json_file)

    num_sample = len(data)
    num_correct = 0

    for item in data:
        cur_id = item['audio_id']
        cur_whisper_trans = load_transcript('/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/data/transcribe_as/librispeech/' + gen_filename(cur_id) + '.json')
        gt, ltue_o, whisper_o = preprocess_text(process_text(item['ref'])), preprocess_text(process_text(item['pred'])), preprocess_text(process_text(cur_whisper_trans))
        if whisper_o in ltue_o:
            ltue_list.append(whisper_o)
        else:
            ltue_list.append(ltue_o)
            print(item['pred'])
            print(ltue_o)
            print(whisper_o)
            print('----------------')
        gt_list.append(gt)
        whisper_list.append(whisper_o)
        ori_ltue_list.append(ltue_o)

    wer_whisper = calculate_wer(whisper_list, gt_list)
    wer_ltue = calculate_wer(ltue_list, gt_list)
    # no postprocessed
    wer_ltue_ori = calculate_wer(ori_ltue_list, gt_list)
    print('wer is ', wer_whisper, wer_ltue, wer_ltue_ori)

    all_res.append([eval_file, wer_whisper, wer_ltue, wer_ltue_ori, len(gt_list)])
    np.savetxt('/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/eval_res/summary/summary_librispeech_{:s}.csv'.format(time_string), all_res, delimiter=',', fmt='%s')