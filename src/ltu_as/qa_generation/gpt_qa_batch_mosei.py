# -*- coding: utf-8 -*-
# @Time    : 4/29/23 3:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gpt_qa.py

import openai
import numpy as np
import json
import re
import time
import pickle
import sys
split_id = sys.argv[1]
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# by combining speech content, fundamental frequency, speed, and volume information
sys_prompt = """Based on the following speech, generate 20 different types of complex open-ended questions that require step-by-step thinking, and corresponding answers. 
Only ask questions about the given speech emotion. No general/background questions. Do not use the given information in the question. 
Ask short, crispy, complex, and diverse questions. Answers need to be longer than 10 words.
Only ask questions that have a clear answer based on the given speech. The questions need to be of different types.
Questions can be e.g., What's the emotion of the speaker; How emotion is inferred from the speech content, f0, speed, and energy; What can be inferred from speech content and emotion and why; How speech content is related to the emotion and why; What is the intent and implicit meaning of the speech and why; What is the potential scenario that the speech could happen and why; If the speech is special and why; what mood the speech conveys, etc. 
Format each QA pair in a single line as a JSON dictionary (key "q" for question, and "a" for answer, wrapped with { and }). Do not include any other explanation."""

def generate_prompt(entry):
    prompt = """Speech content: "{:s}"; Speaker gender: {:s}; Speaker's fundamental frequency (F0) is about {}Hz, so the pitch is {:s} among all people, and is {:s} for a {:s} speaker; Speech volume: {:s}; Speech speed: {:s}; Speech emotion: {:s}. On a scale ranging from highly negative (-3) to highly positive (3), the emotional rating of this speech is {:s}, which is {:s}."""
    emotion_score_dict = {-3: 'highly negative', -2: 'negative', -1: 'weakly negative', 0: 'neutral', 1: 'weakly positive', 2: 'positive', 3: 'highly positive'}
    def gender_f0_rank(pitch, gender):
        if gender == 'male':
            f0_percentiles = [95, 120, 135, 180]
        elif gender == 'female':
            f0_percentiles = [160, 200, 220, 270]
        if pitch < f0_percentiles[0]:
            pitch_gender_rank = 'very low (<{:d} Hz)'.format(f0_percentiles[0])
        elif pitch < f0_percentiles[1]:
            pitch_gender_rank = 'relatively low ({:d}-{:d} Hz)'.format(f0_percentiles[0], f0_percentiles[1])
        elif pitch < f0_percentiles[2]:
            pitch_gender_rank = 'medium ({:d}-{:d} Hz)'.format(f0_percentiles[1], f0_percentiles[2])
        elif pitch < f0_percentiles[3]:
            pitch_gender_rank = 'relatively high ({:d}-{:d} Hz)'.format(f0_percentiles[2], f0_percentiles[3])
        else:
            pitch_gender_rank = 'very high (>240 Hz)'.format(f0_percentiles[3])
        return pitch_gender_rank
    prompt = prompt.format(entry['text'], entry['gender'], entry['pitch_detail'], entry['pitch'], gender_f0_rank(entry['pitch_detail'], entry['gender']), entry['gender'], entry['energy'], entry['speed'], entry['emotion'], entry['sentiment_score'], emotion_score_dict[int(entry['sentiment_score'])])
    return prompt

def decode_output(gpt_out):
    #print(gpt_out)
    gpt_out = gpt_out.replace('\n', '')
    # gpt_out = gpt_out.replace("'q'", "\"q\"")
    # gpt_out = gpt_out.replace("'a'", "\"a\"")
    gpt_out = re.findall(r'\{.*?\}', gpt_out)
    qa_list = [json.loads(x) for x in gpt_out]
    return qa_list

def save_list_of_lists_to_disk(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=1)
    #print(f"List of lists saved to {filename}.")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def complete_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_qa(prompt_list, output_file, max_tokens=1024, total_n_completions=1, model_engine="gpt-3.5-turbo"):

    model_engine, key_id = model_engine.split('_')[0], model_engine.split('_')[1]
    if key_id == 'l':
        # TODO: change to your own openai key
        openai.api_key = 'your_openai_key'
    elif key_id == 'g':
        # TODO: change to your own openai key
        openai.api_key = 'your_openai_key'

    all_outputs = []
    raw_outputs = []
    used_token = 0
    for prompt_pair in prompt_list:
        try:
            print(prompt_pair)
            audio_id, prompt = prompt_pair[0], prompt_pair[1]
            print(sys_prompt + '\n' + prompt)
            response = complete_with_backoff(
                model=model_engine,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": prompt + '" {text}"'}], #  }
                max_tokens=max_tokens,
                n=total_n_completions)

            cur_completion = response.choices[0].message.content.strip()
            raw_outputs.append([cur_completion])
            used_token += len(cur_completion)/4 + len(prompt)/4
            cur_prompt_outputs = decode_output(cur_completion)

            for j in range(len(cur_prompt_outputs)):
                try:
                    new_entry ={}
                    new_entry['audio_id'] = audio_id
                    new_entry['instruction'] = cur_prompt_outputs[j]['q']
                    new_entry['output'] = cur_prompt_outputs[j]['a']
                    all_outputs.append(new_entry)
                except:
                    pass

                if len(all_outputs) % 10 == 0:
                    print('{:d} questions generated, {:d} tokens'.format(len(all_outputs), int(used_token)))
                    with open(output_file, 'w') as f:
                        json.dump(all_outputs, f, indent=1)
                    save_list_of_lists_to_disk(raw_outputs, output_file[:-4]+'_raw.json')
        except:
            pass

    with open(output_file, 'w') as f:
        json.dump(all_outputs, f, indent=1)

model_list = ['gpt-3.5-turbo-0301_l', 'gpt-3.5-turbo-0301_g', 'gpt-3.5-turbo-0613_l', 'gpt-3.5-turbo-0613_g', "gpt-3.5-turbo_l", "gpt-3.5-turbo_g"]
model_engine= model_list[int(split_id) % len(model_list)]
with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/open_end/datafiles/qa/mosei_all/{:s}.json'.format(split_id), 'r') as fp:
    data = json.load(fp)
print(len(data))
prompt_list = [[data[i]['wav'], generate_prompt(data[i])] for i in range(len(data))]
begin = time.time()
generate_qa(prompt_list, '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/open_end/datafiles/qa/mosei_all_output/mosei_{:s}.json'.format(split_id), model_engine=model_engine)
end = time.time()
print('time eclipse, ', end-begin)