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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

sys_prompt = """
Based on the following audio clip, generate 10 different types of complex open-ended questions that require step-by-step thinking, and corresponding step-by-step answers.
The following information is provided: the sound events appear in the audio clip, together with its acoustic features, and corresponding onset and offset time stamps. A description of the content of the audio clip is also provided. 
Questions should be about the audio, e.g., which sound event is recognized and why (e.g., based on its acoustic feature), what can be inferred based on the combination of sound events; the temporal relationship between the sound events and what can be inferred from that; the potential scenario that such an audio clip could happen, if the audio clip is special (e.g., urgent, funny, interesting, abnormal, unique, etc) and why, what mood or atmosphere this audio clip conveys, etc. 
The more complex and diverse the question, the better.
Format each QA pair in a single line as a JSON dictionary (key "q" for question, and "a" for answer, wrapped with { and }). Do not include any other explanation.
"""

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
def generate_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_prompt(prompt_list, output_file, max_tokens=1024, total_n_completions=1, model_engine="gpt-3.5-turbo"):
    if 'gpt-4' in model_engine:
        openai.api_key = 'your_openai_key'
    else:
        openai.api_key = 'your_openai_key'

    all_outputs = []
    raw_outputs = []
    used_token = 0
    for prompt_pair in prompt_list:
        audio_id, prompt = prompt_pair[0], prompt_pair[1]
        response = generate_with_backoff(
            model=model_engine,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": prompt+'" {text}"'}],
            max_tokens=max_tokens,
            n=total_n_completions)

        cur_completion = response.choices[0].message.content.strip()
        raw_outputs.append([cur_completion])
        used_token += len(cur_completion)/4 + len(sys_prompt)/4 + len(prompt)/4
        cur_prompt_outputs = decode_output(cur_completion)

        for j in range(len(cur_prompt_outputs)):
            new_entry ={}
            new_entry['audio_id'] = audio_id
            new_entry['instruction'] = cur_prompt_outputs[j]['q']
            new_entry['output'] = cur_prompt_outputs[j]['a']
            all_outputs.append(new_entry)

            if len(all_outputs) % 10 == 0:
                print('{:d} questions generated, {:d} tokens'.format(len(all_outputs), int(used_token)))
                with open(output_file, 'w') as f:
                    json.dump(all_outputs, f, indent=1)
                save_list_of_lists_to_disk(raw_outputs, output_file[:-4]+'_raw.json')

    with open(output_file, 'w') as f:
        json.dump(all_outputs, f, indent=1)

model_engine="gpt-3.5-turbo"
with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data/all_info/datafiles/as_strong_eval_sample_600.json', 'r') as fp:
    data = json.load(fp)
prompt_list = [(x['audio_id'], 'Sound Events: ' + x['temporal'] + '; Description: ' + x['caption']) for x in data]
begin = time.time()
generate_prompt(prompt_list, '/data/sls/scratch/yuangong/audiollm/src/data/prep_data/all_info/datafiles/as_strong_eval_600_qa_{:s}.json'.format(model_engine), model_engine=model_engine)
end = time.time()
print('time eclipse, ', end-begin)