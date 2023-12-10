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
sys_prompt = """Based on the following music, generate 5 different types of complex open-ended questions that require step-by-step thinking, and corresponding answers. 
Only ask questions about the given music. No general/background questions. Do not use the given information in the question. 
Ask short, crispy, complex, and diverse questions. Answers need to be longer than 10 words.
Only ask questions that have a clear answer based on the given music. The questions need to be of different types.
Questions can be e.g., How to describe the music; What can be inferred from the music and why; What is the genre of the music and why; What can be inferred from the melody and lyrics; What is the potential scenario that the musci can be played; If the music is special (e.g., urgent, funny, interesting, abnormal, unique, etc) and why; what mood the music conveys based on lyrics and melody; what can be the potential title of the music, etc. 
Format each QA pair in a single line as a JSON dictionary (key "q" for question, and "a" for answer, wrapped with { and }). Do not include any other explanation."""

def generate_prompt(entry):
    prompt = """Music genre: {:s}; Music Lyrics: "{:s}"; Suggest Music Title: "{:s}" """
    prompt = prompt.format(', '.join(entry['genre']), entry['text'], entry['title'])
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
with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/open_end/datafiles/qa/fma/{:s}.json'.format(split_id), 'r') as fp:
    data = json.load(fp)
print(len(data))
prompt_list = [[data[i]['audio_id'], generate_prompt(data[i])] for i in range(len(data))]
begin = time.time()
generate_qa(prompt_list, '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/open_end/datafiles/qa/fma_out/fma_{:s}.json'.format(split_id), model_engine=model_engine)
end = time.time()
print('time eclipse, ', end-begin)