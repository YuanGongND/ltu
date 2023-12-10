# -*- coding: utf-8 -*-
# @Time    : 12/9/23 4:11 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : inference_batch.py

# this script loads pre-extracted whisper feature, not raw audio

import os
import fire
import json
import torch
import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def convert_params_to_float32(model):
    for name, param in model.named_parameters():
        if "audio_encoder" in name and "ln" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()

def print_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Data type: {param.dtype}, device '{param.device}'")

# load pretrained feature
def gen_filename(filename, dataset='esc50'):
    if dataset == 'fma':
        dataset = 'fma_mp3'
    ext = filename.split('.')[-1]
    filename = '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/whisper_feat/' + str(dataset) + '/whisper_large-v1/' + filename.split('/')[-1][:-len(ext) - 1] + '.npz'
    return filename

def load_audio(filename, dataset='esc50'):
    filename = gen_filename(filename, dataset)
    audio_feat = torch.FloatTensor(np.load(filename)['arr_0'])
    return audio_feat

def main(
    load_8bit: bool = False,
    base_model: str = "../../pretrained_mdls/vicuna_ltuas/",
    prompt_template: str = "alpaca_short"
):
    eval_mdl_path = '../../pretrained_mdls/ltuas_long_noqa_a6.bin'
    eval_mode = 'joint'
    assert eval_mode in ['asr_only', 'audio_only', 'joint']

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16) #, torch_dtype=torch.float16

    convert_params_to_float32(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    temp, top_p, top_k = 0.1, 0.95, 500
    if eval_mdl_path != 'vicuna':
        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        miss, unexpect = model.load_state_dict(state_dict, strict=False)
        print('unexpect', unexpect)

    model.is_parallelizable = True
    model.model_parallel = True

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    # all these json file can be downloaded from https://www.dropbox.com/scl/fo/o91k6cnwqft84tgmuotwg/h?rlkey=6bnjobvrbqbt4rqt3f1tgaeb8&dl=0
    # you will need to prepare whisper feature by yourself, note please convert all audios to 16khz

    eval_dataset_list = ['esc50', 'fsd50k', 'audiocaps',
                         'librispeech',
                         'fma', 'GTZAN',
                         'mosei', 'iemocap',
                         'voxceleb_gender', 'voxceleb_age', 'ins_speech', 'ins_audio']

    for eval_dataset in eval_dataset_list:
        try:
            if eval_dataset == 'esc50':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/cavmae_esc50_eval_whisper.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
            elif eval_dataset == 'fsd50k':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/fsd50k_eval.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
            elif eval_dataset == 'audiocaps':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/audiocaps_eval.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'acaps_cap': 'Close-ended question: Write an audio caption describing the sound, in AudioCaps style.'}
            elif eval_dataset == 'librispeech':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/librispeech.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'asr': 'Closed-ended question: Can you identify the spoken text?'}
            elif eval_dataset == 'fma':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/fma_test_for_eval.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'genre': 'Describe the music genre with a sentence?'}
            elif eval_dataset == 'GTZAN':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/GTZAN_eval.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'genre': 'Describe the music genre with a sentence?'}
            elif eval_dataset == 'mosei':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/mosei_te_emotion_emotion_cla.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'emotion_cla': "Describe the speaker emotion with a sentence."}
            elif eval_dataset == 'iemocap':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/iemocap_te_emotion.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'emotion_cla': "How would you describe the speaker's emotions? Give the emotion classification."}
            elif eval_dataset == 'voxceleb_gender':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/voxceleb_te_gender_cla_eval.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'gender_cla': "Is the speaker male or female? Output an answer anyways."}
            elif eval_dataset == 'voxceleb_age':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/eval_dataset/voxceleb_te_age_reg_eval.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'age_reg': "How old is the speaker?"}
            elif eval_dataset == 'ins_speech':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/instruction_following/datafiles/speech_ins_q.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'open': ""}
            elif eval_dataset == 'ins_audio':
                with open('/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/instruction_following/datafiles/audio_ins_q.json', 'r') as fp:
                    data_json_1 = json.load(fp)
                task_dict = {'open': ""}

            for task in task_dict.keys():
                result_json = []
                for i in range(len(data_json_1)):
                    if 'output' in data_json_1[i]:
                        cur_answer = data_json_1[i]["output"]
                    else:
                        cur_answer = ''
                    cur_audio_path = data_json_1[i]["audio_id"]

                    audio_path = cur_audio_path

                    if task != 'open' and task != 'text':
                        instruction = task_dict[task]  # fixed task instruction
                    else:
                        instruction = data_json_1[i]["instruction"]

                    begin_time = time.time()
                    sample_input = data_json_1[i]["input"]

                    if eval_mode == 'audio_only':
                        prompt = prompter.generate_prompt(instruction, None) # None
                    else:
                        prompt = prompter.generate_prompt(instruction, sample_input)

                    print('Input prompt: ', prompt)
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)

                    if eval_mode == 'asr_only':
                        cur_audio_input = None
                    else:
                        if audio_path != 'empty':
                            cur_audio_input = load_audio(audio_path, eval_dataset.split('_')[0]).unsqueeze(0)
                            if torch.cuda.is_available() == False:
                                pass
                            else:
                                cur_audio_input = cur_audio_input.half().to(device)  # .half().to(device)
                                print(cur_audio_input.dtype)
                        else:
                            print('loading audio error')
                            cur_audio_input = None

                    generation_config = GenerationConfig(
                        do_sample=True,
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=1.1,
                        max_new_tokens=400,
                        bos_token_id=model.config.bos_token_id,
                        eos_token_id=model.config.eos_token_id,
                        pad_token_id=model.config.pad_token_id,
                        num_return_sequences=1
                    )

                    # Without streaming
                    with torch.no_grad():
                        generation_output = model.generate(
                            input_ids=input_ids.to(device),
                            audio_input=cur_audio_input,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_new_tokens=400,
                        )
                    s = generation_output.sequences[0]
                    output = tokenizer.decode(s)
                    output = output[5:-4]
                    end_time = time.time()
                    print(output)
                    print('eclipse time: ', end_time-begin_time, ' seconds.')

                    result_json.append({'prompt': instruction, 'pred': output[len(prompt)+1:], 'ref': cur_answer, 'audio_id': cur_audio_path})
                    save_name = eval_mdl_path.split('/')[-1].split('.')[0]
                    if os.path.exists('./eval_res') == False:
                        os.mkdir('./eval_res')
                    with open('./eval_res/{:s}_{:s}_{:s}.json'.format(eval_dataset, save_name, task), 'w') as fj:
                        json.dump(result_json, fj, indent=1)
        except Exception as e:
            print("Caught an error: ", str(e))

if __name__ == "__main__":
    fire.Fire(main)
