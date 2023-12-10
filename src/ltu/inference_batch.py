import sys
import os

default_cuda_devices = "0,1,2,3"
if len(sys.argv) > 1:
    argument = sys.argv[1]
    if argument == '4':
        argument = default_cuda_devices
else:
    argument = default_cuda_devices
os.environ["CUDA_VISIBLE_DEVICES"] = argument

import os
import torchaudio
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    return fbank

def main(
    base_model: str = "../../pretrained_mdls/vicuna_ltu/",
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    temp, top_p, top_k = 0.1, 0.95, 500
    # change it to your model path
    eval_mdl_path = '/data/sls/scratch/yuangong/ltu/pretrained_mdls/ltu_ori_paper.bin'
    state_dict = torch.load(eval_mdl_path, map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)

    model.is_parallelizable = True
    model.model_parallel = True

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    eval_dataset_list = ['esc50', 'audiocaps', 'as', 'vgg', 'clotho', 'fsd', 'vs', 'bj', 'tut', 'dcase', 'open-gpt3', 'open-gpt4']

    # all these json file can be downloaded from https://www.dropbox.com/scl/fo/juh1dk9ltvhghuj0l1sag/h?rlkey=0n2cd5kebzh8slwanjzrfn7q6&dl=0
    # you will need to prepare audio by yourself, note please convert all audios to 16khz
    for eval_dataset in eval_dataset_list:
        # ESC-50 classification
        if eval_dataset == 'esc50':
            with open('../../eval_data/esc50.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # AudioCaps captioning
        elif eval_dataset == 'audiocaps':
            with open('../../eval_data/audiocaps_test.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'acaps_cap': 'Close-ended question: Write an audio caption describing the sound, in AudioCaps style.'}
        # AudioSet multi-label classification
        elif eval_dataset == 'as':
            with open('../../eval_data/audioset_eval.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # vggsound classification
        elif eval_dataset == 'vgg':
            with open('../../eval_data/vggsound_eval.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # clotho v2 captioning
        elif eval_dataset == 'clotho':
            with open('../../eval_data/clotho_caption_evaluation.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'clotho_cap': 'Close-ended question: Create a caption for audio, in Clotho style.'}
        # fsd50k multi-label classification
        elif eval_dataset == 'fsd':
            with open('../../eval_data/fsd50k_eval.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # vocalsound classification
        elif eval_dataset == 'vs':
            with open('../../eval_data/vocalsound_test.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # beijing opera classification
        elif eval_dataset == 'bj':
            with open('../../eval_data/beijing_opera.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # tut17 classification
        elif eval_dataset == 'tut':
            with open('../../eval_data/tut_17.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # dcase classification
        elif eval_dataset == 'dcase':
            with open('../../eval_data/dcase17.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'caption': 'Close-ended question: Write an audio caption describing the sound.'}
        # open-ended question generated by GPT-3.5-Turbo, on AudioSet Eval audio/annotation
        elif eval_dataset == 'open-gpt3':
            with open('../../eval_data/open_as_gpt3.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'open': ''}
        # open-ended question generated by GPT-4, on AudioSet Eval audio/annotation
        elif eval_dataset == 'open-gpt4':
            with open('../../eval_data/open_as_gpt4.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'open': ''}
        # alpaca pure language task
        elif eval_dataset == 'alpaca':
            with open('../../eval_data/alpaca_1000.json', 'r') as fp:
                data_json_1 = json.load(fp)
            task_dict = {'text': ''}

        for task in task_dict.keys():
            result_json = []
            for i in range(len(data_json_1)):
                cur_answer = data_json_1[i]["output"]
                cur_audio_path = data_json_1[i]["audio_id"]

                audio_path = cur_audio_path

                if task != 'open' and task!='text':
                    # using fixed instruction
                    instruction = task_dict[task]
                else:
                    instruction = data_json_1[i]["instruction"]

                begin_time = time.time()

                prompt = prompter.generate_prompt(instruction, None)
                print('Input prompt: ', prompt)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

                if audio_path != 'empty':
                    cur_audio_input = load_audio(audio_path).unsqueeze(0)
                    if torch.cuda.is_available() == False:
                        pass
                    else:
                        cur_audio_input = cur_audio_input.half().to(device)
                else:
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
                output = tokenizer.decode(s)[6:-4]
                end_time = time.time()
                print(output)
                print('eclipse time: ', end_time-begin_time, ' seconds.')

                result_json.append({'prompt': instruction, 'pred': output[len(prompt):], 'ref': cur_answer, 'audio_id': cur_audio_path})
                save_name = eval_mdl_path.split('/')[-1].split('.')[0]
                if os.path.exists('./eval_res') == False:
                    os.mkdir('./eval_res')
                with open('./eval_res/{:s}_{:s}_{:s}.json'.format(eval_dataset, save_name, task), 'w') as fj:
                    json.dump(result_json, fj, indent=1)

if __name__ == "__main__":
    fire.Fire(main)
