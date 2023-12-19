import gradio as gr
import json
import os
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
import datetime
import re
import skimage.measure
import whisper_at
from whisper.model import Whisper, ModelDimensions

# this is a dirty workaround to have two whisper instances, whisper model for extract encoder feature, and whisper-at to get transcription.
# in future version, this two instance will be unified

device = "cuda" if torch.cuda.is_available() else "cpu"

def convert_params_to_float32(model):
    for name, param in model.named_parameters():
        if "audio_encoder" in name and "ln" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_text_model = whisper_at.load_model("large-v2", device='cuda:1')

def load_whisper():
    mdl_size = 'large-v1'
    checkpoint_path = '../../pretrained_mdls/{:s}.pt'.format(mdl_size)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    dims = ModelDimensions(**checkpoint["dims"])
    whisper_feat_model = Whisper(dims)
    whisper_feat_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    whisper_feat_model.to('cuda:0')
    return whisper_feat_model
whisper_feat_model = load_whisper()

# do not change this, this will load llm
base_model = "../../pretrained_mdls/vicuna_ltuas/"
prompt_template = "alpaca_short"
# change this to your checkpoint
eval_mdl_path = '../../pretrained_mdls/ltuas_long_noqa_a6.bin'
eval_mode = 'joint'
prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == 'cuda':
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
else:
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")
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

state_dict = torch.load(eval_mdl_path, map_location='cpu')
miss, unexpect = model.load_state_dict(state_dict, strict=False)

model.is_parallelizable = True
model.model_parallel = True

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

eval_log = []
cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_save_path = './inference_log/'
if os.path.exists(log_save_path) == False:
    os.mkdir(log_save_path)
log_save_path = log_save_path + cur_time + '.json'

def print_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Data type: {param.dtype}, device '{param.device}'")

def remove_thanks_for_watching(text):
    variations = [
        "thanks for watching", "Thanks for watching", "THANKS FOR WATCHING",
        "thanks for watching.", "Thanks for watching.", "THANKS FOR WATCHING.",
        "thanks for watching!", "Thanks for watching!", "THANKS FOR WATCHING!",
        "thank you for watching", "Thank you for watching", "THANK YOU FOR WATCHING",
        "thank you for watching.", "Thank you for watching.", "THANK YOU FOR WATCHING.",
        "thank you for watching!", "Thank you for watching!", "THANK YOU FOR WATCHING!"
    ]
    variations = sorted(variations, key=len, reverse=True)
    pattern = "|".join(re.escape(var) for var in variations)
    result = re.sub(pattern, "", text)
    return result

text_cache = {}
def load_audio_trans(filename):
    global text_cache
    if filename not in text_cache:
        result = whisper_text_model.transcribe(filename)
        text = remove_thanks_for_watching(result["text"].lstrip())
        text_cache[filename] = text
    else:
        text = text_cache[filename]
        print('using asr cache')
    _, audio_feat = whisper_feat_model.transcribe_audio(filename)
    audio_feat = audio_feat[0]
    audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
    audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
    audio_feat = audio_feat[1:]  # skip the first layer
    audio_feat = torch.FloatTensor(audio_feat)
    return audio_feat, text

# trim to only keep output
def trim_string(a):
    separator = "### Response:\n"
    trimmed_string = a.partition(separator)[-1]
    trimmed_string = trimmed_string.strip()
    return trimmed_string

def predict(audio_path, question):
    print('audio path, ', audio_path)
    begin_time = time.time()

    if audio_path != None:
        cur_audio_input, cur_input = load_audio_trans(audio_path)
        if torch.cuda.is_available() == False:
            pass
        else:
            cur_audio_input = cur_audio_input.unsqueeze(0).half().to(device)

    instruction = question
    prompt = prompter.generate_prompt(instruction, cur_input)
    print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        max_new_tokens=500,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=500,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output[5:-4]
    end_time = time.time()
    print(trim_string(output))
    cur_res = {'audio_id': audio_path, 'instruction': instruction, 'input': cur_input, 'output': trim_string(output)}
    eval_log.append(cur_res)
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    print('eclipse time: ', end_time-begin_time, ' seconds.')
    return trim_string(output)

link = "https://github.com/YuanGongND/ltu"
text = "[Github]"
paper_link = "https://arxiv.org/pdf/2305.10790.pdf"
paper_text = "[Paper]"
sample_audio_link = "https://drive.google.com/drive/folders/17yeBevX0LIS1ugt0DZDOoJolwxvncMja?usp=sharing"
sample_audio_text = "[sample audios from AudioSet evaluation set]"
demo = gr.Interface(fn=predict,
                    inputs=[gr.Audio(type="filepath"), gr.Textbox(value='What can be inferred from the spoken text and sounds? Why?', label='Edit the textbox to ask your own questions!')],
                    outputs=[gr.Textbox(label="LTU Output")],
                    cache_examples=True,
                    title="Demo of LTU-2 Beta",
                    description="LTU-2 Beta an improved version of LTU. LTU-2 is stronger in spoken text understanding and music understanding. <br>" +
                    "LTU is authored by Yuan Gong, Alexander H. Liu, Hongyin Luo, Leonid Karlinsky, and James Glass (MIT & MIT-IBM Watson AI Lab). <br>" +
                    "**Please note that the model is under construction and may be buggy. It is trained with some new techniques that are not described in LTU paper. I.e., using method described in LTU paper cannot reproduce this model.**<br>" +
                    "Input should be wav file sampled at 16kHz. This demo trim input audio to 10 seconds."
                    "**Research Demo, No Commercial Use (Due to license of LLaMA).**")
demo.launch(debug=False, share=True)