# Listen, Think, and Understand
- [Introduction](#introduction)
- [Citation](#citation)
- [OpenAQA (LTU) and OpenASQA (LTU-AS) Dataset](#openaqa-ltu-and-openasqa-ltu-as-dataset)
   * [OpenAQA (LTU)](#for-ltu-openaqa)
   * [OpenASQA (LTU-AS)](#for-ltu-as-openasqa)
- [Set the Virtual Environment](#set-the-virtual-environment)
- [Inference ](#inference)
   * [Option 1. Inference via HuggingFace Space (No Code Needed)](#option-1-inference-via-huggingface-space-no-code-needed)
   * [Option 2. Inference with API (No GPU Needed)](#option-2-inference-with-api-no-gpu-needed)
   * [Option 3. Local Inference](#option-3-local-inference)
- [Finetune LTU and LTU-AS](#finetune-ltu-and-ltu-as)
   * [Finetune the LTU/LTU-AS Model with Toy Data](#finetune-the-ltultu-as-model-with-toy-data)
   * [Finetune the LTU/LTU-AS Model with Your Own Data](#finetune-the-ltultu-as-model-with-your-own-data)
- [Reproduce LTU and LTU-AS Training](#reproduce-ltu-and-ltu-as-training)
- [Pretrained Models](#pretrained-models)
- [Important Code](#important-code)
- [Required Computational Resources](#required-computational-resources)
- [Mirror Links](#mirror-links)
---
## Introduction

<p align="center"><img src="https://github.com/YuanGongND/ltu/blob/main/ltu.png?raw=true" alt="Illustration of CAV-MAE." width="900"/></p>

This repository contains the official implementation (in PyTorch), pretrained checkpoints, and datasets of LTU and LTU-AS. 
LTU and LTU-AS are the first generation of audio and speech large language model that bridges audio/speech perception with understanding.
They not only achieve SOTA on multiple closed-ended audio and speech tasks, but also can answer any open-ended question based on the given audio. 
Please try the interactive demos to see how good they are!

[**[LTU Interactive Demo]**](https://huggingface.co/spaces/yuangongfdu/LTU)

[**[LTU-AS Interactive Demo]**](https://huggingface.co/spaces/yuangongfdu/ltu-2)

---
## Citation


**LTU (First Generation, Only Supports Audio):**

*LTU was accepted at ICLR 2024. See you in Vienna!*

**[[Paper]](https://openreview.net/pdf?id=nBZBPXdJlC)**  **[[HuggingFace Space]](https://huggingface.co/spaces/yuangongfdu/LTU)** **[[ICLR Peer Review]](https://openreview.net/forum?id=nBZBPXdJlC)**

**Authors:** [Yuan Gong](https://yuangongnd.github.io/), [Hongyin Luo](https://luohongyin.github.io/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Leonid Karlinsky](https://mitibmwatsonailab.mit.edu/people/leonid-karlinsky/), and [James Glass](https://people.csail.mit.edu/jrg/) (MIT & MIT-IBM Watson AI Lab)

```  
@article{gong2023listen,
  title={Listen, Think, and Understand},
  author={Gong, Yuan and Luo, Hongyin and Liu, Alexander H and Karlinsky, Leonid and Glass, James},
  journal={arXiv preprint arXiv:2305.10790},
  year={2023}
}
```  


---

**LTU-AS (Second Generation, Supports Speech and Audio):**

*LTU-AS was accepted at ASRU 2023 (top 3% paper). See you in Taipei!*

**[[Paper]](https://arxiv.org/pdf/2309.14405.pdf)** **[[HuggingFace Space]](https://huggingface.co/spaces/yuangongfdu/ltu-2)** **[[ASRU Peer Review]](https://github.com/YuanGongND/ltu/tree/main/asru_review)**

**Authors:** [Yuan Gong](https://yuangongnd.github.io/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Hongyin Luo](https://luohongyin.github.io/), [Leonid Karlinsky](https://mitibmwatsonailab.mit.edu/people/leonid-karlinsky/), and [James Glass](https://people.csail.mit.edu/jrg/) (MIT & MIT-IBM Watson AI Lab)

```  
@inproceedings{gong_ltuas,
  title={Joint Audio and Speech Understanding},
  author={Gong, Yuan and Liu, Alexander H and Luo, Hongyin, and Karlinsky, Leonid and Glass, James},
  year={2023},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
}
```

---
## OpenAQA (LTU) and OpenASQA (LTU-AS) Dataset

We release the training data for LTU (OpenAQA) and LTU-AS (OpenASQA). Specifically, we release the (`question`, `answer`, `audio_id`) tuples.
The actual audio files are from existing public datasets and need to be downloaded by the users. 
We provide the full dataset (including all AQAs) as well as breakdowns (closed-ended and open-ended subsets, subsets of each original dataset, etc). All links are hosted on Dropbox and support `wget`.

### For LTU (OpenAQA)

**Toy Set (Contains Raw Audio Files, for Testing Purpose Only)**:

For LTU: [[Meta]](https://www.dropbox.com/scl/fi/3g2b9dzeklunqwjs4ae05/openaqa_toy.json?rlkey=1a7gjjtjrvvpbnucur8wq46vr&dl=1) [[Audio]](https://www.dropbox.com/scl/fo/jdlkm9ggj3ascp2g8oehk/h?rlkey=tuh31ii3dpyg70zoaaxtllx3a&dl=1)

**OpenAQA Training (Only Audio Datasets, 5.6M AQAs in Total)**:

Full Dataset (2.3GB): [[Download]](https://www.dropbox.com/scl/fi/4k64am7ha8gy2ojl1t29e/openaqa_5.6M.json?rlkey=hqhg6bly8kbqqm09l1735ke0f&dl=1)

Breakdown Subsets:  [[Download]](https://www.dropbox.com/scl/fo/iip9zxdt693esl8db3vkq/h?rlkey=nlesigkgjz106uwv448s7bxlh&dl=1)

**LTU Evaluation Data**: 
[[Download]](https://www.dropbox.com/scl/fo/juh1dk9ltvhghuj0l1sag/h?rlkey=0n2cd5kebzh8slwanjzrfn7q6&dl=1)

---

### For LTU-AS (OpenASQA)

**Toy Set (Contains Raw Audio Files, for Testing Purpose Only)**:

For LTU-AS: [[Meta]](https://www.dropbox.com/scl/fi/63szdwo0mv519o4nmgvd3/openasqa_toy.json?rlkey=coch9fc1hwyor8ezxx1bf5d9b&dl=1) [[Audio and Whisper Feature]](https://www.dropbox.com/scl/fo/ko9qykuwbe4nodtsx8vl4/h?rlkey=nxaslb9f9g8j8k82xfxsf86ls&dl=1)

**OpenASQA Training (Audio and Speech Datasets, 10.2M AQAs in Total)**:

Full Dataset (4.6GB): [[Download]](https://www.dropbox.com/scl/fi/nsnjivql7vb1i7vcf37d1/openasqa_10.3M_v2.json?rlkey=hy9xmbkaorqdlf72xmir1exe7&dl=1)

Breakdown Subsets: [[Download]](https://www.dropbox.com/scl/fo/nvn2k1wrmgjz4wglcs9zy/h?rlkey=2l4lz83d90swlooxizn26uubp&dl=1)

**LTU-AS Evaluation Data**: 
[[Download]](https://www.dropbox.com/scl/fo/o91k6cnwqft84tgmuotwg/h?rlkey=6bnjobvrbqbt4rqt3f1tgaeb8&dl=1)

---

**When preparing audio files, please make sure all audio files use the same sampling rate of 16kHz.**

The format of the dataset is a JSON file of a list of dicts, in the following format:

```json
[
 {
  "instruction": "What is the significance of the sound of crying in this audio clip?", % the question
  "input": "I am so sad...", % the speech content
  "audio_id": "/data/sls/audioset/dave_version/audio/LZq4Neh-oWU.flac", % the audio id
  "dataset": "as_strong_train", % the original dataset (optional)
  "task": "open-ended question", % question type (optional)
  "output": "The sound of crying suggests that there is a sad or emotional situation happening in the audio clip." % the answer
 },
  ...
]
```

---

## Set the Virtual Environment

For almost all usages, you would need to set up a virtual environment. 
Note that LTU and LTU-AS need different environments. Their `hf-dev` and `peft-main` are different. **Please do not mix use the venvs of LTU and LTU-AS.**

Clone or download this repository as `ltu-main`, then,

For LTU:

```bash
cd /ltu-main/src/ltu
conda create --name venv_ltu python=3.10
conda activate venv_ltu
pip install -r requirements.txt
# install customized hugging face transformer, the original transformer won't work
pip install -e hf-dev/transformers-main
# install customized hugging face peft, original peft won't work
pip install -e peft-main
```

For LTU-AS:

```bash
cd /ltu-main/src/ltu_as
conda create --name venv_ltu_as python=3.10
conda activate venv_ltu_as
pip install -r requirements.txt
# install customized hugging face transformer, the original transformer won't work
pip install -e hf-dev/transformers-main
# install customized hugging face peft, original peft won't work
pip install -e peft-main/
# install customized openai-whisper, original whisper won't work 
pip install -e whisper/
```

## Inference 

We provide three options for inference.

### Option 1. Inference via HuggingFace Space (No Code Needed)

<p align="center"><img src="https://github.com/YuanGongND/ltu/blob/main/usage.gif?raw=true" alt="Illustration of CAV-MAE." width="900"/></p>

[**[LTU Interactive Demo]**](https://huggingface.co/spaces/yuangongfdu/LTU) 

[**[LTU-AS Interactive Demo]**](https://huggingface.co/spaces/yuangongfdu/ltu-2)

### Option 2. Inference with API (No GPU Needed)

API supports batch inference with a simple for loop.

```python
!pip install gradio_client
```

For LTU:

```python
from gradio_client import Client

client = Client("https://yuangongfdu-ltu.hf.space/")
result = client.predict(
      "path_to_your_wav/audio.wav",  # your audio file in 16K
      "What can be inferred from the audio?",    # your question
      api_name="/predict"
)
print(result)
```

For LTU-AS:

```python
# For LTU-AS
from gradio_client import Client

client = Client("https://yuangongfdu-ltu-2.hf.space/")
result = client.predict(
            "path_to_your_wav/audio.wav",  # your audio file in 16K
            "",
            "What can be inferred from the audio?",    # your question
            "7B (Default)",    # str in 'LLM size' Radio component
            api_name="/predict"
)
print(result)
```

### Option 3. Local Inference

For users interested in training/finetuning, we suggest starting with running inference. This would help debugging. 
The bash scripts will automatically download default LTU/LTU-AS models, you do not need to do it by yourself.
`inference_gradio.py` can be run on CPU or GPU.

**For LTU:**

```bash
conda activate venv_ltu
cd ltu-main/src/ltu
chmod 777 *
./inference.sh
```
The script may output some warnings which can be ignored. After the script finishes, it will provide a gradio link for inference, which can be run on a browser of any machine. You can also modify the script to run it on a local terminal.

We also provide batch inference script `inference_batch.py`.

**For LTU-AS:**

```bash
conda activate venv_ltu_as
cd ltu-main/src/ltu_as
chmod 777 *
./inference.sh
```

The script may output some warnings which can be ignored. After the script finishes, it will provide a gradio link for inference, which can be run on a browser of any machine.

We also provide batch inference script `inference_batch.py`, note this script loads pre-extracted whisper features, rather than raw WAV files. 
If you want to use raw audio files, please use `inference_gradio.py`. For how to extract whisper features, see [**[here]**](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/extract_whisper_feature.py).

*GPU Issue for LTU-AS: We find that Open-AI whisper features are different on different GPUs, which impacts the performance of LTU-AS as it takes the Whisper feature as input. In the paper, we always use features generated by old GPUs (Titan-X). But we do release a checkpoint that uses a feature generated by newer GPUs (A5000/A6000), please manually switch the checkpoint if you are running on old/new GPUs (by default this code uses a new GPU feature). A mismatch of training and inference GPU does not completely destroy the model, but would cause a performance drop.

## Finetune LTU and LTU-AS

### Finetune the LTU/LTU-AS Model with Toy Data

We do not provide raw audio files for OpenAQA and OpenASQA due to copyright reasons. However, for easy reproduction, we provide audio files and Whisper audio features for a small sample set (toy set). 
Specifically, we provide a very simple, almost one-click script to finetune the model. Once successful, you can change the toy data to your own data.

For both scripts:
- You do not need to download the toy data, `prep_train.sh` will do this for you.
- You do not need to download the pretrained model, `prep_train.sh` will download the default pretrained model. However, you can change which pretrained model to use in `finetune_toy.sh`.

**For LTU:**

```bash
conda activate venv_ltu
# this path matters, many codes require relative path
cd ltu-main/src/ltu/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
# for (multiple) GPUs with <48GB memory use, slower
#./finetune_toy_low_resource.sh
```

You should get something similar as 

```
trainable params: 93065216 || all params: 6831480832 || trainable%: 1.3622993065290356
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 6306/6306 [00:02<00:00, 2626.08 examples/s]
{'loss': 0.6383, 'learning_rate': 1e-05, 'epoch': 0.41}                                                                                               
{'loss': 0.6052, 'learning_rate': 2e-05, 'epoch': 0.81}                                                                                               
{'train_runtime': 142.0142, 'train_samples_per_second': 44.404, 'train_steps_per_second': 0.169, 'train_loss': 0.6136090755462646, 'epoch': 0.97}    
```

**For LTU-AS:**

```bash
conda activate venv_ltu_as
# this path matters, many codes require relative path
cd ltu-main/src/ltu_as/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
# for (multiple) GPUs with <48GB memory use, slower
#./finetune_toy_low_resource.sh
```

You should get something like:

```
trainable params: 48793600 || all params: 6787209216 || trainable%: 0.718905200166442
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 8769/8769 [00:04<00:00, 2088.17 examples/s]
{'loss': 0.6029, 'learning_rate': 2e-05, 'epoch': 0.29}                                                                                               
{'loss': 0.5805, 'learning_rate': 4e-05, 'epoch': 0.58}                                                                                               
{'loss': 0.5397, 'learning_rate': 6e-05, 'epoch': 0.87}                                                                                               
{'train_runtime': 175.7491, 'train_samples_per_second': 49.895, 'train_steps_per_second': 0.193, 'train_loss': 0.5713561913546394, 'epoch': 0.99} 
```

### Finetune the LTU/LTU-AS Model with Your Own Data

For LTU, it is simple, you just need to replace `--data_path '../../../openaqa/data/openaqa_toy_relative.json'` in `finetune_toy.sh` to your own data. Note please make sure your own audios are 16kHz, absolute paths are encouraged, we use relative paths just for simple one-click sample. 

For LTU-AS, it is a bit more complex, our script does not load raw audio, but pre-extracted Whisper features, so you would also need to first extract Whisper features for your own audio, and then change the code in the HF transformer package to point to your dir for Whisper feature (basically need to change [[these lines of code]](https://github.com/YuanGongND/ltu/blob/6869e4780d332b5758662091bad1c69daa572ca9/src/ltu_as/hf-dev/transformers-main/src/transformers/data/data_collator.py#L561C9-L571)). For how to extract whisper features, see [**[here]**](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/extract_whisper_feature.py).

## Reproduce LTU and LTU-AS Training

We suggest you first try finetuning with toy data then do this. 

This is similar to finetuning, the difference is that both LTU and LTU-AS training have multi-stage curriculums, so you would need to start from stage 1, and then stage 2,...
For stage 2, you would need to change `--base_model 'your_path_to_mdl/pytorch_model.bin'` to the checkpoint of the trained model in stage 1. And so on and so forth.

**For LTU:**
```bash
conda activate venv_ltu
# this path matters, many codes require relative path
cd ltu/src/ltu/train_script
# allow script executable
chmod 777 *
# prepare data and pretrained models
./prep_train.sh
# run finetuning on the data
./stage1_proj_cla.sh
./stage2_all_cla.sh # need to specify the checkpoint in stage 1 training
./stage3_all_close.sh # need to specify the checkpoint in stage 2 training
./stage4_all_mix.sh # need to specify the checkpoint in stage 3 training
```

**For LTU-AS:**

```bash
conda activate venv_ltu_as
# this path matters, many codes require relative path
cd ltu/src/ltu_as/train_script
# allow script executable
chmod 777 *
# prepare data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
./stage1_proj_cla.sh
./stage2_all_cla.sh # need to specify the checkpoint in stage 1 training
./stage4_all_mix_v2.sh # need to specify the checkpoint in stage 2 training
```

## Pretrained Models

For most above applications, our script handles model download (so you do not need to do it by yourself), but we do provide more checkpoints.

Other models mentioned in the paper may be provided upon request, please create a GitHub issue to ask.

**LTU Models**

| LTU Model                                                     | Size    | Train Seq Length     | Train Steps  |  Whisper Feature GPU     | Not Answerable Questions     |                                                            Link                                                                  |
|--------------------------------------------------------------|:----: |:----------------:    |:-----------: |:---------------------:   |:------------------------:    |:------------------------------------------------------------------------------------------------------------------------------:|
| Original in Paper (Default)                                  | 370M     |        108           |    20000     |           -              |         Included             |   [Download](https://www.dropbox.com/scl/fi/ir69ci3bhf4cthxnnnl76/ltu_ori_paper.bin?rlkey=zgqin9hh1nn2ua39jictcdhil&dl=1)        |
| Full-Finetuned (include LLM Parameters)                       |  27G    |        108           |    20000     |           -              |         Included             | [Download](https://www.dropbox.com/scl/fi/m9ypar9aydlec5i635zl2/full_ft_2e-5_20000.bin?rlkey=jxv8poda31exdja0r777mtfbd&dl=1)      |

**LTU-AS Models**

| LTU-AS Model                                 | Size    | Train Seq Length | Train Steps  |  Whisper Feature GPU     | Not Answerable Questions     |                                                                  Link                                                                       |
|----------------------------------------------|:----: |:----------------:|:-----------: |:---------------------:   |:------------------------:    |:-------------------------------------------------------------------------------------------------------------------------------------------:|
| Original in Paper                            | 200M    |       108        |    40000     |    Old GPUs (Titan)      |         Included             |         [Download](https://www.dropbox.com/scl/fi/34y1p8bfuwlcdepwd2e1o/ltuas_ori_paper.bin?rlkey=um20nlxzng1nig9o4ib2difoo&dl=1)             |
| Long_sequence_exclude_noqa_old_gpu           | 200M    |       160        |    40000     |    Old GPUs (Titan)      |         Excluded             |     [Download](https://www.dropbox.com/scl/fi/co2m4ljyxym7f3w3dl6u4/ltuas_long_noqa_old_gpu.bin?rlkey=23sxa0f6l98wbci4t67y0se7v&dl=1)         |
| Long_sequence_exclude_noqa_new_gpu (Default) | 200M    |       160        |    40000     | New GPUs (A5000/6000)    |         Excluded             |       [Download](https://www.dropbox.com/scl/fi/ryoqai0ayt45k07ib71yt/ltuas_long_noqa_a6.bin?rlkey=1ivttmj8uorf63dptbdd6qb2i&dl=1)           |
| Full-Finetuned (include LLM Parameters)      |  27G  |       160        |    40000     |    Old GPUs (Titan)      |         Excluded             | [Download](https://www.dropbox.com/scl/fi/iq1fwkgkzueugqioge83g/ltuas_long_noqa_old_gpus_fullft.bin?rlkey=yac3gbjp6fbjy446qtblmht0w&dl=1)     |

## More Pretrained Models

We provide the following models to help reproduction. 

**1. Checkpoints of Each Stage in the Training Curriculum (with Loss Log)**

These checkpoints can be used for continue training from any stage, e.g., you can train your own stage 4 model based on a stage 3 checkpoint. You can compare our loss log with yours to ensure everything is OK.

**LTU**: [[Download Link]](https://www.dropbox.com/scl/fo/dqe01g38sfl1oo8mqpjs7/h?rlkey=kch4eogr9bzmb39plyausdhuv&dl=0)

Including Stage 1/2/3/4 checkpoints. Training arguments and loss logs are provided to help reproduction.

**LTU-AS**: [[Download Link]](https://www.dropbox.com/scl/fo/fujzoplziworiw29nleji/h?rlkey=tdl68lnu5ftbd27dnpi6zclrv&dl=0)

Including Stage 1/2/3 checkpoints, for the final stage3 checkpoint, provide v1 and v2 (with more joint audio and speech training data) checkpoints. Training arguments and loss logs are provided to help reproduction. 

**Where are the loss logs?** Click one of above links, in folders named "checkpoint-xxxx", find files called `trainer_state.json`, you should see something like:

``` 
"log_history": [
    {
      "epoch": 0.0,
      "learning_rate": 0.0001,
      "loss": 8.7039,
      "step": 10
    },
    {
      "epoch": 0.0,
      "learning_rate": 0.0002,
      "loss": 5.5624,
      "step": 20
    },
    {
      "epoch": 0.01,
      "learning_rate": 0.0003,
      "loss": 4.1076,
      "step": 30
```

That is the actual log loss. We released the logs for all stages for both LTU and LTU-AS to help reproduction.

**2. LLaMA 13B Models (including 13B model script)**

Our papers mostly focus on LLaMA-7B models, but we provide LLaMA-13B checkpoints. You would need to replace the model script [For LTU](https://github.com/YuanGongND/ltu/blob/main/src/ltu/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py) and [For LTU-AS](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py) with the 13B version ones, which can be downloaded with the following links.

**LTU-13B**: [[Download Link]](https://www.dropbox.com/scl/fo/pik0pubn1qg7q0rorkn3b/h?rlkey=x83qhk4mpwgrgns0e4rholcsr&dl=0)

Including Stage 1/2/3/4 checkpoints. For stage 4, provide a standard seq length model (108) and a longer sequence model (192). We recommend to use the model `stage4_all_mix_long_seq`. 

**LTU_AS-13B**: [[Download Link]](https://www.dropbox.com/scl/fo/mldw9tuwhx8kv010zahax/h?rlkey=gtdl7jshhg1bnoow8rod8env7&dl=0)

Including Stage 1/2/3 checkpoints. For stage 3, provide a model trained with not-answerable QA training data and a model trained without not-answerable QA training data. We recommend to use the model `stage3_long_seq_exclude_noqa`. 

## Important Code

This is a large code base, and we are unable to explain the code one by one. Below are the codes that we think are important. 

1. The LTU/LTU model architecture are in [LTU Architecture](https://github.com/YuanGongND/ltu/blob/main/src/ltu/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py) and [LTU-AS Architecture](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py), respectively.
2. The training data collector for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/blob/f7b2c009be1a9e5c350d2861ac7f57bdc60f9dfe/src/ltu/hf-dev/transformers-main/src/transformers/data/data_collator.py#L561-L651) and [here](https://github.com/YuanGongND/ltu/blob/f7b2c009be1a9e5c350d2861ac7f57bdc60f9dfe/src/ltu_as/hf-dev/transformers-main/src/transformers/data/data_collator.py#L561-L641), respectively.
3. The text generation code for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/blob/main/src/ltu/hf-dev/transformers-main/src/transformers/generation/utils.py) and [here](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/generation/utils.py), respectively.
4. The closed-ended evaluation codes for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/eval) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/eval), respectively.
5. The GPT-assisted data generation code for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/qa_generation) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/qa_generation), respectively.
6. The Whisper-feature extraction code for LTU-AS is in [here](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/extract_whisper_feature.py).
7. The training shell scripts with our hyperparameters for each stage for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/train_scripts) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/train_scripts), respectively.
8. The finetuning Python script (which will be called by the above shell scripts) for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/train_scripts) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/train_scripts), respectively.

For training, the start point is the training shell scripts at [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/train_scripts) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/train_scripts),
these shell scripts will call `ltu-main/{ltu,ltu_as}/finetune.py`, which will call the customized hugging face transformer which contains the LTU/LTU-AS model and peft package.

If you have a question about the code, please create an issue.

## Required Computational Resources

For LTU/LTU-AS training, we use 4 X A6000 (4 X 48GB=196GB VRAM). The code can be run on 1 X A6000 (or similar GPUs). 

To train/finetuning on smaller GPUs, turn on model parallelism, we were able to run it on 4 X A5000 (4 X 24GB = 96GB), we provide sample script for low-resource training for [LTU](https://github.com/YuanGongND/ltu/blob/main/src/ltu/train_script/finetune_toy_low_resource.sh) and [LTU-AS](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/train_script/finetune_toy_low_resource.sh)). Please note they are slower than normal training scripts.

For inference, the minimal would be 2 X TitanX (2 X 12GB = 24GB) for LTU and 4 X TitanX (4 X 12GB = 48GB) for LTU-AS (as Whisper takes some memory). However, you can run inference on CPUs.

## Mirror Links

All resources are hosted on Dropbox, support `wget`, and should be available for most countries/areas. For those who cannot access Dropbox, A VPN is recommended in this case, but we do provide a mirror link at [Tencent Cloud 腾讯微云](https://share.weiyun.com/ee7k45lH), however, you would need to manually place the model/data in to desired place, our automatic script will fail. 

## Contact
If you have a question, please create an issue, I usually respond promptly, if delayed, please ping me. 
For more personal or confidential requests, please send me an email [yuangong@mit.edu](yuangong@mit.edu).
