# Listen, Think, and Understand
- [Introduction](#introduction)
- [Citation](#citation)
- [OpenAQA (LTU) and OpenASQA (LTU-AS) Dataset](#openaqa-ltu-and-openasqa-ltu-as-dataset)
- [Set the Virtual Environment](#set-the-virtual-environment)
- [Inference ](#inference)
- [Finetune LTU and LTU-AS](#finetune-ltu-and-ltu-as)
   * [Finetune the LTU/LTU-AS Model with Toy Data](#finetune-the-ltultu-as-model-with-toy-data)
   * [Finetune the LTU/LTU-AS Model with Your Own Data](#finetune-the-ltultu-as-model-with-your-own-data)
- [Reproduce LTU and LTU-AS Training](#reproduce-ltu-and-ltu-as-training)
- [Pretrained Models](#pretrained-models)
- [Contact](#contact)
---
## Introduction

<p align="center"><img src="https://github.com/YuanGongND/ltu/blob/main/ltu.png?raw=true" alt="Illustration of CAV-MAE." width="900"/></p>

This repository contains the official implementation (in PyTorch), pretrained checkpoints, and datasets of LTU and LTU-AS. 
LTU and LTU-AS are first generation of audio and speech large language model that bridges audio/speech perception with understanding.
They not only achieve SOTA on multiple closed-ended audio and speech tasks, but also can answer any open-ended question based on the given audio. 
Please try the interactive demos to see how good they are! 

[**[LTU Interactive Demo]**](https://huggingface.co/spaces/yuangongfdu/LTU) 

[**[LTU-AS Interactive Demo]**](https://huggingface.co/spaces/yuangongfdu/ltu-2)

---
## Citation

**LTU-AS (Second Generation, Supports Speech and Audio):**

*LTU-AS was accepted at ASRU 2023. See you in Taipei!*

**[[Paper]](https://arxiv.org/pdf/2309.14405.pdf)** **[[HuggingFace Space]](https://huggingface.co/spaces/yuangongfdu/ltu-2)** **[[ASRU Peer Review]](https://github.com/YuanGongND/ltu/tree/main/asru_review)** **[[Compare LTU-1 and LTU-AS]](https://huggingface.co/spaces/yuangongfdu/LTU-Compare)**

**Authors:** [Yuan Gong](https://yuangongnd.github.io/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Hongyin Luo](https://luohongyin.github.io/), [Leonid Karlinsky](https://mitibmwatsonailab.mit.edu/people/leonid-karlinsky/), and [James Glass](https://people.csail.mit.edu/jrg/) (MIT & MIT-IBM Watson AI Lab)

```  
@inproceedings{gong_ltuas,
  title={Joint Audio and Speech Understanding},
  author={Gong, Yuan and Liu, Alexander H and Luo, Hongyin and Karlinsky, Leonid and Glass, James},
  year={2023},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
}
```  

---

**LTU (First Generation, Only Supports Audio):**

**[[Paper]](https://arxiv.org/abs/2305.10790)**  **[[HuggingFace Space]](https://huggingface.co/spaces/yuangongfdu/LTU)**

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
## OpenAQA (LTU) and OpenASQA (LTU-AS) Dataset

We release the training data for LTU (OpenAQA) and LTU-AS (OpenASQA). Specifically, we release the (`question`, `answer`, `audio_id`) tuples.
The actual audio wav files are from existing public datasets and need to be downloaded by the users. 
We provide full dataset (including all AQAs) as well as breakdowns (closed-ended and open-ended subsets, subsets of each original dataset, etc). All links are host on Dropbox and support `wget`.

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

**When prepare audio files, please make sure all audio files use a same sampling rate of 16kHz.**

The format of the dataset is a json file of a list of dict, in the following format:

```json
[
 {
  "instruction": "What is the significance of the sound of crying in this audio clip?", % the question
  "input": "I am so sad...", % the speech content
  "audio_id": "/data/sls/audioset/dave_version/audio/LZq4Neh-oWU.flac", % the audio id
  "dataset": "as_strong_train", % the original dataset
  "task": "open-ended question", % question type
  "output": "The sound of crying suggests that there is a sad or emotional situation happening in the audio clip." % the answer
 },
  ...
]
```

---

## Set the Virtual Environment

For almost all usages, you would need to set a virtual environment. 
Note LTU and LTU-AS needs different environments. Their `hf-dev` and `peft-main` are different. **Please do not mix use the venvs of LTU and LTU-AS.**

Clone or download this repository as `ltu-main`, then,

For LTU:

```bash
cd /ltu-main/src/ltu
conda create --name venv_ltu python=3.10
conda activate venv_ltu
pip install -r requirements.txt
# install customerized huggingface transformer, original transformer won't work
pip install -e hf-dev/transformers-main
# install customerized huggingface peft, original peft won't work
pip install -e peft-main
```

For LTU-AS:

```bash
cd /ltu-main/src/ltu_as
conda create --name venv_ltu_as python=3.10
conda activate venv_ltu_as
# install customerized huggingface transformer, original transformer won't work
pip install -e hf-dev/transformers-main
# install customerized huggingface peft, original peft won't work
pip install -e peft-main/
# install customerized openai-whisper, original whisper won't work 
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
		"path_to_your_wav/audio.wav",	# your audio file in 16K
		"What can be inferred from the audio?",	# your question
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
				"path_to_your_wav/audio.wav",	# your audio file in 16K
				"",
				"What can be inferred from the audio?",	# your question
				"7B (Default)",	# str in 'LLM size' Radio component
				api_name="/predict"
)
print(result)
```

### Option 3. Local Inference

For users interested in training/finetuning, we suggest to start with running inference. This would help debugging. 
The bash scripts will automatically download default LTU/LTU-AS models, you do not need to do it by yourself.
`inference_gradio.py` can be run on cpu or gpu.

**For LTU:**

```bash
conda activate venv_ltu
cd ltu-main/src/ltu
chmod 777 *
./inference.sh
```
The script may output some warnings which can be ignored. After the script finishs, it will provide a gradio link for inference, which can be run on a browser of any machine. You can also modify the script to run it on local terminal.

We also provide batch inference script `inference_batch.py`.

**For LTU-AS:**

```bash
conda activate venv_ltu_as
cd ltu-main/src/ltu_as
chmod 777 *
./inference.sh
```

The script may output some warnings which can be ignored. After the script finishs, it will provide a gradio link for inference, which can be run on a browser of any machine.

We also provide batch inference script `inference_batch.py`, note this script loads pre-extracted whisper features, rather than raw wav files. 
If you want to use raw audio file, please use `inference_gradio.py`. For how to extract whisper features, see [**[here]**](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/extract_whisper_feature.py).

*GPU Issue for LTU-AS: We find that Open-AI whisper features are different on different GPUs, which impacts the performance of LTU-AS as it takes Whisper feature as input. In the paper, we always use feature generated by old GPUs (Titan-X). But we do release a checkpoint that uses feature generated by newer GPUs (A5000/A6000), please manually switch the checkpoint if you are running on old/new GPUs (by default this code uses new GPU feature). Mismatch of training and inference GPU does not completely destroy the model, but would cause a performance drop.

## Finetune LTU and LTU-AS

### 1. Finetune the LTU/LTU-AS Model with Toy Data

We do not provide raw audio files for OpenAQA and OpenASQA due to copyright reasons. However, for easy reproduction, we provide audio files and Whisper audio features for a small sample set (toy set). 
Specifically, we provide very simple, almost one-click script to finetune the model. Once success, you can change the toy data to your own data.

For both scripts:
- You do not need to download the toy data, `prep_train.sh` will do this for you.
- You do not need to download the pretrained model, `prep_train.sh` will download the default pretrained model. However, you can change which pretrained model to use in `finetune_toy.sh`.

**For LTU:**

```bash
conda activate venv_ltu
# this path matters, many code requires relative path
cd ltu-main/src/ltu/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
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
# this path matters, many code requires relative path
cd ltu-main/src/ltu_as/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
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

For LTU, it is simple, you just need to replace `--data_path '../../../openaqa/data/openaqa_toy_relative.json'` in `finetune_toy.sh` to your own data. Note please make sure your own audios are 16kHz, absolute paths are encouraged, we use relative path just for simple one-click sample. 

For LTU-AS, it is a bit more complex, our script does not load raw audio, but pre-extracted Whisper-features, so you would also need to first extract Whisper features for your own audio, and then change the code in HF transformer package to point to your dir for Whisper feature. For how to extract whisper features, see [**[here]**](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/extract_whisper_feature.py).

## Reproduce LTU and LTU-AS Training

We suggest you first try finetuning with toy data then do this. 

This is similar to finetuning, the difference is that both LTU and LTU-AS training are with multi-stage curriculums, so you would need to start from stage 1, and then stage 2, ....
For stage 2, you would need to change `--base_model 'your_path_to_mdl/pytorch_model.bin'` to the checkpoint of trained model in stage 1. And so on so forth.

**For LTU:**
```bash
conda activate venv_ltu
# this path matters, many code requires relative path
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
# this path matters, many code requires relative path
cd ltu/src/ltu_as/train_script
# allow script executable
chmod 777 *
# prepare data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
./stage1_proj_cla.sh
./stage2_all_cla.sh # need to specify the checkpoint in stage 1 training
./stage4_all_mix_v2.sh # need to specify the checkpoint in stage 3 training
```

## Pretrained Models

For most above applications, our script handles model download (so you do not need to do it by yourself), but we do provide more checkpoints.

Other models mentioned in the paper may be provided upon request, please create an github issue to ask.

| LTU Model                          	                         | Size 	| Train Seq Length 	| Train Steps 	|  Whisper Feature GPU  	| Not Answerable Questions 	|                                                            Link   	                                                            |
|--------------------------------------------------------------|:----:	|:----------------:	|:-----------:	|:---------------------:	|:------------------------:	|:------------------------------------------------------------------------------------------------------------------------------:|
| Original in Paper (Default)                 	                | 370M 	|        108       	|    20000    	|           -           	|         Included         	|   [Download](https://www.dropbox.com/scl/fi/ir69ci3bhf4cthxnnnl76/ltu_ori_paper.bin?rlkey=zgqin9hh1nn2ua39jictcdhil&dl=1) 	    |
| Full-Finetuned (include LLM Parameters)                    	 |  27G 	|        108       	|    20000    	|           -           	|         Included         	| [Download](https://www.dropbox.com/scl/fi/m9ypar9aydlec5i635zl2/full_ft_2e-5_20000.bin?rlkey=jxv8poda31exdja0r777mtfbd&dl=1) 	 |

| LTU Model                          	         | Size 	| Train Seq Length 	| Train Steps 	|  Whisper Feature GPU  	| Not Answerable Questions 	|                                                                  Link   	                                                                   |
|----------------------------------------------|:----:	|:----------------:	|:-----------:	|:---------------------:	|:------------------------:	|:-------------------------------------------------------------------------------------------------------------------------------------------:|
| Original in Paper                  	         | 200M 	|        108       	|    40000    	|    Old GPUs (Titan)   	|         Included         	|         [Download](https://www.dropbox.com/scl/fi/34y1p8bfuwlcdepwd2e1o/ltuas_ori_paper.bin?rlkey=um20nlxzng1nig9o4ib2difoo&dl=1) 	         |
| Long_sequence_exclude_noqa_old_gpu 	         | 200M 	|        192       	|    40000    	|    Old GPUs (Titan)   	|         Excluded         	|     [Download](https://www.dropbox.com/scl/fi/co2m4ljyxym7f3w3dl6u4/ltuas_long_noqa_old_gpu.bin?rlkey=23sxa0f6l98wbci4t67y0se7v&dl=1) 	     |
| Long_sequence_exclude_noqa_new_gpu (Default)	 | 200M 	|        192       	|    40000    	| New GPUs (A5000/6000) 	|         Excluded         	|       [Download](https://www.dropbox.com/scl/fi/ryoqai0ayt45k07ib71yt/ltuas_long_noqa_a6.bin?rlkey=1ivttmj8uorf63dptbdd6qb2i&dl=1) 	        |
| Full-Finetuned (include LLM Parameters)      |  27G 	|        192       	|    40000    	|    Old GPUs (Titan)   	|         Excluded         	| [Download](https://www.dropbox.com/scl/fi/iq1fwkgkzueugqioge83g/ltuas_long_noqa_old_gpus_fullft.bin?rlkey=yac3gbjp6fbjy446qtblmht0w&dl=1) 	 |

## Important Code

This is a large code base, and we are unable to explain the code one by one. Below are the code that we think are important. 

1. The LTU/LTU model architecture are in [LTU Architecture](https://github.com/YuanGongND/ltu/blob/main/src/ltu/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py) and [LTU-AS Architecture](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py), respectively.
2. The training data collector for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/blob/f7b2c009be1a9e5c350d2861ac7f57bdc60f9dfe/src/ltu/hf-dev/transformers-main/src/transformers/data/data_collator.py#L561-L651) and [here](https://github.com/YuanGongND/ltu/blob/f7b2c009be1a9e5c350d2861ac7f57bdc60f9dfe/src/ltu_as/hf-dev/transformers-main/src/transformers/data/data_collator.py#L561-L641), respectively.
3. The text generation code for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/blob/main/src/ltu/hf-dev/transformers-main/src/transformers/generation/utils.py) and [here](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/generation/utils.py), respectively.
4. The closed-ended evaluation codes for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/eval) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/eval), respectively.
5. The GPT-assisted data generation code for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/qa_generation) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/qa_generation), respectively.
6. The Whisper-feature extraction code for LTU-AS is in [here](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/extract_whisper_feature.py).
7. The training shell scripts with our hyperparameters for each stage for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/train_scripts) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/train_scripts), respectively.
8. The finetuning python script (which will be called by the above shell scripts) for LTU and LTU-AS are in [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/train_scripts) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/train_scripts), respectively.

For training, the start point is the training shell scripts at [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/train_scripts) and [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu_as/train_scripts),
these shell scripts will call `ltu-main/{ltu,ltu_as}/finetune.py`, which will call the customerized huggingface transformer which contains the LTU/LTU-AS model and peft package.

If you have a question about the code, please create an issue.

## Required Computational Resources

For LTU/LTU-AS training, we use 4 X A6000 (4 X 48GB=196GB VRAM). The code can be run on 1 X A6000 (or similar GPUs). To run on smaller GPUs, turn on model parallelism, we were able to run it on 4 X A5000 (4 X 24GB = 96GB) for LTU-AS (as Whisper takes some memory).

For inference, the minimal would be 2 X TitanX (2 X 12GB = 24GB) for LTU and 4 X TitanX (4 X 12GB = 48GB). However, you can run inference on CPUs.

## Mirror Links

All resources are hosted on Dropbox, support `wget`, and should be available for most countries/areas. For those who cannot access Dropbox, A VPN is recommended in this case, but we do provide a mirror link at [Tencent Cloud 腾讯微云](https://share.weiyun.com/ee7k45lH), however, you would need to manually place the model/data in to desired place, our automatic script will fail. 

## Contact
If you have a question, please create an issue, I usually respond promptly, if delayed, please ping me. 
For more personal or confidential request, please send me an email [yuangong@mit.edu](yuangong@mit.edu).
