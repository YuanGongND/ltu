# Listen, Think, and Understand

<p align="center"><img src="https://github.com/YuanGongND/ltu/blob/main/usage.gif?raw=true" alt="Illustration of CAV-MAE." width="900"/></p>

---

- [Citation](#citation)
- [OpenAQA (LTU) and OpenASQA (LTU-AS) Dataset](#openaqa-ltu-and-openasqa-ltu-as-dataset)
- [Set the Virtual Environment](#set-the-virtual-environment)
- [Inference ](#inference)
- [Finetune the Model with Toy Data](#finetune-the-model-with-toy-data)
- [Reproduce LTU and LTU-AS Training](#reproduce-ltu-and-ltu-as-training)
- [Pretrained Models](#pretrained-models)
- [Contact](#contact)

<!-- TOC end -->

## Citation

**LTU-AS (Second Generation, Supports Speech and Audio):**

*LTU-AS was accepted at ASRU 2023. See you in Taipei!*

**[[Paper]](https://arxiv.org/pdf/2309.14405.pdf)** **[[HuggingFace Space]](https://huggingface.co/spaces/yuangongfdu/ltu-2)** **[[ASRU Peer Review]](https://github.com/YuanGongND/ltu/tree/main/asru_review)** **[[Compare LTU-1 and LTU-AS]](https://huggingface.co/spaces/yuangongfdu/LTU-Compare)**

**Authors:** [Yuan Gong](https://yuangongnd.github.io/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Hongyin Luo](https://luohongyin.github.io/), [Leonid Karlinsky](https://mitibmwatsonailab.mit.edu/people/leonid-karlinsky/), and [James Glass](https://people.csail.mit.edu/jrg/) (MIT & MIT-IBM Watson AI Lab)

BibTeX:

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

BibTeX:

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

We release the training data for OpenAQA and OpenASQA. Specifically, we release the (`question`, `answer`, `audio_id`) tuples.
The actual audio wav files are from existing public datasets and need to be downloaded by the users. 
We provide full dataset (including all AQAs) as well as breakdowns (closed-ended and open-ended subsets, subsets of each original dataset, etc). All links are host on Dropbox and support `wget`.

**Toy Set (Contains Raw Audio Files, for Testing Purpose Only)**:

For LTU: [Meta](https://www.dropbox.com/scl/fi/3g2b9dzeklunqwjs4ae05/openaqa_toy.json?rlkey=1a7gjjtjrvvpbnucur8wq46vr&dl=1) [Audio](https://www.dropbox.com/scl/fo/jdlkm9ggj3ascp2g8oehk/h?rlkey=tuh31ii3dpyg70zoaaxtllx3a&dl=1)

For LTU-AS: [Meta](https://www.dropbox.com/scl/fi/63szdwo0mv519o4nmgvd3/openasqa_toy.json?rlkey=coch9fc1hwyor8ezxx1bf5d9b&dl=1) [Audio and Whisper Feature](https://www.dropbox.com/scl/fo/ko9qykuwbe4nodtsx8vl4/h?rlkey=nxaslb9f9g8j8k82xfxsf86ls&dl=1)

**OpenAQA Training (Only Audio Datasets, 5.6M AQAs in Total)**:

Full Dataset (2.3GB): [Download](https://www.dropbox.com/scl/fi/4k64am7ha8gy2ojl1t29e/openaqa_5.6M.json?rlkey=hqhg6bly8kbqqm09l1735ke0f&dl=1)

Breakdown Subsets:  [Download](https://www.dropbox.com/scl/fo/iip9zxdt693esl8db3vkq/h?rlkey=nlesigkgjz106uwv448s7bxlh&dl=1)

**OpenASQA Training (Audio and Speech Datasets, 10.2M AQAs in Total)**:

Full Dataset (4.6GB): [Download](https://www.dropbox.com/scl/fi/nsnjivql7vb1i7vcf37d1/openasqa_10.3M_v2.json?rlkey=hy9xmbkaorqdlf72xmir1exe7&dl=1)

Breakdown Subsets: [Download](https://www.dropbox.com/scl/fo/nvn2k1wrmgjz4wglcs9zy/h?rlkey=2l4lz83d90swlooxizn26uubp&dl=1)

**LTU Evaluation Data**

[Download](https://www.dropbox.com/scl/fo/juh1dk9ltvhghuj0l1sag/h?rlkey=0n2cd5kebzh8slwanjzrfn7q6&dl=1)

**LTU-AS Evaluation Data**

[Download](https://www.dropbox.com/scl/fo/o91k6cnwqft84tgmuotwg/h?rlkey=6bnjobvrbqbt4rqt3f1tgaeb8&dl=1)

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
Note LTU and LTU-AS needs different environments. Their `hf-dev` and `peft-main` are different. 

For LTU:

```bash
cd /ltu/src/ltu
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
cd /ltu/src/ltu_as
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

For all users, even you are only interested in training/finetuning, we suggest to start with running inference. This would help debugging. 
The bash scripts will automatically download default LTU/LTU-AS models, you do not need to do it by yourself.
`inference_gradio.py` can be run on cpu or gpu.

**For LTU:**

```bash
conda activate venv_ltu
cd ltu/src/ltu
./inference_gradio.py
```
The script may output some warnings which can be ignored. After the script finishs, it will provide a gradio link for inference, which can be run on a browser of any machine. You can also modify the script to run it on local terminal.

We also provide batch inference script `inference_batch.py`

**For LTU-AS:**

```bash
conda activate venv_ltu_as
cd ltu/src/ltu_as
./inference_gradio.py
```

The script may output some warnings which can be ignored. After the script finishs, it will provide a gradio link for inference, which can be run on a browser of any machine.

We also provide batch inference script `inference_batch.py`, note this script loads pre-extracted whisper features, rather than raw wav files. 
If you want to use raw audio file, please use `inference_gradio.py`.

*GPU Issue: We find that Open-AI whisper features are different on different GPUs, which impacts the performance of LTU-AS as it takes Whisper feature as input. In the paper, we always use feature generated by old GPUs (Titan-X). But we do release a checkpoint that uses feature generated by newer GPUs (A5000/A6000), please manually switch the checkpoint if you are running on newer GPUs. Mismatch of training and inference GPU does not totally destroy the model, but would cause a performance drop.

## Finetune the Model with Toy Data

We do not provide raw audio files for OpenAQA and OpenASQA due to copyright reasons. However, for easy reproduction, we provide audio files and Whisper audio features for a small sample set. 
Specifically, we very simple, almost one-click script to finetune the model. Once success, you can change the toy data to your own data.

For both scripts:
- You do not need to download the toy data, `prep_train.sh` will do this for you.
- You do not need to download the pretrained model, `prep_train.sh` will download the default pretrained model. However, you can change which pretrained model to use in `finetune_toy.sh`.

**For LTU:**

```bash
conda activate venv_ltu
# this path matters, many code requires relative path
cd ltu/src/ltu/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
```

**For LTU-AS:**

```bash
conda activate venv_ltu_as
# this path matters, many code requires relative path
cd ltu/src/ltu_as/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
```

**Finetune the LTU/LTU-AS Model**

For LTU, it is simple, you just need to replace `--data_path '../../../openaqa/data/openaqa_toy_relative.json'` in `finetune_toy.sh` to your own data. Note please make sure your own audios are 16kHz, absolute paths are encouraged, we use relative path just for simple one-click sample. 

For LTU-AS, it is a bit more complex, our script does not load raw audio, but pre-extracted Whisper-features, so you would also need to first extract Whisper features for your own audio, and then change the code in HF transformer package to point to your dir for Whisper feature. 

## Reproduce LTU and LTU-AS Training

We suggest you first try a toy data finetuning then do this. 

This is similar to finetuning, the difference is that both LTU and LTU-AS training are multi-stage curriculum, so you would need to start from stage 1, and then stage 2, ....
For stage 2, you would need to change `--base_model 'your_path_to_mdl/pytorch_model.bin'` to the checkpoint of trained model in stage 1. And so on so forth.

**For LTU:**
```bash
conda activate venv_ltu
# this path matters, many code requires relative path
cd ltu/src/ltu/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./stage1_proj_cla.sh
./stage2_all_cla.sh
./stage3_all_close.sh
./stage4_all_mix.sh
```

**For LTU-AS:**

```bash
conda activate venv_ltu_as
# this path matters, many code requires relative path
cd ltu/src/ltu_as/train_script
# allow script executable
chmod 777 *
# prepare toy data and pretrained models
./prep_train.sh
# run finetuning on the data
./finetune_toy.sh
./stage1_proj_cla.sh
./stage2_all_cla.sh
./stage4_all_mix_v2.sh
```

## Pretrained Models

For most above applications, our script handles model download (so you do not need to do it by yourself), but we do provide more checkpoints.

Other models mentioned in the paper may be also available upon request, please create an github issue to ask.

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

## Contact
If you have a question, please bring up an issue, I usually respond promptly, if delayed, please ping me. 
For more personal or confidential request, please send me an email [yuangong@mit.edu](yuangong@mit.edu).

[//]: # ()
[//]: # (## Important Code)

[//]: # ()
[//]: # (data collector)

[//]: # ()
[//]: # (model definition )

[//]: # ()
[//]: # (generation )

[//]: # ()
[//]: # (evaluation code )

[//]: # ()
[//]: # (GPT data generation)

[//]: # ()
[//]: # (whisper feature extraction )