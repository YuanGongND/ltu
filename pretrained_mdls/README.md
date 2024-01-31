# Pretrained Models

We provide the following pretrained models for LTU and LTU-AS. Other models mentioned in the paper may be also available upon request, please create an github issue to ask.
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

# More Pretrained Models

We provide the following models to help reproduction. 

**1. Checkpoints of Each Stage in the Training Curriculum (with Loss Log)**

These checkpoints can be used for continue training from any stage, e.g., you can train your own stage 4 model based on a stage 3 checkpoint. You can compare our loss log with yours to ensure everything is OK.

**LTU**: [[Download Link]](https://www.dropbox.com/scl/fo/dqe01g38sfl1oo8mqpjs7/h?rlkey=kch4eogr9bzmb39plyausdhuv&dl=0)

Including Stage 1/2/3/4 checkpoints. Training arguments and loss logs are provided to help reproduction.

**LTU-AS**: [[Download Link]](https://www.dropbox.com/scl/fo/fujzoplziworiw29nleji/h?rlkey=tdl68lnu5ftbd27dnpi6zclrv&dl=0)

Including Stage 1/2/3 checkpoints, for the final stage3 checkpoint, provide v1 and v2 (with more joint audio and speech training data) checkpoints. Training arguments and loss logs are provided to help reproduction. 

**2. LLaMA 13B Models (including 13B model script)**

Our papers mostly focus on LLaMA-7B models, but we provide LLaMA-13B checkpoints. You would need to replace the model script [For LTU](https://github.com/YuanGongND/ltu/blob/main/src/ltu/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py) and [For LTU-AS](https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py) with the 13B version ones, which can be downloaded with the following links.

**LTU-13B**: [[Download Link]](https://www.dropbox.com/scl/fo/pik0pubn1qg7q0rorkn3b/h?rlkey=x83qhk4mpwgrgns0e4rholcsr&dl=0)

Including Stage 1/2/3/4 checkpoints. For stage 4, provide a standard seq length model (108) and a longer sequence model (192). We recommend to use the model `stage4_all_mix_long_seq`. 

**LTU_AS-13B**: [[Download Link]](https://www.dropbox.com/scl/fo/mldw9tuwhx8kv010zahax/h?rlkey=gtdl7jshhg1bnoow8rod8env7&dl=0)

Including Stage 1/2/3 checkpoints. For stage 3, provide a model trained with not-answerable QA training data and a model trained without not-answerable QA training data. We recommend to use the model `stage3_long_seq_exclude_noqa`. 
