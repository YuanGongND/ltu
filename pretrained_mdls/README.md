# Pretrained Models

We provide the following pretrained models for LTU and LTU-AS. Other models mentioned in the paper may be also available upon request, please create an github issue to ask.

| LTU Model                          	                         | Size 	| Train Seq Length 	| Train Steps 	|  Whisper Feature GPU  	| Not Answerable Questions 	|    Link   	    |
|--------------------------------------------------------------|:----:	|:----------------:	|:-----------:	|:---------------------:	|:------------------------:	|:--------------:|
| Original in Paper (Default)                 	                | 370M 	|        108       	|    20000    	|           -           	|         Included         	| [Download]() 	 |
| Full-Finetuned (include LLM Parameters)                    	 |  27G 	|        108       	|    20000    	|           -           	|         Included         	|   [Download]() 	   |

| LTU Model                          	         | Size 	| Train Seq Length 	| Train Steps 	|  Whisper Feature GPU  	| Not Answerable Questions 	|                                                                  Link   	                                                                   |
|----------------------------------------------|:----:	|:----------------:	|:-----------:	|:---------------------:	|:------------------------:	|:-------------------------------------------------------------------------------------------------------------------------------------------:|
| Original in Paper                  	         | 200M 	|        108       	|    40000    	|    Old GPUs (Titan)   	|         Included         	|                                                               [Download]() 	                                                                |
| Long_sequence_exclude_noqa_old_gpu 	         | 200M 	|        192       	|    40000    	|    Old GPUs (Titan)   	|         Excluded         	|     [Download](https://www.dropbox.com/scl/fi/co2m4ljyxym7f3w3dl6u4/ltuas_long_noqa_old_gpu.bin?rlkey=23sxa0f6l98wbci4t67y0se7v&dl=1) 	     |
| Long_sequence_exclude_noqa_new_gpu (Default)	 | 200M 	|        192       	|    40000    	| New GPUs (A5000/6000) 	|         Excluded         	|       [Download](https://www.dropbox.com/scl/fi/ryoqai0ayt45k07ib71yt/ltuas_long_noqa_a6.bin?rlkey=1ivttmj8uorf63dptbdd6qb2i&dl=1) 	        |
| Full-Finetuned (include LLM Parameters)      |  27G 	|        192       	|    40000    	|    Old GPUs (Titan)   	|         Excluded         	| [Download](https://www.dropbox.com/scl/fi/iq1fwkgkzueugqioge83g/ltuas_long_noqa_old_gpus_fullft.bin?rlkey=yac3gbjp6fbjy446qtblmht0w&dl=1) 	 |