# Listen, Think, and Understand

**[[Paper]](https://arxiv.org/abs/2305.10790)** **[[Interactive Demo]](https://c941ef60cc8ce23153.gradio.live/)**

<p align="center"><img src="https://github.com/YuanGongND/ltu/blob/main/ltu.png?raw=true" alt="Illustration of CAV-MAE." width="900"/></p>

**Authors:** [Yuan Gong](https://yuangongnd.github.io/), [Hongyin Luo](https://luohongyin.github.io/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Leonid Karlinsky](https://mitibmwatsonailab.mit.edu/people/leonid-karlinsky/), and [James Glass](https://people.csail.mit.edu/jrg/) (MIT & MIT-IBM Watson AI Lab)

**Abstract:** 

The ability of artificial intelligence (AI) systems to perceive and comprehend audio signals is crucial for many applications. Although significant progress has been made in this area since the development of AudioSet, most existing models are designed to map audio inputs to pre-defined, discrete sound label sets. In contrast, humans possess the ability to not only classify sounds into coarse-grained categories, but also to listen to the details of the sounds, explain the reason for the predictions, think what the sound infers, and understand the scene and what action needs to be taken. Such capabilities beyond perception are not yet present in existing audio models. On the other hand, modern large language models (LLMs) exhibit emerging reasoning ability but they lack audio perception capabilities. Therefore, we ask the question: can we build an AI model that has both audio perception and a reasoning ability?

In this paper, we propose a novel audio foundation model, called LTU (Listen, Think, and Understand). To train LTU, we created a new OpenAQA-5M dataset consisting of 1.9 million closed-ended and 3.7 million open-ended, diverse (audio, question, answer) tuples, and used an autoregressive training framework and a perception-to-understanding curriculum. LTU demonstrates strong performance and generalization ability on conventional audio tasks such as classification and captioning. Moreover, it exhibits remarkable reasoning and comprehension abilities in the audio domain. To the best of our knowledge, LTU is the first audio-enabled large language model that bridges audio perception with advanced reasoning.

**How about the code?**
We plan to release the code but our institute needs to review the software release, we are working on preparing for the review. 