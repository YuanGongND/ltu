# -*- coding: utf-8 -*-
# @Time    : 9/6/23 2:05 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ltu2_api_demo.py

from gradio_client import Client
import time

client = Client("https://yuangongfdu-ltu-2.hf.space/")

for repeat in range(3):
    result = client.predict(
                    "./sample_audio.wav",
                    "What can be inferred from the spoken text and sounds? Why?",	# str in 'Edit the textbox to ask your own questions!' Textbox component
                    api_name="/predict"
    )
    print(result)
    # please do add a sleep to avoid long queue, for longer audio, sleep needs to be longer
    time.sleep(10)