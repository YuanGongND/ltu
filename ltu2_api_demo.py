# -*- coding: utf-8 -*-
# @Time    : 9/6/23 2:05 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : api_demo.py

# method 1

# import json
# import gradio as gr
# import requests
# import os
#
# def is_file_larger_than_30mb(file_path):
#     try:
#         file_size = os.path.getsize(file_path)
#         return file_size > (30 * 1024 * 1024)
#     except FileNotFoundError:
#         return False
#     except PermissionError:
#         return False
#     except Exception as e:
#         return False
#
# def upload_audio(audio_path):
#     try:
#         size = is_file_larger_than_30mb(audio_path)
#         if size == True:
#             return 'size'
#         with open(audio_path, 'rb') as audio_file:
#             response = requests.post('http://sls-titan-6.csail.mit.edu:8080/upload/', files={'audio_file': audio_file})
#         if response.status_code == 200:
#             return response.json()["path"]
#     except:
#         return None
#
# def predict(audio_path, question):
#     upload_statues = upload_audio(audio_path)
#     if upload_statues == None:
#         return 'Please upload an audio file.'
#     if upload_statues == 'size':
#         return 'This demo does not support audio file size larger than 30MB.'
#     if question == '':
#         return 'Please ask a question.'
#     print(audio_path, question)
#     response = requests.put('http://sls-titan-6.csail.mit.edu:8080/items/0', json={
#         'audio_path': audio_path, 'question': question
#     })
#     answer = json.loads(response.content)
#     ans_str = answer['output']
#     return ans_str
#
# ans = predict('./sample_audio.wav', 'What did you hear?')
# print(ans)

# # method 2

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