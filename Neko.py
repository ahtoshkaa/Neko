import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
from openai import OpenAI
     
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

system = open_file('system.txt')

messages = [
    {"role": "system", "content": system},
]
     
XAI_API_KEY = open_file('key_xai.txt').strip()
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)


#os.environ["OPENAI_API_KEY"] = open_file('key_openai.txt').strip()
#client = OpenAI()


while True:
    message = input('\n\nUser: ')
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        response = client.chat.completions.create(
            model="grok-beta", 
            messages=messages,
            stream=True  # Enable streaming
        )
        
        collected_messages = []
        print("Neko: ", end='', flush=True)
        
        # Process each chunk as it arrives
        for chunk in response:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                collected_messages.append(chunk_message)
                print(chunk_message, end='', flush=True)
        
        full_reply_content = ''.join(collected_messages)
        #print(f"Neko: {full_reply_content}")

        # Add the entire reply to the messages
        messages.append({"role": "assistant", "content": full_reply_content})
    
    # Wait for 1 second before making the next API request
    sleep(1)