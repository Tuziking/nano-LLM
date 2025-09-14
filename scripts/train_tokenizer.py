from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
import os 
import json

# 读取数据
def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']  


            