import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import SFTDataset  # ⚠️ 改成你保存SFTDataset类的文件路径

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you! How can I help you today?"}
]

answer = messages.pop()['content']  # 提取最后一条消息作为答案
print("Answer:", answer)
print("Messages:", messages)