import json
import numpy as np
import torch
from torch.utils.data import Dataset

class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 如果 tokenizer 没有 pad_token，手动设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        line = json.loads(self.data[index])

        # 数据里已经包含特殊token，就不要再加了
        text = line['text']

        # 转 input_ids
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
        input_ids = np.array(input_ids)
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
        }

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        line = json.loads(line)
        conversations = line['conversations']
        
        messages = []
        for conv in conversations:
            messages.append({'role': conv['role'], 'content': conv['content']})
        
        # 构建对话文本，弹出最后一条作为answer
        answer = messages.pop()['content'] + self.tokenizer.eos_token
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False) 

        # 转 input_ids
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        input_ids = prompt_input_ids + answer_input_ids
        labels = [0] * len(prompt_input_ids) + answer_input_ids
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
            labels = labels + [0] * (self.max_seq_len - text_len)
        
        input_ids = input_ids[:-1]
        labels = labels[1:]
        return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
        
    def __getitem__(self, index):
        sample = self.datas[index]
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(text=text)['input_ids']
        rejected_inputs = self.tokenizer(text=rejected)['input_ids'] + [self.tokenizer.eos_token_id]
        chosen_inputs = self.tokenizer(text=chosen)['input_ids'] + [self.tokenizer.eos_token_id]
        return [prompt_inputs, chosen_inputs, rejected_inputs]
    
    def __len__(self):
        return len(self.datas)
    
    
class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, features):
        inputs_ids = []
        labels = []
        
        for feature in features:
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0]*len(feature[0]) + feature[1])
        for feature in features:
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0]*len(feature[0]) + feature[2])
            
        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []
            
            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels
        
        inputs_ids, labels = process(inputs_ids, labels)
        
        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels)
            }