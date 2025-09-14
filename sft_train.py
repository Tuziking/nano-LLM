import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset, LLMDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from moe_train import LLM, Config
import wandb

if __name__ == '__main__':
    AutoConfig.register("moe_model", Config)
    # AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained('./saves/moe_model1_6epoch')
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    wandb.init(project="nanoLLM", name="sft_moe_model1_512")
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./saves/moe_model1_6epoch", use_fast=True)
    args = TrainingArguments(output_dir='./sft/sft_moe_model1_512', 
                            num_train_epochs=2, 
                            do_train=True, 
                            per_device_train_batch_size=256,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=100,
                            report_to='wandb',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=5e-7,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)          
    dataset = SFTDataset('./dataset/minimind_dataset/sft_mini_512.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/sft_moe_model1_512')
    trainer.save_state()
    # trainer.save_pretrained('./saves/sft_model1_512/tokenizer')