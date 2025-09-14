from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any
import wandb
from tqdm import tqdm

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,llm_model_path = 'model/Qwen2.5-0.5B-Instruct',
                 vision_model_path = 'model/siglip2-so400m-patch14-384',
                 freeze_vision_model = True,
                 image_pad_num = 81,
                **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)
        
        
        
class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)

        # --- 动态计算 ---
        # 获取视觉模型的输出维度
        vision_hidden_size = self.vision_model.config.vision_config.hidden_size
        
        # 计算图像块的数量（即序列长度）
        image_size = self.vision_model.config.vision_config.image_size
        patch_size = self.vision_model.config.vision_config.patch_size
        num_patches = (image_size // patch_size) ** 2

        # 压缩因子 = 原始块数量 / 目标Token数量
        # 断言确保可以整除
        assert num_patches % self.config.image_pad_num == 0, \
            f"图像块数量 ({num_patches}) 无法被 image_pad_num ({self.config.image_pad_num}) 整除。"
        
        compression_factor = num_patches // self.config.image_pad_num
        
        # 根据动态计算结果来定义线性层
        projector_input_size = vision_hidden_size * compression_factor
        llm_hidden_size = self.llm_model.config.hidden_size
        
        self.linear1 = nn.Linear(projector_input_size, llm_hidden_size)
        # --- 动态计算结束 ---

        self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
                
        # 在预训练阶段，通常也会冻结LLM，只训练投影层（线性层）
        # 您原始的代码就是这样做的
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        # image_embeds 的形状: (b, num_patches, vision_hidden_size) -> 例如 (64, 729, 1152)
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 
        b, s, d = image_embeds.shape
        
        # 像 __init__ 中一样，再次计算压缩因子
        compression_factor = s // self.config.image_pad_num
        
        # 动态重塑: (b, s, d) --> (b, s/k, d*k)
        # 例如: (64, 729, 1152) --> (64, 81, 1152 * 9)
        image_embeds = image_embeds.view(b, self.config.image_pad_num, d * compression_factor)
        
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        text_embeds = text_embeds.to(image_features.dtype)
        
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        # 这个方法的实现逻辑是正确的，它找到所有图像占位符的位置，并用计算出的图像特征替换它们
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        # 确保占位符的数量和图像特征的数量匹配
        assert len(batch_indices) == num_images * num_image_patches, "占位符数量与图像特征数量不匹配"

        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        
        return inputs_embeds
    
class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        print("正在加载和过滤数据，请稍候...")
        with open(data_path, 'r', encoding='utf-8') as f:
            original_datas = json.load(f)

        self.datas = []
        # 使用 tqdm 显示过滤进度
        for sample in tqdm(original_datas, desc="过滤数据集中"):
            image_name = sample.get('image') # 安全获取
            if image_name and image_name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                # 顺便检查文件是否存在，一举两得
                full_path = os.path.join(self.images_path, image_name)
                if os.path.exists(full_path):
                    self.datas.append(sample)
        
        print(f"数据过滤完成。原始数据量: {len(original_datas)}, 有效数据量: {len(self.datas)}")
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
            
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 
     

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}
            
        
        
if __name__ == '__main__':
    config = VLMConfig(image_pad_num=81)
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'saves/multi_mode_pretrain' 
    wandb.init(project="nanoLLM-V", name="pretrain_multi_mode")
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=128,
        learning_rate=1e-4,
        num_train_epochs=1,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=1,
        logging_steps=100,
        report_to='wandb',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('saves/pretrain_multi_mode')
    trainer.save_state()
    
    

    
    