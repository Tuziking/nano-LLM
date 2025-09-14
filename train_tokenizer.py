import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import os

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

random.seed(42)


def train_tokenizer():
    import os
    import json
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    # 读取 JSONL 文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    data_path = './dataset/minimind_dataset/pretrain_hq.jsonl'

    # 初始化 tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 定义特殊 token（加入 pad_token）
    special_tokens = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]

    # 设置训练器并添加特殊 token
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练 tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊 token 是否存在
    for tok in special_tokens:
        assert tokenizer.token_to_id(tok) is not None, f"{tok} not found in vocab"

    # 保存 tokenizer
    tokenizer_dir = "./tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            str(tokenizer.token_to_id(tok)): {
                "content": tok,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
            for tok in special_tokens
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<pad>",          # ✅ 这里显式指定 pad_token
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": (
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\n' + content + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ content + '<|im_end|>\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved with pad_token.")

def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print("生成的对话模板：")
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids)

    print('decoder和原始文本是否一致：', response == new_prompt)
    if response != new_prompt:
        print('⚠️ 注意：解码结果和原始文本可能在空格或换行符上有差异')
    print('解码后的文本：', response)


def main():
    train_tokenizer()
    eval_tokenizer()

if __name__ == '__main__':
    main()