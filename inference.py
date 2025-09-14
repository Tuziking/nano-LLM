from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from pretrain_1 import LLM, Config
# from moe_train import LLM, Config

# t = AutoTokenizer.from_pretrained('saves/sft_moe_model1_512')
t = AutoTokenizer.from_pretrained('saves/sft_model1_512')
# AutoConfig.register("moe_model", Config)
AutoConfig.register("small_model", Config)
AutoModelForCausalLM.register(Config, LLM)
# model = AutoModelForCausalLM.from_pretrained('saves/sft_moe_model1_512')
model = AutoModelForCausalLM.from_pretrained('saves/sft_model1_512')
print("=" * 50)
print(f"👤 用户: 中国的首都是")
print("🤖 Dense Model: ", end="")
# input_data = [t.bos_token_id] + t.encode('1+1等于')
# print(input_data)
for output in model.generate(
    {"input_ids": torch.tensor([[t.bos_token_id] + t.encode("中国的首都是")]),
     "labels": None},
    eos=t.eos_token_id,
    max_new_tokens=100,
    temperature=0,
    top_k=10,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,  # 不允许重复 3-gram
    stream=False
):
    print(t.decode(output[0], skip_special_tokens=True))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# from moe_train import LLM, Config
from pretrain_1 import LLM, Config


t = AutoTokenizer.from_pretrained("saves/sft_model1_512")
AutoConfig.register("small_model", Config)
# AutoConfig.register("moe_model", Config)
AutoModelForCausalLM.register(Config, LLM)
model = AutoModelForCausalLM.from_pretrained("saves/sft_model1_512")
print("=" * 50)
print(f"👤 用户: 中国的首都是")
print("🤖 Dense Model SFT: ", end="")
input_data = t.apply_chat_template([{'role':'user', 'content':'中国的首都是'}])
# print(input_data)
# print(t.decode(torch.tensor(input_data)))
for output in model.generate(
    {"input_ids": torch.tensor(input_data).unsqueeze(0),
     "labels": None},
    eos=t.eos_token_id,
    max_new_tokens=200,
    temperature=0,
    top_k=8,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,  # 不允许重复 3-gram
    stream=False
):
    print(t.decode(output[0], skip_special_tokens=True))



from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# from pretrain_1 import LLM, Config
from moe_train import LLM, Config

t = AutoTokenizer.from_pretrained('saves/moe_model1_6epoch')
# t = AutoTokenizer.from_pretrained('saves/sft_model1_512')
AutoConfig.register("moe_model", Config)
# AutoConfig.register("small_model", Config)
AutoModelForCausalLM.register(Config, LLM)
model = AutoModelForCausalLM.from_pretrained('saves/moe_model1_6epoch')
print("=" * 50)
print(f"👤 用户: 中国的首都是")
print("🤖 MOE Model: ", end="")
for output in model.generate(
    {"input_ids": torch.tensor([[t.bos_token_id] + t.encode("中国的首都是")]),
     "labels": None},
    eos=t.eos_token_id,
    max_new_tokens=100,
    temperature=0,
    top_k=10,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,  # 不允许重复 3-gram
    stream=False
):
    print(t.decode(output[0], skip_special_tokens=True))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from moe_train import LLM, Config
# from pretrain_1 import LLM, Config


t = AutoTokenizer.from_pretrained("saves/sft_moe_model1_512")
# AutoConfig.register("small_model", Config)
AutoConfig.register("moe_model", Config)
AutoModelForCausalLM.register(Config, LLM)
model = AutoModelForCausalLM.from_pretrained("saves/sft_moe_model1_512")
print("=" * 50)
print(f"👤 用户: 中国的首都是")
print("🤖 MOE Model SFT: ", end="")
input_data = t.apply_chat_template([{'role':'user', 'content':'中国的首都是'}])
# print(input_data)
# print(t.decode(torch.tensor(input_data)))
for output in model.generate(
    {"input_ids": torch.tensor(input_data).unsqueeze(0),
     "labels": None},
    eos=t.eos_token_id,
    max_new_tokens=200,
    temperature=0,
    top_k=8,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,  # 不允许重复 3-gram
    stream=False
):
    print(t.decode(output[0], skip_special_tokens=True))