from modelscope.hub.snapshot_download import snapshot_download

# 定义数据集信息
# dataset_id = 'ddzhu123/seq-monkey'
# save_dir = './dataset/'
# file_to_download = 'mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2'

# 定义模型信息
model_id = 'google/siglip-so400m-patch14-384'
save_dir = './model/siglip-so400m-patch14-384'

# 使用 snapshot_download 函数进行下载
# 使用 allow_patterns 参数指定要下载的文件，支持通配符
local_path = snapshot_download(
    repo_id=model_id,
    repo_type='model',            # 明确指定是数据集
    local_dir=save_dir,             # 下载到指定目录
    # allow_patterns=[file_to_download] # 使用列表指定允许下载的文件模式
)

print(f"仓库 '{model_id}'  已下载到: {save_dir}")
print(f"返回的本地仓库路径为: {local_path}")