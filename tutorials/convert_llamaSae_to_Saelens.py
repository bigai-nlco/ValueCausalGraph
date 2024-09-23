import os
import json
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from sae_lens.config import LOCAL_SAE_MODEL_PATH

#base_dir = '/data02/tuwenming/SAE/Pretrained_SAEs/sae-llama-3-8b-32x-v2/layers.{}/'
base_dir = LOCAL_SAE_MODEL_PATH + '/EleutherAI/sae-llama-3-8b-32x/layers.{}/'

def delete_original_sae(layer_index):
    # 构建文件路径
    layer_dir = base_dir.format(layer_index)
    sae_file_path = os.path.join(layer_dir, 'sae.safetensors')
    
    # 检查文件是否存在
    if os.path.exists(sae_file_path):
        os.remove(sae_file_path)
        print(f"Deleted {sae_file_path}")
    else:
        print(f"File {sae_file_path} does not exist, skipping.")

def process_layer(layer_index):
    # 构建文件路径
    layer_dir = base_dir.format(layer_index)
    sae_file_path = os.path.join(layer_dir, 'sae.safetensors')
    cfg_file_path = os.path.join(layer_dir, 'cfg.json')
    new_sae_file_path = os.path.join(layer_dir, 'sae_weights.safetensors')
    
    # 处理 sae.safetensors 文件
    state_dict = load_file(sae_file_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key
        if "encoder.bias" in key:
            new_key = key.replace("encoder.bias", "b_enc")
        elif "encoder.weight" in key:
            new_key = key.replace("encoder.weight", "W_enc")
            # 对权重进行转置
            new_state_dict[new_key] = state_dict[key].transpose(0, 1).contiguous()
            continue  # 跳过下面的赋值语句，避免重复添加
        
        new_state_dict[new_key] = state_dict[key]
    
    # 保存修改后的state_dict到新的safetensors文件
    save_file(new_state_dict, new_sae_file_path)
    
    # 处理 cfg.json 文件
    with open(cfg_file_path, 'r') as cfg_file:
        cfg_data = json.load(cfg_file)
    
    # 添加新的键值对
    cfg_data.update({
        "d_sae": 131072,
        "dtype": "float32",
        "dataset_path": "togethercomputer/RedPajama-Data-1T-Sample",
        "context_size": 256,
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "hook_name": f"blocks.{layer_index}.hook_resid_post",
        "hook_layer": str(layer_index),
        "hook_head_index": None
    })
    
    # 将修改后的内容写回 cfg.json
    with open(cfg_file_path, 'w') as cfg_file:
        json.dump(cfg_data, cfg_file, indent=4)

# 遍历所有层次 (i 从 0 到 30)
for i in tqdm(range(3,30,1)):
    process_layer(i)
    delete_original_sae(i)