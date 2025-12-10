import torch
from timm.models import create_model
import sys, os
from transformers import XLMRobertaTokenizer
import torch.nn.functional as F # 导入 functional

# --- 配置区 ---
# 确保 MUSK 库的路径正确
musk_lib_path = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/MUSK-main"
if musk_lib_path not in sys.path:
    sys.path.insert(0, musk_lib_path)

try:
    from musk import utils, modeling # 确保能导入 utils
except ImportError:
    print(f"错误：无法从 '{musk_lib_path}' 导入 musk 库。请检查路径是否正确。")
    sys.exit(1)

# 设备设置
device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 模型和文件路径
model_config = "musk_large_patch16_384"
local_ckpt = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/checkpoint/model.safetensors"
# 使用你环境中 tokenizer.spm 的正确路径
tokenizer_path = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/MUSK-main/musk/models/tokenizer.spm"
# 输出文件路径
save_path = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/MUSK-text-feature/averaged_text_feature.pt" # 给新文件一个清晰的名字

# 分词器最大长度 (参照 Demo，这个值可能解决了之前的 CUDA 错误)
max_token_len = 100
print(f"使用的最大Token长度: {max_token_len}")

# 你的目标文本提示
prompts = [
    "histopathology image of liver tissue with hepatic lobular inflammation",
    "non-alcoholic steatohepatitis showing mild lobular inflammation",
    "grade 2 lobular inflammation (2 foci per 20x field)",       #肝小叶炎症
]
# --- 配置区结束 ---


# --- 1. 加载模型 ---
print(f"加载模型: {model_config}")
model = create_model(model_config).eval() # 先创建模型
print(f"加载检查点: {local_ckpt}")
try:
    # 加载权重
    utils.load_model_and_may_interpolate(local_ckpt, model, 'model|module', '')
    # 加载权重后移动到设备并设置精度
    model.to(device, dtype=torch.float16)
    model.eval()
    print("模型加载并移动到设备成功。")
except FileNotFoundError:
    print(f"错误: 检查点文件未找到 '{local_ckpt}'")
    sys.exit(1)
except Exception as e:
    print(f"加载模型时出错: {e}")
    sys.exit(1)

# --- 2. 加载分词器 ---
print(f"加载分词器: {tokenizer_path}")
try:
    tokenizer = XLMRobertaTokenizer(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"分词器加载成功。词汇表大小: {vocab_size}")
except Exception as e:
    print(f"加载分词器时出错: {e}")
    sys.exit(1)

# --- 3. 处理文本提示 (使用 utils.xlm_tokenizer) ---
print(f"正在使用 utils.xlm_tokenizer 处理 {len(prompts)} 个文本提示...")
all_text_ids = []
all_paddings = []
try:
    for i, txt in enumerate(prompts):
        # 调用 musk.utils 中的分词函数
        txt_ids, pad = utils.xlm_tokenizer(txt, tokenizer, max_len=max_token_len)

        # 检查生成的 ID 是否越界 (可选但推荐)
        max_id = max(txt_ids)
        min_id = min(txt_ids)
        if max_id >= vocab_size or min_id < 0:
            print(f"警告: 提示 {i} ('{txt[:30]}...') 生成的 ID ({min_id}-{max_id}) 超出词汇表范围 [0, {vocab_size-1}]！")
            # 这里可以选择跳过这个提示或停止脚本，取决于你的需求
            # continue # 选择跳过

        # 添加 batch 维度并收集
        all_text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
        all_paddings.append(torch.tensor(pad).unsqueeze(0))

    if not all_text_ids:
        print("错误：没有成功处理任何文本提示。")
        sys.exit(1)

    # 将列表中的 tensors 合并成一个 batch
    batch_text_ids = torch.cat(all_text_ids).to(device)
    batch_paddings = torch.cat(all_paddings).to(device) # 这个是 padding mask
    print(f"Tokenized IDs 形状: {batch_text_ids.shape}")
    print(f"Padding mask 形状: {batch_paddings.shape}")

except Exception as e:
    print(f"文本处理或分词时出错: {e}")
    sys.exit(1)

# --- 4. 提取文本嵌入 ---
print("正在模型中提取文本嵌入...")
try:
    with torch.inference_mode():
        # 输入模型，获取文本嵌入 (输出元组的第 [1] 个元素)
        # 注意 padding_mask 参数使用的是 utils.xlm_tokenizer 返回的 pad
        text_embeddings_per_prompt = model(
            text_description=batch_text_ids,
            padding_mask=batch_paddings, # 使用 utils 处理得到的 padding mask
            with_head=True,
            out_norm=True # 假设模型内部会做归一化
        )[1] # return (vision_cls, text_cls)，我们取 text_cls
    print(f"提取的各提示嵌入形状: {text_embeddings_per_prompt.shape}") # 应该是 [N_prompts, embedding_dim]
except Exception as e:
    print(f"模型推理时出错: {e}")
    # 再次捕获可能的 CUDA 错误
    if "CUDA error: device-side assert triggered" in str(e):
         print("模型推理时再次触发 CUDA assert。请检查：")
         print(f"- max_len ({max_token_len}) 是否仍然过大？")
         print("- 分词器与模型检查点是否绝对匹配？")
         print("- 运行环境（PyTorch, CUDA 版本）是否兼容？")
    sys.exit(1)

# --- 5. 计算平均特征并归一化 ---
print("计算平均特征向量并进行归一化...")
# 沿着 prompt 维度（dim=0）计算平均值，保持维度为 1 (形状变为 [1, embedding_dim])
averaged_embedding = text_embeddings_per_prompt.mean(dim=0, keepdim=True)

# 对平均后的向量进行 L2 归一化
# 注意：如果模型设置 out_norm=True 已经对每个 prompt 的输出做了归一化,
# 对平均向量再次归一化在数学上不等同于先平均再归一化，但通常是可接受的近似。
# 如果 out_norm=False，则这一步是必须的。
final_feature = F.normalize(averaged_embedding, dim=-1)
print(f"最终平均特征向量形状: {final_feature.shape}") # 应该是 [1, embedding_dim]

# --- 6. 保存特征 ---
# 移动到 CPU 并转换为半精度（如果需要节省空间）
feature_to_save = final_feature.cpu().half()
print(f"准备保存的特征数据类型: {feature_to_save.dtype}")

# 确保输出目录存在
output_dir = os.path.dirname(save_path)
os.makedirs(output_dir, exist_ok=True)

print(f"保存特征到: {save_path}")
try:
    torch.save(feature_to_save, save_path)
    print("特征保存成功！")
except Exception as e:
    print(f"保存特征时出错: {e}")
    sys.exit(1)