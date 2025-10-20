# text_utils.py
import torch
import clip
import os # <--- 导入 os 模块

def get_label_text_embeddings(class_names, device):
    """
    使用 CLIP 的文本编码器为给定的类名生成文本嵌入 (LTE)。
    此版本会从一个指定的本地路径加载模型。
    """
    print("正在使用 CLIP 生成标签文本嵌入 (LTE)...")
    
    # --- VVV 核心修改点 VVV ---
    # 定义您手动下载的模型权重文件的路径
    # 假设 ViT-B-32.pt 文件与 main.py 在同一目录下
    model_path = "./model/ViT-B-32.pt" 

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'。")
        print("请确保您已手动下载 ViT-B-32.pt 并将其放置在正确的路径下。")
        return None

    try:
        # clip.load() 的第一个参数可以直接接收一个本地文件路径
        model, _ = clip.load(model_path, device=device)
        print(f"成功从本地路径 '{model_path}' 加载 CLIP 模型。")

    except Exception as e:
        print(f"从本地加载 CLIP 模型失败: {e}")
        return None

    # 根据论文建议，使用模板 "A photo of a {class_name}"
    prompts = [f"a photo of a {name}" for name in class_names]
    
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_embeddings = model.encode_text(text_tokens)
    
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    print("LTE 生成完毕。")
    return text_embeddings