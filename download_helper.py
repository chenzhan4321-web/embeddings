#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时下载帮助脚本 - 使用镜像加速下载
"""
import os
import sys

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 80)
print("使用 Hugging Face 镜像加速下载")
print("镜像地址: https://hf-mirror.com")
print("=" * 80)

try:
    from sentence_transformers import SentenceTransformer
    
    model_name = "Qwen/Qwen3-Embedding-4B"
    print(f"\n开始下载模型: {model_name}")
    print("请耐心等待，首次下载约 8GB...\n")
    
    # 使用镜像下载
    model = SentenceTransformer(
        model_name,
        device='cpu',
        trust_remote_code=True,
        cache_folder='./model_cache'  # 指定本地缓存目录
    )
    
    print("\n✅ 模型下载成功！")
    print(f"缓存位置: ./model_cache")
    
except ImportError:
    print("❌ 错误: 请先安装依赖")
    print("运行: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 尝试使用魔法上网")
    print("3. 手动从 https://hf-mirror.com/Qwen/Qwen3-Embedding-4B 下载")
    sys.exit(1)
