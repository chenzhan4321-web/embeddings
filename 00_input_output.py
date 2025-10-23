#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简单的 Embedding Pipeline: 输入文本 -> 输出向量
"""
import config  # 自动启用镜像加速
from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 80)
print("Qwen3-Embedding 文本向量化 Pipeline")
print("=" * 80)
print()

# 加载模型（首次运行会自动下载，约 8GB）
print(f"正在加载模型: {config.get_model_name()}")
model = SentenceTransformer(config.get_model_name(), device='cpu', trust_remote_code=True)
print("✅ 模型加载完成！")
print()

print("=" * 80)
print("使用说明:")
print("  - 输入任意文本，按回车生成向量")
print("  - 输入 'quit' 或 'exit' 退出程序")
print("=" * 80)
print()

# 交互循环
while True:
    # 等待用户输入（回车结束）
    user_input = input("请输入文本: ").strip()
    
    # 退出条件
    if user_input.lower() in ['quit', 'exit', 'q', '']:
        print("\n👋 再见！")
        break
    
    # 生成向量
    embedding = model.encode(user_input, convert_to_numpy=True)
    
    # 输出结果
    print()
    print("-" * 80)
    print(f"输入文本: {user_input}")
    print(f"向量维度: {embedding.shape[0]}")
    print(f"向量范数: {np.linalg.norm(embedding):.4f}")
    print()
    print("向量前 10 维:")
    print(embedding[:10])
    print()
    print("完整向量:")
    print(embedding)
    print("-" * 80)
    print()
