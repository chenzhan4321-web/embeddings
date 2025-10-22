#!/bin/bash
# 快速修复下载问题的脚本

echo "=========================================="
echo "Qwen3-Embedding 下载问题修复脚本"
echo "=========================================="
echo ""

# 1. 设置镜像环境变量
echo "✅ 步骤 1: 设置 Hugging Face 镜像"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
echo "   镜像地址: $HF_ENDPOINT"
echo "   下载超时: 600秒"
echo ""

# 2. 检查依赖
echo "✅ 步骤 2: 检查依赖包"
if python -c "import sentence_transformers" 2>/dev/null; then
    echo "   ✓ sentence-transformers 已安装"
else
    echo "   ✗ sentence-transformers 未安装"
    echo "   正在安装..."
    pip install -q sentence-transformers
fi
echo ""

# 3. 检查网络
echo "✅ 步骤 3: 检查网络连接"
if curl -s -I https://hf-mirror.com > /dev/null; then
    echo "   ✓ 可以访问 hf-mirror.com"
else
    echo "   ✗ 无法访问 hf-mirror.com，请检查网络"
    exit 1
fi
echo ""

# 4. 检查磁盘空间
echo "✅ 步骤 4: 检查磁盘空间"
FREE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
echo "   可用空间: $FREE_SPACE"
echo ""

# 5. 运行测试
echo "✅ 步骤 5: 开始下载和测试"
echo "   这可能需要几分钟，请耐心等待..."
echo "=========================================="
echo ""

python 01_basic_usage.py

echo ""
echo "=========================================="
echo "如果还是卡住，请查看 fix_download.md 文件"
echo "=========================================="
