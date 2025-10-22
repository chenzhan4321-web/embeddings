# 4B 模型下载卡住的解决方案

## 问题诊断
模型下载停在 `Fetching 2 files: 0%` 通常是因为：
1. 网络连接到 Hugging Face 不稳定
2. 没有使用国内镜像
3. 下载超时设置太短

## 解决方案（按推荐顺序）

### ✅ 方案 1：使用国内镜像（最推荐）

```bash
# 设置环境变量（临时，当前终端有效）
export HF_ENDPOINT=https://hf-mirror.com

# 或者永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc

# 然后重新运行脚本
python 01_basic_usage.py
```

### ✅ 方案 2：使用 ModelScope（国内最快）

修改 `config.py`，添加 ModelScope 支持：

```python
# config.py 新增代码
MODEL_SOURCE = "modelscope"  # 或 "huggingface"

def get_model_name():
    if MODEL_SOURCE == "modelscope":
        # ModelScope 镜像（国内极快）
        if MODEL_SIZE == "4B":
            return "Qwen/Qwen3-Embedding-4B"  # ModelScope 同名
        else:
            return "Qwen/Qwen3-Embedding-8B"
    else:
        # Hugging Face 原站
        if MODEL_SIZE == "4B":
            return "Qwen/Qwen3-Embedding-4B"
        else:
            return "Qwen/Qwen3-Embedding-8B"
```

然后安装 ModelScope：
```bash
pip install modelscope
```

使用 ModelScope 下载：
```python
from modelscope import snapshot_download

# 下载模型到本地
model_dir = snapshot_download('Qwen/Qwen3-Embedding-4B', cache_dir='./model_cache')

# 使用本地模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_dir, device='cuda', trust_remote_code=True)
```

### ✅ 方案 3：手动下载（最稳定）

1. 访问镜像站: https://hf-mirror.com/Qwen/Qwen3-Embedding-4B/tree/main
2. 下载以下文件到 `./local_model/Qwen3-Embedding-4B/` 目录：
   - `config.json`
   - `model.safetensors`（主要模型文件，约 8GB）
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `special_tokens_map.json`
   - `vocab.txt`

3. 修改代码使用本地路径：
```python
model = SentenceTransformer('./local_model/Qwen3-Embedding-4B', trust_remote_code=True)
```

### ✅ 方案 4：增加超时和重试

创建 `~/.cache/huggingface/accelerate/default_config.yaml`：
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
use_cpu: false
```

设置环境变量：
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5分钟超时
export TRANSFORMERS_OFFLINE=0       # 确保在线模式
```

### ✅ 方案 5：切换到更小的模型测试

如果急需测试，可以先用更小的模型：

```python
# 使用官方的小模型测试（~400MB）
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

## 检查网络连接

```bash
# 测试能否访问 Hugging Face
curl -I https://huggingface.co

# 测试能否访问镜像
curl -I https://hf-mirror.com

# 查看下载进度（另开终端）
watch -n 1 'du -sh ~/.cache/huggingface/'
```

## 推荐组合方案

**最快速（国内用户）**:
```bash
# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 增加超时
export HF_HUB_DOWNLOAD_TIMEOUT=600

# 3. 重新运行
python 01_basic_usage.py
```

## 如果还是不行

可能需要：
1. 检查防火墙设置
2. 使用代理/VPN
3. 检查磁盘空间（至少需要 10GB）
4. 尝试在非高峰时段下载
5. 使用迅雷等下载工具手动下载大文件

## 验证下载成功

```bash
# 检查缓存目录大小
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B

# 应该显示约 8GB
```
