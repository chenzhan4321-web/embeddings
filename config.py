#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding Model Configuration File

Modify MODEL_SIZE here to switch between different model sizes
"""
import os
from sentence_transformers import SentenceTransformer

# ============================================================================
# Model Configuration
# ============================================================================

# Choose model size: "4B" or "8B"
# - 4B: Faster speed, smaller memory footprint (~8GB), only 2% lower performance than 8B
# - 8B: Best performance, #1 on MTEB multilingual leaderboard, requires more memory (~16GB)
MODEL_SIZE = "4B"  # Default to 4B

# ============================================================================
# Download Mirror Configuration (speeds up downloads in China)
# ============================================================================

USE_MODELSCOPE = True  # Whether to use ModelScope mirror for accelerated downloads

# ModelScope model mapping (HuggingFace -> ModelScope)
MODELSCOPE_MODELS = {
    "Qwen/Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B"
}

# ============================================================================
# Model Details (no modification needed)
# ============================================================================

MODEL_CONFIGS = {
    "4B": {
        "name": "Qwen/Qwen3-Embedding-4B",
        "params": "4B",
        "model_size": "~8GB",
        "memory_required": "8-10GB",
        "c_mteb_score": 72.27,
        "mteb_score": 69.45,
        "description": "Medium size, excellent performance, suitable for most users"
    },
    "8B": {
        "name": "Qwen/Qwen3-Embedding-8B",
        "params": "8B",
        "model_size": "~16GB",
        "memory_required": "16GB+",
        "c_mteb_score": 73.84,
        "mteb_score": 70.58,
        "description": "Largest model, #1 on MTEB leaderboard, requires more resources"
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_model_name():
    """Get the currently configured model name"""
    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {MODEL_SIZE}. Please choose '4B' or '8B'")
    return MODEL_CONFIGS[MODEL_SIZE]["name"]


def get_model_info():
    """Get detailed information about the current model"""
    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {MODEL_SIZE}. Please choose '4B' or '8B'")
    return MODEL_CONFIGS[MODEL_SIZE]


def load_model(device='cuda', **kwargs):
    """
    Load Qwen3-Embedding model (automatically uses ModelScope mirror for acceleration)

    Args:
        device: Device type, 'cuda' or 'cpu'
        **kwargs: Additional parameters passed to SentenceTransformer

    Returns:
        SentenceTransformer model instance
    """
    model_name = get_model_name()

    # If ModelScope mirror is enabled
    if USE_MODELSCOPE:
        try:
            print(f"🚀 Using ModelScope mirror to download model: {model_name}")
            print("   (Faster for users in China, usually completes in 1-3 minutes)")

            # Use modelscope's snapshot_download to pre-download model
            from modelscope import snapshot_download
            modelscope_name = MODELSCOPE_MODELS.get(model_name, model_name)

            # Download model to local cache
            cache_dir = snapshot_download(modelscope_name, cache_dir=None)
            print(f"✅ Model download complete: {cache_dir}")

            # Load model from local cache
            model = SentenceTransformer(
                cache_dir,
                device=device,
                trust_remote_code=True,
                **kwargs
            )
            print(f"✅ Model loaded successfully!")
            return model

        except Exception as e:
            print(f"⚠️  ModelScope download failed: {e}")
            print(f"   Trying to download from HuggingFace official source...")

    # Fallback to HuggingFace official source
    print(f"Loading model from HuggingFace: {model_name}")
    print("   (If slow, consider installing modelscope: pip install modelscope)")
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        **kwargs
    )
    print(f"✅ Model loaded successfully!")
    return model


def print_model_info():
    """Print current model configuration information"""
    info = get_model_info()
    print("=" * 80)
    print("Model Configuration Information")
    print("=" * 80)
    print(f"Model Name: {info['name']}")
    print(f"Parameters: {info['params']}")
    print(f"Model Size: {info['model_size']}")
    print(f"Memory Required: {info['memory_required']}")
    print(f"C-MTEB Chinese Score: {info['c_mteb_score']}")
    print(f"MTEB Multilingual Score: {info['mteb_score']}")
    print(f"Description: {info['description']}")
    print("=" * 80)


# ============================================================================
# Model Comparison Information
# ============================================================================

def print_model_comparison():
    """Print model comparison information"""
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"{'Metric':<20} {'4B':<25} {'8B':<25}")
    print("-" * 80)
    print(f"{'Parameters':<20} {'4B':<25} {'8B':<25}")
    print(f"{'Model Size':<20} {'~8GB':<25} {'~16GB':<25}")
    print(f"{'Memory Required':<20} {'8-10GB':<25} {'16GB+':<25}")
    print(f"{'C-MTEB Chinese':<20} {'72.27':<25} {'73.84':<25}")
    print(f"{'MTEB Multilingual':<20} {'69.45':<25} {'70.58':<25}")
    print(f"{'Relative Speed':<20} {'1.4x (40% faster)':<25} {'1.0x (baseline)':<25}")
    print(f"{'Download Time':<20} {'~12 minutes':<25} {'~25 minutes':<25}")
    print("-" * 80)
    print(f"{'Recommended For':<20}")
    print(f"  4B: Most users, balances performance and resource usage")
    print(f"  8B: Best performance seekers with sufficient GPU memory")
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    print_model_info()
    print("\n")
    print_model_comparison()
    print(f"\nCurrently selected model: {get_model_name()}")
