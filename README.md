# Qwen3-Embedding Demo

A comprehensive demonstration project showcasing the capabilities of the Qwen3-Embedding model (ranked #1 on MTEB multilingual leaderboard) for text embedding, semantic similarity calculation, semantic search, and text clustering visualization.

## Overview

This project demonstrates practical applications of state-of-the-art text embedding models through four progressive example scripts. It supports both the 4B and 8B versions of Qwen3-Embedding, offering flexibility between performance and resource requirements.

## Features

- **Multiple Model Support**: Choose between 4B (recommended) and 8B models via simple configuration
- **Multilingual Capabilities**: Supports 100+ languages including Chinese, English, French, and German
- **Interactive Examples**: Four progressively advanced demonstration scripts
- **Visualization**: Interactive 2D/3D clustering visualizations with Plotly
- **Optimized Performance**: Supports GPU acceleration, batch processing, and half-precision inference
- **Fast Downloads**: ModelScope mirror support for accelerated downloads in China

## Model Information

### Qwen3-Embedding Series

| Feature | 4B Model (Default) | 8B Model |
|---------|-------------------|----------|
| Parameters | 4B | 8B |
| Output Dimension | 4096 | 4096 |
| Supported Languages | 100+ | 100+ |
| MTEB Multilingual Score | 69.45 | 70.58 (#1) |
| C-MTEB Chinese Score | 72.27 | 73.84 |
| Model Size | ~8GB | ~16GB |
| Memory Required | 8-10GB | 16GB+ |
| Inference Speed | 1.4x (40% faster) | 1.0x (baseline) |
| Download Time | ~12 minutes | ~25 minutes |

**Recommendation**:
- **4B Model**: Recommended for most users - excellent performance (98% of 8B) with lower resource requirements
- **8B Model**: For users seeking the absolute best performance with sufficient GPU memory

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ GPU memory for 4B model, 16GB+ for 8B model

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd test.embedding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional but recommended) Install UMAP for better dimensionality reduction:
```bash
pip install umap-learn
```

4. (Optional) For faster downloads in China, install ModelScope:
```bash
pip install modelscope
```

## Configuration

### Model Selection

Edit `config.py` to choose your model size:

```python
# Choose model size: "4B" or "8B"
MODEL_SIZE = "4B"  # Default - recommended for most users
# MODEL_SIZE = "8B"  # For ultimate performance
```

View current configuration:
```bash
python config.py
```

### Download Acceleration

For users in China, the project automatically uses ModelScope mirror for faster downloads. You can also manually set the Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

### Example Scripts

Run the four example scripts in order to understand the progression from basic to advanced features:

#### 1. Basic Usage (01_basic_usage.py)

Learn the fundamentals of model loading and embedding generation:

```bash
python 01_basic_usage.py
```

**What it demonstrates:**
- Loading the Qwen3-Embedding model
- Generating embeddings for text
- Understanding embedding properties (dimensions, norms, distributions)
- Example texts in multiple languages (Chinese, English, French, German)

#### 2. Similarity Calculation (02_similarity_calculation.py)

Explore semantic similarity between texts:

```bash
python 02_similarity_calculation.py
```

**What it demonstrates:**
- Computing cosine similarity between text embeddings
- Same-language and cross-language similarity comparison
- Similarity matrix visualization
- Finding the most similar texts from a candidate set

**Key functions:**
- `print_similarity_matrix()`: Pretty-print similarity matrices
- `find_most_similar()`: Find top-k most similar candidates

#### 3. Semantic Search (03_semantic_search.py)

Build and use a semantic search engine:

```bash
python 03_semantic_search.py
```

**What it demonstrates:**
- Indexing a document collection
- Performing semantic search queries
- Interactive search loop (enter queries, get results)
- Cross-topic semantic matching

**Features:**
- Pre-computes document embeddings for fast search
- Supports interactive queries
- Returns top-k most relevant documents with scores
- Exit with 'quit', 'exit', 'q' or Ctrl+C

#### 4. Text Clustering Visualization (04_text_clustering_visualization.py)

Visualize text clusters in 2D and 3D:

```bash
python 04_text_clustering_visualization.py
```

**What it demonstrates:**
- Automatic text clustering with K-means
- Dimensionality reduction (t-SNE and UMAP)
- Interactive 2D and 3D visualizations
- Comparison between true labels and clustering results

**Generated visualizations:**
- `clustering_2d_by_label.html` - 2D plot colored by true categories
- `clustering_2d_by_cluster.html` - 2D plot colored by cluster assignments
- `clustering_3d_by_label.html` - 3D plot colored by true categories
- `clustering_3d_by_cluster.html` - 3D plot colored by cluster assignments
- `clustering_umap_2d.html` - 2D UMAP visualization (if UMAP installed)
- `clustering_umap_3d.html` - 3D UMAP visualization (if UMAP installed)

All visualizations are interactive HTML files - open in your browser to explore with hover, zoom, and rotation.

## Example Data

The project includes example texts in multiple languages:

- **Chinese**: 机器学习、深度学习、自然语言处理 (Machine Learning, Deep Learning, NLP)
- **English**: "Machine learning", "Deep learning", "Natural language processing"
- **French**: "Apprentissage automatique", "Apprentissage profond"
- **German**: "Maschinelles Lernen", "Tiefes Lernen"

The semantic search demo includes a diverse document collection covering technology, programming, lifestyle, and business topics.

## Application Scenarios

1. **Semantic Search**: Document retrieval, knowledge base Q&A
2. **Text Classification**: Similarity-based classification
3. **Recommendation Systems**: Content and article recommendations
4. **Clustering Analysis**: Topic discovery, content grouping
5. **Deduplication**: Identifying similar or duplicate content
6. **Cross-language Retrieval**: Multilingual document matching

## Advanced Usage

### Custom Model Loading

```python
import config

# Load model with configuration
model = config.load_model(device='cuda')

# Generate embeddings
texts = ["Your text here", "Another text"]
embeddings = model.encode(texts, convert_to_tensor=True)
```

### Performance Optimization

```python
# Enable half-precision for memory savings
model.half()

# Batch processing with progress bar
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True
)

# Save embeddings to avoid recomputation
import numpy as np
np.save('embeddings.npy', embeddings.cpu().numpy())

# Load saved embeddings
embeddings = np.load('embeddings.npy')
```

### Memory Management

If you encounter out-of-memory errors:

1. Use the 4B model (set in `config.py`)
2. Enable half-precision: `model.half()`
3. Use CPU mode: `device='cpu'` (slower but uses system RAM)
4. Reduce batch size in `model.encode()`
5. Process documents in smaller chunks

## Technical Details

### Embedding Generation
- Embeddings are 4096-dimensional vectors
- Normalized to unit length (L2 norm = 1)
- Suitable for cosine similarity computation

### Similarity Calculation
- Uses cosine similarity: ranges from -1 (opposite) to 1 (identical)
- Symmetric matrix: `similarity[i][j] == similarity[j][i]`
- Diagonal values are 1.0 (perfect self-similarity)

### Dimensionality Reduction
- **t-SNE**: Best for small datasets, preserves local structure, slower
- **UMAP**: Best for large datasets, preserves global structure, faster
- Both support 2D and 3D visualization
- Results are stochastic (set `random_state` for reproducibility)

### Clustering
- K-means clustering in high-dimensional space (4096D)
- Requires pre-specifying number of clusters
- Results visualized after dimensionality reduction

## Requirements

Core dependencies:
- `sentence-transformers>=3.0.0`
- `torch>=2.0.0`
- `transformers>=4.34.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `plotly>=5.18.0`

Optional:
- `umap-learn>=0.5.5` (recommended for better visualizations)
- `modelscope` (for faster downloads in China)

See `requirements.txt` for complete list.

## Project Structure

```
test.embedding/
├── config.py                              # Model configuration
├── download_helper.py                     # Download utilities
├── 01_basic_usage.py                      # Basic embedding example
├── 02_similarity_calculation.py           # Similarity calculation example
├── 03_semantic_search.py                  # Semantic search example
├── 04_text_clustering_visualization.py    # Clustering visualization example
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── CLAUDE.md                              # Claude Code guidance
└── .gitignore                             # Git ignore rules
```

## Troubleshooting

### Slow Downloads
- Install ModelScope: `pip install modelscope`
- Or set HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`

### Out of Memory
- Switch to 4B model in `config.py`
- Enable half-precision: `model.half()`
- Use CPU mode (slower)
- Reduce batch size

### Import Errors
- Ensure `config.py` is in the same directory as example scripts
- Install all dependencies: `pip install -r requirements.txt`

### Interactive Search Won't Exit
- Type 'quit', 'exit', 'q', or press Ctrl+C

## License

MIT License

The Qwen3-Embedding models are licensed under Apache 2.0.

## Citation

If you use Qwen3-Embedding in your research, please cite:

```bibtex
@article{qwen3-embedding,
  title={Qwen3-Embedding: A New Benchmark for Multilingual Text Embeddings},
  author={Qwen Team},
  year={2024}
}
```

## References

- **Qwen3 Embedding Official Blog**: https://qwenlm.github.io/blog/qwen3-embedding/
- **Hugging Face Model Page**: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- **GitHub Repository**: https://github.com/QwenLM/Qwen3-Embedding
- **Paper**: https://arxiv.org/abs/2506.05176
- **MTEB Leaderboard**: https://huggingface.co/spaces/mteb/leaderboard

## Acknowledgments

- Qwen Team for the excellent Qwen3-Embedding models
- Sentence-Transformers library for the easy-to-use interface
- All open-source contributors

---

**Note**: On first run, the model will be automatically downloaded from Hugging Face (or ModelScope if configured). This may take 12-25 minutes depending on your model choice and internet speed.
