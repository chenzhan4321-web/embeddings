# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demonstration project for the Qwen3-Embedding-8B model, showcasing applications of text embedding, semantic similarity calculation, semantic search, and text clustering visualization. The project uses Qwen3-Embedding-8B (ranked #1 on the MTEB multilingual leaderboard) for text vectorization.

## Common Commands

### Model Configuration (⭐ Important)
```bash
# Edit config.py to select model size
# MODEL_SIZE = "4B"  # Default, recommended for most users
# MODEL_SIZE = "8B"  # For ultimate performance

# View current model configuration
python config.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install UMAP for dimensionality reduction (recommended)
pip install umap-learn

# Optional: Use mirror to accelerate model downloads
export HF_ENDPOINT=https://hf-mirror.com
```

### Run Example Scripts
```bash
# Run the four example scripts in order
python 01_basic_usage.py                    # Basic usage: model loading and embedding generation
python 02_similarity_calculation.py         # Similarity calculation: text semantic similarity matrix
python 03_semantic_search.py                # Semantic search: build a search engine (includes interactive search)
python 04_text_clustering_visualization.py  # Text clustering: generate 2D/3D interactive visualizations
```

Note: 03_semantic_search.py includes interactive search functionality. Enter 'quit', 'exit', 'q' to exit.

## Core Architecture

### Tech Stack
- **Model**: Qwen3-Embedding (supports 4B and 8B, 4096-dimensional vectors)
  - **4B Model** (default): ~8GB, 98% performance, recommended for most users
  - **8B Model**: ~16GB, 100% performance, #1 on MTEB leaderboard
- **Configuration System**: config.py (centralized model selection)
- **Core Libraries**: sentence-transformers, torch, transformers
- **Data Processing**: numpy, scikit-learn
- **Visualization**: plotly (interactive HTML visualizations)
- **Dimensionality Reduction**: t-SNE, UMAP (optional)
- **Clustering**: K-means

### Project Files

- **config.py**: Model configuration file, controls whether to use 4B or 8B model
  - `MODEL_SIZE`: Set model size ("4B" or "8B")
  - `get_model_name()`: Get the current configured model name
  - `get_model_info()`: Get detailed model information
  - `print_model_info()`: Print configuration information
  - `print_model_comparison()`: Print model comparison

### Code Structure

#### 1. Basic Usage (01_basic_usage.py)
- Demonstrates the most basic model loading and embedding generation
- Shows embedding statistical properties (dimensions, norms, distributions, etc.)
- Suitable for beginners to understand embedding basics

#### 2. Similarity Calculation (02_similarity_calculation.py)
**Core Functions**:
- `print_similarity_matrix()`: Format and print similarity matrix
- `find_most_similar()`: Find candidate texts most similar to the query text
- Supports same-language and cross-language (Chinese-English) similarity calculation
- Uses cosine similarity (`util.cos_sim()`) for semantic comparison

**Key Technical Points**:
- Use `convert_to_tensor=True` parameter to accelerate computation
- Similarity values range from [-1, 1], closer to 1 means more similar

#### 3. Semantic Search (03_semantic_search.py)
**Core Class**: `SemanticSearchEngine`
- `index_documents()`: Batch index document library, generate and store all document embeddings
- `search()`: Execute semantic search, return top-k most relevant documents
- Supports interactive search loop

**Implementation Principles**:
- Pre-compute all document embeddings (indexing phase)
- Only compute query text embedding during search
- Use `topk()` to quickly get the most similar results
- Avoid redundant computation, improve search efficiency

**Dataset Structure**:
- Four themes: technology, programming, lifestyle, business
- 5-10 documents per theme
- Demonstrates semantic search's cross-theme matching capabilities

#### 4. Text Clustering Visualization (04_text_clustering_visualization.py)
**Core Class**: `TextClusteringVisualizer`

**Main Methods**:
- `prepare_data()`: Generate text embeddings
- `reduce_dimensions_tsne()`: t-SNE dimensionality reduction (2D/3D)
- `reduce_dimensions_umap()`: UMAP dimensionality reduction (2D/3D, optional)
- `cluster_kmeans()`: K-means automatic clustering
- `visualize_2d()`: Create 2D interactive visualization
- `visualize_3d()`: Create 3D interactive visualization

**Visualization Outputs**:
- `clustering_2d_by_label.html`: 2D visualization (colored by true labels)
- `clustering_2d_by_cluster.html`: 2D visualization (colored by cluster results)
- `clustering_3d_by_label.html`: 3D visualization (colored by true labels)
- `clustering_3d_by_cluster.html`: 3D visualization (colored by cluster results)
- If UMAP is installed: `clustering_umap_2d.html` and `clustering_umap_3d.html`

**Technical Details**:
- Uses Plotly to create interactive HTML, supports hover, zoom, rotation
- t-SNE parameters: perplexity=15 (suitable for small datasets), max_iter=1000
- UMAP parameters: n_neighbors=15, min_dist=0.1
- Auto-generates hover text, includes category and text preview (first 100 characters)

**Sample Dataset**:
- `create_sample_dataset()` creates text data for 4 themes (technology, food, sports, travel)
- 10 texts per theme, 40 samples total
- Used to demonstrate clustering and visualization effects

### Model Features

**Qwen3-Embedding Series**:

| Feature | 4B Model | 8B Model |
|---------|----------|----------|
| Output Dimension | 4096 | 4096 |
| Supported Languages | 100+ | 100+ |
| MTEB Multilingual | 69.45 | 70.58 (#1) |
| C-MTEB Chinese | 72.27 | 73.84 |
| Model Size | ~8GB | ~16GB |
| Memory Required | 8-10GB | 16GB+ |
| Inference Speed | 1.4x | 1.0x |
| Default Config | ✅ Yes | No |

**Performance Optimization Options**:
```python
import config

# Use the model from configuration file
model_name = config.get_model_name()
model = SentenceTransformer(model_name, device='cuda', trust_remote_code=True)

# Half-precision inference (saves memory)
model.half()

# Batch processing
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

### Application Scenarios

1. **Semantic Search**: Document library retrieval, knowledge base Q&A
2. **Text Classification**: Classification based on semantic similarity
3. **Recommendation Systems**: Content recommendations, similar article suggestions
4. **Clustering Analysis**: Topic discovery, content grouping
5. **Deduplication**: Similar content identification
6. **Cross-language Retrieval**: Multilingual document matching

## Important Technical Notes

### Embedding Generation
- Use `model.encode()` to generate embeddings
- Parameter `convert_to_tensor=True` generates PyTorch tensors (for GPU acceleration)
- Parameter `convert_to_numpy=True` generates numpy arrays (for scikit-learn)
- First run will automatically download the model from Hugging Face (~16GB)

### Similarity Calculation
- Uses cosine similarity: `util.cos_sim(emb1, emb2)`
- Similarity matrix is symmetric (similarity[i][j] == similarity[j][i])
- Diagonal values are 1.0 (text is perfectly similar to itself)

### Dimensionality Reduction Visualization
- **t-SNE**: Suitable for small datasets, preserves local structure, slower computation
- **UMAP**: Suitable for large datasets, preserves global and local structure, faster computation
- perplexity parameter affects t-SNE clustering effect (recommended between 5-50)
- Dimensionality reduction is a stochastic process, set random_state to ensure reproducibility

### Clustering Algorithms
- K-means requires pre-specifying the number of clusters `n_clusters`
- Can evaluate clustering quality by comparing true labels and clustering results
- Clustering is performed in high-dimensional embedding space (4096 dimensions), then visualized with dimensionality reduction

## Development Notes

1. **Model Selection**:
   - Default uses **4B model** (98% performance, 8GB memory)
   - For best performance, change to `MODEL_SIZE = "8B"` in `config.py`
   - After switching models, all scripts will automatically use the new model

2. **Memory Management**: If memory is insufficient:
   - Use 4B model (set in config.py)
   - Use `model.half()` to enable half-precision
   - Use CPU mode (slower)
   - Reduce batch size

3. **Configuration Import**: All example scripts need to import `import config`, ensure config.py is in the same directory

4. **Data Saving**: Can use `numpy.save()` and `numpy.load()` to save/load embeddings, avoiding redundant computation

5. **Interactive Scripts**: 03_semantic_search.py contains an infinite loop for interactive search, use Ctrl+C or enter exit commands to terminate

6. **HTML Visualization**: 04_text_clustering_visualization.py generates large HTML files (~5MB), open with browser for interaction

7. **Encoding**: All scripts use UTF-8 encoding (`# -*- coding: utf-8 -*-`), ensuring correct handling of multilingual text
