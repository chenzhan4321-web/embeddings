#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding Text Clustering and Visualization Example

This script demonstrates how to perform text clustering and visualization using Qwen3-Embedding:
- Use t-SNE and UMAP for dimensionality reduction
- Use K-means for automatic clustering
- Generate interactive 2D and 3D visualizations
- Save as HTML files
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import warnings
import os
import config  # Import configuration file

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed, will only use t-SNE for dimensionality reduction. Install with: pip install umap-learn")


class TextClusteringVisualizer:
    """Text clustering visualization tool"""

    def __init__(self, model_name: str = None):
        """Initialize visualization tool"""
        # If no model specified, use model from config file
        if model_name is None:
            # Use config file's load_model() function (automatically uses ModelScope mirror)
            self.model = config.load_model(device='cpu')
        else:
            print(f"Loading specified model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print("✓ Model loaded successfully!")

        self.texts = []
        self.labels = []
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None

    def prepare_data(self, texts: List[str], labels: Optional[List[str]] = None):
        """
        Prepare data

        Args:
            texts: List of texts
            labels: Optional list of labels (for displaying true categories)
        """
        print(f"\nPreparing {len(texts)} texts...")
        self.texts = texts
        self.labels = labels if labels else [f"Text{i+1}" for i in range(len(texts))]

        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✓ Embeddings generated successfully! Shape: {self.embeddings.shape}")

    def reduce_dimensions_tsne(self, n_components: int = 2, perplexity: int = 30,
                              random_state: int = 42) -> np.ndarray:
        """
        Reduce dimensions using t-SNE

        Args:
            n_components: Target dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter (between 5-50)
            random_state: Random seed

        Returns:
            Reduced coordinates
        """
        if self.embeddings is None:
            raise ValueError("Please prepare data first using prepare_data()")

        print(f"\nReducing dimensions to {n_components}D using t-SNE...")
        print(f"Parameters: perplexity={perplexity}, random_state={random_state}")

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            verbose=1
        )

        self.reduced_embeddings = tsne.fit_transform(self.embeddings)
        print(f"✓ t-SNE dimensionality reduction completed! Shape: {self.reduced_embeddings.shape}")

        return self.reduced_embeddings

    def reduce_dimensions_umap(self, n_components: int = 2, n_neighbors: int = 15,
                              min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
        """
        Reduce dimensions using UMAP (if available)

        Args:
            n_components: Target dimensions (2 or 3)
            n_neighbors: UMAP number of neighbors
            min_dist: UMAP minimum distance parameter
            random_state: Random seed

        Returns:
            Reduced coordinates
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed, use: pip install umap-learn")

        if self.embeddings is None:
            raise ValueError("Please prepare data first using prepare_data()")

        print(f"\nReducing dimensions to {n_components}D using UMAP...")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, random_state={random_state}")

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=True
        )

        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        print(f"✓ UMAP dimensionality reduction completed! Shape: {self.reduced_embeddings.shape}")

        return self.reduced_embeddings

    def cluster_kmeans(self, n_clusters: int, random_state: int = 42) -> np.ndarray:
        """
        Cluster using K-means

        Args:
            n_clusters: Number of clusters
            random_state: Random seed

        Returns:
            Cluster labels
        """
        if self.embeddings is None:
            raise ValueError("Please prepare data first using prepare_data()")

        print(f"\nClustering with K-means into {n_clusters} clusters...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        print(f"✓ K-means clustering completed!")

        # Count items in each cluster
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nCluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} texts")

        return self.cluster_labels

    def visualize_2d(self, title: str = "Text Clustering Visualization (2D)",
                     color_by: str = "cluster", save_path: Optional[str] = None) -> go.Figure:
        """
        Create 2D interactive visualization

        Args:
            title: Chart title
            color_by: Coloring basis ("cluster" or "label")
            save_path: Optional save path

        Returns:
            Plotly figure object
        """
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] < 2:
            raise ValueError("Please reduce dimensions to 2D first")

        print(f"\nCreating 2D visualization...")

        # Determine coloring basis
        if color_by == "cluster" and self.cluster_labels is not None:
            colors = [f"Cluster {label}" for label in self.cluster_labels]
            color_label = "Cluster"
        else:
            colors = self.labels
            color_label = "Category"

        # Create figure
        fig = go.Figure()

        # Get unique color categories
        unique_colors = list(set(colors))
        color_palette = px.colors.qualitative.Set3[:len(unique_colors)]

        # Create a trace for each category
        for i, category in enumerate(unique_colors):
            mask = [c == category for c in colors]
            indices = [j for j, m in enumerate(mask) if m]

            # Prepare hover texts
            hover_texts = [
                f"<b>{category}</b><br>" +
                f"True Category: {self.labels[j]}<br>" +
                f"Text: {self.texts[j][:100]}{'...' if len(self.texts[j]) > 100 else ''}"
                for j in indices
            ]

            fig.add_trace(go.Scatter(
                x=self.reduced_embeddings[indices, 0],
                y=self.reduced_embeddings[indices, 1],
                mode='markers',
                name=category,
                marker=dict(
                    size=10,
                    color=color_palette[i % len(color_palette)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Arial, sans-serif')
            ),
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white',
            legend=dict(
                title=color_label,
                font=dict(size=12)
            )
        )

        # Save (using safer method)
        if save_path:
            try:
                print(f"  Saving to: {save_path}")
                # Use to_html() method directly, which is the most stable approach
                html_str = fig.to_html(
                    include_plotlyjs='cdn',
                    config={'displayModeBar': True, 'responsive': True},
                    include_mathjax=False
                )
                # Write directly to file, avoiding write_html()
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_str)
                print(f"✓ 2D visualization saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Error saving HTML file: {e}")
                import traceback
                traceback.print_exc()

        return fig

    def visualize_3d(self, title: str = "Text Clustering Visualization (3D)",
                     color_by: str = "cluster", save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D interactive visualization

        Args:
            title: Chart title
            color_by: Coloring basis ("cluster" or "label")
            save_path: Optional save path

        Returns:
            Plotly figure object
        """
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] < 3:
            raise ValueError("Please reduce dimensions to 3D first")

        print(f"\nCreating 3D visualization...")

        # Determine coloring basis
        if color_by == "cluster" and self.cluster_labels is not None:
            colors = [f"Cluster {label}" for label in self.cluster_labels]
            color_label = "Cluster"
        else:
            colors = self.labels
            color_label = "Category"

        # Create figure
        fig = go.Figure()

        # Get unique color categories
        unique_colors = list(set(colors))
        color_palette = px.colors.qualitative.Set3[:len(unique_colors)]

        # Create a trace for each category
        for i, category in enumerate(unique_colors):
            mask = [c == category for c in colors]
            indices = [j for j, m in enumerate(mask) if m]

            # Prepare hover texts
            hover_texts = [
                f"<b>{category}</b><br>" +
                f"True Category: {self.labels[j]}<br>" +
                f"Text: {self.texts[j][:100]}{'...' if len(self.texts[j]) > 100 else ''}"
                for j in indices
            ]

            fig.add_trace(go.Scatter3d(
                x=self.reduced_embeddings[indices, 0],
                y=self.reduced_embeddings[indices, 1],
                z=self.reduced_embeddings[indices, 2],
                mode='markers',
                name=category,
                marker=dict(
                    size=6,
                    color=color_palette[i % len(color_palette)],
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Arial, sans-serif')
            ),
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white',
            legend=dict(
                title=color_label,
                font=dict(size=12)
            )
        )

        # Save (using safer method)
        if save_path:
            try:
                print(f"  Saving to: {save_path}")
                # Use to_html() method directly, which is the most stable approach
                html_str = fig.to_html(
                    include_plotlyjs='cdn',
                    config={'displayModeBar': True, 'responsive': True},
                    include_mathjax=False
                )
                # Write directly to file, avoiding write_html()
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_str)
                print(f"✓ 3D visualization saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Error saving HTML file: {e}")
                import traceback
                traceback.print_exc()

        return fig


def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """
    Create sample dataset with words, phrases, and sentences

    Returns:
        (List of texts, List of labels)
    """
    # Technology - Words, Phrases, and Sentences
    tech_texts = [
        # Words
        "人工智能", "机器学习", "深度学习", "神经网络", "大数据",
        # Phrases
        "人工智能应用", "机器学习算法", "深度神经网络", "自然语言处理", "计算机视觉",
        # Chinese sentences
        "人工智能正在改变我们的生活方式",
        "机器学习算法可以从数据中学习模式",
        "深度神经网络在图像识别方面表现出色",
        "自然语言处理让计算机理解人类语言",
        "量子计算将带来计算能力的革命性突破",
        # English sentences
        "Artificial intelligence is revolutionizing healthcare diagnostics",
        "Machine learning models can predict customer behavior patterns",
        "Neural networks require massive amounts of training data",
        "Quantum computing will solve previously intractable problems",
        "Blockchain ensures transparent and immutable transactions",
        # Latin sentences
        "Intelligentia artificialis mundum nostrum transformat",
        "Computatio quantica potentiam computandi magnopere auget",
    ]

    # Food - Words, Phrases, and Sentences
    food_texts = [
        # Words
        "红烧肉", "寿司", "披萨", "火锅", "咖啡",
        # Phrases
        "新鲜的寿司", "意大利披萨", "麻辣火锅", "法式甜点", "手工拉面",
        # Chinese sentences
        "这道红烧肉肥而不腻，入口即化",
        "新鲜的寿司配上芥末和酱油非常美味",
        "意大利披萨的芝士拉丝效果让人食欲大增",
        "香浓的咖啡搭配可颂是完美的早餐组合",
        "麻辣火锅的辣味让人欲罢不能",
        # English sentences
        "This homemade pasta is perfectly al dente with rich tomato sauce",
        "Fresh oysters with lemon juice are a delightful seafood experience",
        "The chocolate lava cake has a perfectly gooey center",
        "Authentic Thai curry balances spicy, sweet, and savory flavors",
        "Freshly baked croissants are flaky, buttery, and irresistible",
        # Latin sentences
        "Cibus Italicus in toto mundo amatur et laudatur",
        "Panis recens ex furno optimum saporem et odorem habet",
    ]

    # Sports - Words, Phrases, and Sentences
    sport_texts = [
        # Words
        "跑步", "游泳", "篮球", "足球", "瑜伽",
        # Phrases
        "晨跑锻炼", "游泳运动", "篮球比赛", "足球训练", "瑜伽练习",
        # Chinese sentences
        "每天晨跑让我保持良好的体能状态",
        "瑜伽练习帮助我放松身心、提高柔韧性",
        "篮球比赛需要团队协作和战术配合",
        "游泳是一项很好的全身性有氧运动",
        "足球运动员需要出色的体能和技术",
        # English sentences
        "Running marathons requires months of dedicated training",
        "Swimming provides excellent cardiovascular exercise with low impact",
        "Weight training builds muscle strength and improves metabolism",
        "Yoga improves flexibility, balance, and mental clarity",
        "Playing tennis enhances hand-eye coordination and agility",
        # Latin sentences
        "Mens sana in corpore sano per exercitationem",
        "Exercitatio quotidiana sanitatem corporis et animi servat",
    ]

    # Travel - Words, Phrases, and Sentences
    travel_texts = [
        # Words
        "巴黎", "长城", "海滩", "樱花", "极光",
        # Phrases
        "埃菲尔铁塔", "马尔代夫海滩", "京都樱花", "冰岛极光", "威尼斯水城",
        # Chinese sentences
        "巴黎的埃菲尔铁塔是浪漫之都的象征",
        "马尔代夫的碧海蓝天让人流连忘返",
        "长城是中华民族的伟大建筑奇迹",
        "京都的古寺和樱花展现了日本的传统美",
        "威尼斯的水城风光独一无二",
        # English sentences
        "The Grand Canyon offers breathtaking views of geological history",
        "Ancient Roman ruins in Italy tell stories of a glorious empire",
        "The Northern Lights in Norway are a spectacular natural phenomenon",
        "Tropical beaches in the Caribbean provide the perfect relaxation",
        "Hiking through the Amazon rainforest is an unforgettable adventure",
        # Latin sentences
        "Roma aeterna urbs plena historia et cultura est",
        "Peregrinatio mentem aperit et animum recreat semper",
    ]

    # 组合数据
    all_texts = tech_texts + food_texts + sport_texts + travel_texts
    all_labels = (
        ["Technology"] * len(tech_texts) +
        ["Food"] * len(food_texts) +
        ["Sports"] * len(sport_texts) +
        ["Travel"] * len(travel_texts)
    )

    return all_texts, all_labels


def main():
    # Set environment variables to avoid multiprocessing issues on macOS
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Set matplotlib to non-interactive backend (if plotly uses it internally)
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

    print("=" * 80)
    print("Qwen3-Embedding Text Clustering Visualization Example")
    print("=" * 80)

    # 1. Create visualization tool
    print("\n[1/6] Initializing visualization tool...")
    config.print_model_info()
    print()
    visualizer = TextClusteringVisualizer()

    # 2. Prepare data
    print("\n[2/6] Preparing sample dataset...")
    texts, labels = create_sample_dataset()
    print(f"Dataset contains {len(texts)} texts (words, phrases, and sentences)")
    print(f"Divided into {len(set(labels))} categories")
    print(f"Categories: {', '.join(set(labels))}")

    visualizer.prepare_data(texts, labels)

    # 3. t-SNE dimensionality reduction (2D)
    print("\n[3/6] Reducing dimensions to 2D using t-SNE...")
    visualizer.reduce_dimensions_tsne(n_components=2, perplexity=15)

    # 4. Create 2D visualization
    print("\n[4/6] Creating 2D visualization...")

    # Color by true labels
    fig_2d_label = visualizer.visualize_2d(
        title="Text Clustering Visualization - Colored by Category (t-SNE 2D)",
        color_by="label",
        save_path="clustering_2d_by_label.html"
    )

    # 5. t-SNE dimensionality reduction (3D)
    print("\n[5/6] Reducing dimensions to 3D using t-SNE...")
    visualizer.reduce_dimensions_tsne(n_components=3, perplexity=15)

    # 6. Create 3D visualization
    print("\n[6/6] Creating 3D visualization...")

    # Color by true labels
    fig_3d_label = visualizer.visualize_3d(
        title="Text Clustering Visualization - Colored by Category (t-SNE 3D)",
        color_by="label",
        save_path="clustering_3d_by_label.html"
    )

    # UMAP dimensionality reduction (if available)
    if UMAP_AVAILABLE:
        print("\n[Extra] Using UMAP for dimensionality reduction...")

        # UMAP 2D
        visualizer.reduce_dimensions_umap(n_components=2, n_neighbors=15)
        fig_umap_2d = visualizer.visualize_2d(
            title="Text Clustering Visualization - Colored by Category (UMAP 2D)",
            color_by="label",
            save_path="clustering_umap_2d.html"
        )

        # UMAP 3D
        visualizer.reduce_dimensions_umap(n_components=3, n_neighbors=15)
        fig_umap_3d = visualizer.visualize_3d(
            title="Text Clustering Visualization - Colored by Category (UMAP 3D)",
            color_by="label",
            save_path="clustering_umap_3d.html"
        )

    # Summary
    print("\n" + "=" * 80)
    print("✓ Text clustering visualization completed!")
    print("=" * 80)
    print("\nGenerated visualization files:")
    print("  1. clustering_2d_by_label.html      - 2D visualization (t-SNE)")
    print("  2. clustering_3d_by_label.html      - 3D visualization (t-SNE)")

    if UMAP_AVAILABLE:
        print("  3. clustering_umap_2d.html          - UMAP 2D visualization")
        print("  4. clustering_umap_3d.html          - UMAP 3D visualization")

    print("\nOpen these HTML files in a browser to view interactive visualizations!")
    print("\nTips:")
    print("  - Hover your mouse to view text content")
    print("  - Click legend items to hide/show specific categories")
    print("  - 3D charts can be rotated and zoomed")
    print("=" * 80)


if __name__ == "__main__":
    main()
