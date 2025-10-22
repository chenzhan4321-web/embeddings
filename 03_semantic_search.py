#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding Semantic Search Example

This script demonstrates how to build a simple semantic search engine using Qwen3-Embedding
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Tuple
import config  # Import configuration file


class SemanticSearchEngine:
    """Simple semantic search engine"""

    def __init__(self, model_name: str = None):
        """Initialize search engine"""
        # If no model specified, use model from config file
        if model_name is None:
            # Use config file's load_model() function (automatically uses ModelScope mirror)
            self.model = config.load_model(device='cpu')
        else:
            print(f"Loading specified model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print("✓ Model loaded successfully!")

        self.documents = []
        self.document_embeddings = None

    def index_documents(self, documents: List[str]):
        """Index document collection"""
        print(f"\nIndexing {len(documents)} documents...")
        self.documents = documents
        self.document_embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print("✓ Document indexing completed!")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for most relevant documents"""
        if self.document_embeddings is None:
            raise ValueError("Please index documents first using index_documents()")

        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate similarity
        similarities = util.cos_sim(query_embedding, self.document_embeddings)[0]

        # Get top_k results
        top_results = similarities.topk(k=min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((self.documents[idx], score.item()))

        return results


def print_search_results(query: str, results: List[Tuple[str, float]]):
    """Print search results"""
    print(f"\nQuery: \"{query}\"")
    print("-" * 80)

    if not results:
        print("No relevant results found")
        return

    for i, (doc, score) in enumerate(results, 1):
        # Visualize similarity
        bar_length = int(score * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)

        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   [{bar}]")
        print(f"   Document: {doc}")


def main():
    print("=" * 80)
    print("Qwen3-Embedding Semantic Search Example")
    print("=" * 80)

    # 1. Create search engine
    print("\n[1/4] Initializing search engine...")
    config.print_model_info()
    print()
    search_engine = SemanticSearchEngine()

    # 2. Prepare document collection
    print("\n[2/4] Preparing document collection...")

    documents = [
        # Technology
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需显式编程",
        "深度学习使用多层神经网络来学习数据的层次化表示",
        "自然语言处理研究计算机如何理解、解释和生成人类语言",
        "计算机视觉使机器能够从图像和视频中获取高层次的理解",
        "强化学习是一种机器学习方法，智能体通过与环境交互来学习最优策略",

        # Programming
        "Python是一种高级编程语言，广泛用于数据科学和机器学习",
        "JavaScript是Web开发的核心语言，用于创建交互式网页",
        "Git是一个分布式版本控制系统，用于跟踪代码变更",
        "Docker是一个容器化平台，用于构建、部署和运行应用程序",
        "Kubernetes是一个容器编排系统，用于自动化部署和管理容器化应用",

        # Lifestyle
        "今天天气晴朗，适合外出散步和运动",
        "这家餐厅的菜品味道很好，环境也很舒适",
        "我喜欢在周末看电影、读书和听音乐",
        "旅行可以让人放松心情，体验不同的文化",
        "运动对身体健康非常重要，建议每天坚持锻炼",

        # Business
        "电子商务正在改变传统的零售业态",
        "数字营销通过互联网平台推广产品和服务",
        "供应链管理优化商品从生产到消费者的整个流程",
        "创业需要创新思维、执行力和风险管理能力",
        "投资理财需要了解市场规律和风险控制",
    ]

    print(f"Document collection contains {len(documents)} documents")

    # 3. Index documents
    print("\n[3/4] Indexing document collection...")
    search_engine.index_documents(documents)

    # 4. Execute searches
    print("\n[4/4] Executing semantic searches")
    print("=" * 80)

    # Search Example 1: Technology query
    print("\n【Example 1: Technology Query】")
    query1 = "如何让机器自动学习"
    results1 = search_engine.search(query1, top_k=3)
    print_search_results(query1, results1)

    # Search Example 2: Programming query
    print("\n\n【Example 2: Programming Query】")
    query2 = "Web前端开发"
    results2 = search_engine.search(query2, top_k=3)
    print_search_results(query2, results2)

    # Search Example 3: Lifestyle query
    print("\n\n【Example 3: Lifestyle Query】")
    query3 = "休息日可以做什么"
    results3 = search_engine.search(query3, top_k=3)
    print_search_results(query3, results3)

    # Search Example 4: Business query
    print("\n\n【Example 4: Business Query】")
    query4 = "如何在网上卖东西"
    results4 = search_engine.search(query4, top_k=3)
    print_search_results(query4, results4)

    # Search Example 5: Fuzzy query
    print("\n\n【Example 5: Fuzzy Query】")
    query5 = "神经网络和AI"
    results5 = search_engine.search(query5, top_k=5)
    print_search_results(query5, results5)

    # Interactive search
    print("\n" + "=" * 80)
    print("Interactive Search (enter 'quit' or 'exit' to quit)")
    print("=" * 80)

    while True:
        try:
            user_query = input("\nEnter your query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q', '退出']:
                print("Thank you for using! Goodbye!")
                break

            if not user_query:
                print("Query cannot be empty, please try again")
                continue

            results = search_engine.search(user_query, top_k=5)
            print_search_results(user_query, results)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted, thank you for using!")
            break
        except Exception as e:
            print(f"Error occurred: {e}")

    print("\n" + "=" * 80)
    print("✓ Semantic search example completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
