#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding Text Similarity Calculation Example

This script demonstrates how to use Qwen3-Embedding to calculate semantic similarity between texts
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import config  # Import configuration file


def print_similarity_matrix(texts, similarities):
    """Print similarity matrix"""
    print("\nSimilarity Matrix (values closer to 1 indicate higher similarity):")
    print("-" * 80)

    # Print header
    print(f"{'':30}", end="")
    for i in range(len(texts)):
        print(f"Text{i+1:2d}  ", end="")
    print()

    # Print each row
    for i, text in enumerate(texts):
        # Truncate long texts
        text_short = text[:25] + "..." if len(text) > 25 else text
        print(f"Text{i+1:2d}: {text_short:25}", end="")

        for j in range(len(texts)):
            sim_value = similarities[i][j].item()
            print(f"{sim_value:6.3f}  ", end="")
        print()


def find_most_similar(query_text, candidate_texts, model):
    """Find the most similar candidate texts to the query text"""
    # Generate embeddings
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

    # Calculate similarity
    similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]

    # Sort
    results = []
    for i, sim in enumerate(similarities):
        results.append((candidate_texts[i], sim.item()))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    print("=" * 80)
    print("Qwen3-Embedding Text Similarity Calculation Example")
    print("=" * 80)

    # 1. Load model
    print("\n[1/4] Loading model...")
    config.print_model_info()
    print()

    # Use config file's load_model() function (automatically uses ModelScope mirror)
    model = config.load_model(device='cpu')

    # 2. Example 1: Calculate similarity matrix for a group of texts
    print("\n" + "=" * 80)
    print("Example 1: Calculate Text Similarity Matrix")
    print("=" * 80)

    texts = [
        "机器学习是人工智能的重要分支",
        "深度学习属于机器学习的一种方法",
        "自然语言处理用于理解人类语言",
        "今天天气很好，阳光明媚",
        "我喜欢在周末看电影"
    ]

    print("\n[2/4] Test texts:")
    for i, text in enumerate(texts, 1):
        print(f"  Text{i}: {text}")

    print("\nGenerating embeddings and calculating similarity...")
    embeddings = model.encode(texts, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings, embeddings)

    print_similarity_matrix(texts, similarities)

    # 3. Example 2: Find the most similar texts
    print("\n" + "=" * 80)
    print("Example 2: Find Most Similar Texts to Query")
    print("=" * 80)

    query = "人工智能和机器学习"
    candidates = [
        "深度学习是机器学习的子领域",
        "今天的天气非常晴朗",
        "神经网络是深度学习的基础",
        "我喜欢吃中国菜",
        "自然语言处理是AI的应用",
        "周末我想去爬山",
    ]

    print(f"\n[3/4] Query text: {query}")
    print("\nCandidate texts:")
    for i, text in enumerate(candidates, 1):
        print(f"  {i}. {text}")

    print("\nFinding most similar texts...")
    results = find_most_similar(query, candidates, model)

    print("\nSimilarity Ranking (highest to lowest):")
    print("-" * 80)
    for i, (text, similarity) in enumerate(results, 1):
        # Visualize similarity with progress bar
        bar_length = int(similarity * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{i}. [{bar}] {similarity:.4f} - {text}")

    # 4. Example 3: Cross-language similarity (if model supports)
    print("\n" + "=" * 80)
    print("Example 3: Multilingual Text Similarity")
    print("=" * 80)

    multilingual_texts = [
        # Topic 1: Machine Learning (5 languages)
        "机器学习是人工智能的分支",  # Chinese
        "Machine learning is a branch of artificial intelligence",  # English
        "L'apprentissage automatique est une branche de l'intelligence artificielle",  # French
        "Maschinelles Lernen ist ein Zweig der künstlichen Intelligenz",  # German
        "Machina discendi est ramus intelligentiae artificialis",  # Latin

        # Topic 2: Weather (5 languages)
        "今天天气很好",  # Chinese
        "The weather is nice today",  # English
        "Il fait beau aujourd'hui",  # French
        "Das Wetter ist heute schön",  # German
        "Hodie tempestas bona est",  # Latin
    ]

    print("\n[4/4] Multilingual test texts:")
    print("\n  Topic 1 - Machine Learning (5 languages):")
    for i in range(5):
        lang_names = ["Chinese", "English", "French", "German", "Latin"]
        print(f"    {lang_names[i]:10}: {multilingual_texts[i]}")

    print("\n  Topic 2 - Weather (5 languages):")
    for i in range(5, 10):
        lang_names = ["Chinese", "English", "French", "German", "Latin"]
        print(f"    {lang_names[i-5]:10}: {multilingual_texts[i]}")

    print("\nCalculating cross-language similarity...")
    ml_embeddings = model.encode(multilingual_texts, convert_to_tensor=True)
    ml_similarities = util.cos_sim(ml_embeddings, ml_embeddings)

    print("\nSimilarity Matrix (showing first 5 texts - Machine Learning topic):")
    print("-" * 80)
    # Print header
    print(f"{'':15}", end="")
    for i in range(5):
        print(f"Lang{i+1:2d}  ", end="")
    print()

    # Print each row
    lang_short = ["CN", "EN", "FR", "DE", "LA"]
    for i in range(5):
        print(f"Lang{i+1:2d} ({lang_short[i]:2}):   ", end="")
        for j in range(5):
            sim_value = ml_similarities[i][j].item()
            print(f"{sim_value:6.3f}  ", end="")
        print()

    print("\nCross-language similarity analysis:")
    print("\n  Same topic across different languages (should be high):")
    print(f"    Chinese (ML) vs English (ML):  {ml_similarities[0][1]:.4f}")
    print(f"    Chinese (ML) vs French (ML):   {ml_similarities[0][2]:.4f}")
    print(f"    Chinese (ML) vs German (ML):   {ml_similarities[0][3]:.4f}")
    print(f"    Chinese (ML) vs Latin (ML):    {ml_similarities[0][4]:.4f}")
    print(f"    English (ML) vs German (ML):   {ml_similarities[1][3]:.4f}")
    print(f"    French (ML) vs Latin (ML):     {ml_similarities[2][4]:.4f}")

    print("\n  Different topics in same language (should be lower):")
    print(f"    Chinese (ML) vs Chinese (Weather):   {ml_similarities[0][5]:.4f}")
    print(f"    English (ML) vs English (Weather):   {ml_similarities[1][6]:.4f}")
    print(f"    French (ML) vs French (Weather):     {ml_similarities[2][7]:.4f}")
    print(f"    Latin (ML) vs Latin (Weather):       {ml_similarities[4][9]:.4f}")

    print("\n  Cross-topic cross-language (should be lowest):")
    print(f"    Chinese (ML) vs English (Weather):   {ml_similarities[0][6]:.4f}")
    print(f"    English (ML) vs Chinese (Weather):   {ml_similarities[1][5]:.4f}")
    print(f"    French (ML) vs German (Weather):     {ml_similarities[2][8]:.4f}")

    print("\n" + "=" * 80)
    print("✓ Text similarity calculation example completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
