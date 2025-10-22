#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding Basic Usage Example

This script demonstrates how to load the Qwen3-Embedding model and generate embeddings for text at different granularities
Including: word, phrase, and sentence embeddings, along with comparative analysis
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import config  # Import configuration file


def print_embedding_stats(text, embedding, prefix=""):
    """Print embedding statistics"""
    print(f"{prefix}Text: {text}")
    print(f"{prefix}  - Vector Dimension: {embedding.shape[0]}")
    print(f"{prefix}  - Min Value: {np.min(embedding):.4f}")
    print(f"{prefix}  - Max Value: {np.max(embedding):.4f}")
    print(f"{prefix}  - Mean Value: {np.mean(embedding):.4f}")
    print(f"{prefix}  - Std Deviation: {np.std(embedding):.4f}")
    print(f"{prefix}  - Vector Norm: {np.linalg.norm(embedding):.4f}")


def main():
    print("=" * 80)
    print("Qwen3-Embedding Basic Usage Example")
    print("Demo: Generating and comparing embeddings for words, phrases, and sentences")
    print("=" * 80)

    # 1. Load model (first run will auto-download, takes a few minutes)
    print("\n[1/5] Loading Qwen3-Embedding model...")

    # Display current model configuration
    config.print_model_info()

    model_info = config.get_model_info()
    print(f"\nNote: First run will auto-download the model (approx. {model_info['model_size']})")
    print(f"To switch models, edit the MODEL_SIZE parameter in config.py\n")

    # Use config file's load_model() function (automatically uses ModelScope mirror)
    model = config.load_model(device='cpu')  # Use CPU, change to 'cuda' if you have GPU

    # 2. Prepare test texts at different granularities
    print("\n[2/5] Preparing test texts at different granularities...")

    # Words
    words = [
        "机器学习",
        "深度学习",
        "人工智能",
        "苹果",
        "天气"
    ]

    # Phrases
    phrases = [
        "机器学习算法",
        "深度神经网络",
        "人工智能应用",
        "新鲜的苹果",
        "今天的天气"
    ]

    # Sentences
    sentences = [
        "机器学习是人工智能的一个重要分支",
        "深度学习使用多层神经网络进行学习",
        "自然语言处理研究计算机与人类语言的交互",
        "今天天气很好，适合出去散步",
        "我喜欢在周末看电影和读书"
    ]

    print("\nWord Examples:")
    for i, text in enumerate(words, 1):
        print(f"  {i}. {text}")

    print("\nPhrase Examples:")
    for i, text in enumerate(phrases, 1):
        print(f"  {i}. {text}")

    print("\nSentence Examples:")
    for i, text in enumerate(sentences, 1):
        print(f"  {i}. {text}")

    # 3. Generate embeddings at different granularities
    print("\n[3/5] Generating embeddings at different granularities...")

    print("  Generating word embeddings...")
    word_embeddings = model.encode(words, show_progress_bar=False)

    print("  Generating phrase embeddings...")
    phrase_embeddings = model.encode(phrases, show_progress_bar=False)

    print("  Generating sentence embeddings...")
    sentence_embeddings = model.encode(sentences, show_progress_bar=False)

    print("✓ All embeddings generated successfully!")

    # 4. Display basic information
    print("\n[4/5] Embedding Basic Information")
    print("=" * 80)
    print(f"Vector dimension for all texts: {word_embeddings.shape[1]}")
    print(f"Word embeddings shape: {word_embeddings.shape} (count: {word_embeddings.shape[0]})")
    print(f"Phrase embeddings shape: {phrase_embeddings.shape} (count: {phrase_embeddings.shape[0]})")
    print(f"Sentence embeddings shape: {sentence_embeddings.shape} (count: {sentence_embeddings.shape[0]})")

    # 5. Comparative analysis: words vs phrases vs sentences
    print("\n[5/5] Comparative Analysis: Words vs Phrases vs Sentences")
    print("=" * 80)

    # 5.1 Statistical characteristics comparison
    print("\n▶ Example 1: Statistical characteristics comparison for '机器学习' related texts")
    print("-" * 80)
    print_embedding_stats(words[0], word_embeddings[0], "Word - ")
    print()
    print_embedding_stats(phrases[0], phrase_embeddings[0], "Phrase - ")
    print()
    print_embedding_stats(sentences[0], sentence_embeddings[0], "Sentence - ")

    # 5.2 Semantic similarity comparison
    print("\n▶ Example 2: Semantic similarity between texts at different granularities")
    print("-" * 80)

    # Calculate similarity between three "机器学习" related texts
    test_texts = [words[0], phrases[0], sentences[0]]
    test_embeddings = model.encode(test_texts, convert_to_tensor=True)
    similarities = util.cos_sim(test_embeddings, test_embeddings)

    print("Similarity Matrix:")
    print(f"{'':30} Word      Phrase    Sentence")
    print(f"Word ({words[0]:15}) {similarities[0][0]:.4f}   {similarities[0][1]:.4f}   {similarities[0][2]:.4f}")
    print(f"Phrase ({phrases[0]:15}) {similarities[1][0]:.4f}   {similarities[1][1]:.4f}   {similarities[1][2]:.4f}")
    print(f"Sentence ({sentences[0][:15]}...) {similarities[2][0]:.4f}   {similarities[2][1]:.4f}   {similarities[2][2]:.4f}")

    print("\nAnalysis:")
    print(f"  - Word vs Phrase: {similarities[0][1]:.4f} (phrase contains word, highly semantically related)")
    print(f"  - Word vs Sentence: {similarities[0][2]:.4f} (sentence discusses word's topic)")
    print(f"  - Phrase vs Sentence: {similarities[1][2]:.4f} (both related to machine learning)")

    # 5.3 Cross-topic comparison
    print("\n▶ Example 3: Similarity comparison across different topics")
    print("-" * 80)

    # Compare "机器学习" and "天气" two different topics
    cross_texts = [
        words[0],      # 机器学习
        phrases[0],    # 机器学习算法
        words[4],      # 天气
        phrases[4]     # 今天的天气
    ]
    cross_embeddings = model.encode(cross_texts, convert_to_tensor=True)
    cross_sim = util.cos_sim(cross_embeddings, cross_embeddings)

    print("Similarity across topics:")
    print(f"  Same topic (machine learning):")
    print(f"    '{cross_texts[0]}' vs '{cross_texts[1]}': {cross_sim[0][1]:.4f}")
    print(f"  Same topic (weather):")
    print(f"    '{cross_texts[2]}' vs '{cross_texts[3]}': {cross_sim[2][3]:.4f}")
    print(f"  Cross-topic:")
    print(f"    '{cross_texts[0]}' vs '{cross_texts[2]}': {cross_sim[0][2]:.4f}")
    print(f"    '{cross_texts[1]}' vs '{cross_texts[3]}': {cross_sim[1][3]:.4f}")

    # 5.4 Display first 10 dimensions of first word's embedding
    print("\n▶ Example 4: Embedding Vector Preview")
    print("-" * 80)
    print(f"First 10 dimensions of word '{words[0]}' embedding:")
    print(f"  {word_embeddings[0][:10]}")
    print(f"\nFirst 10 dimensions of sentence '{sentences[0][:20]}...' embedding:")
    print(f"  {sentence_embeddings[0][:10]}")

    # 5.5 Multilingual word comparison
    print("\n▶ Example 5: Multilingual Word Embeddings")
    print("-" * 80)

    multilingual_words = [
        "机器学习",  # Chinese
        "Machine Learning",  # English
        "Apprentissage Automatique",  # French
        "Maschinelles Lernen",  # German
        "Machina Discendi",  # Latin
    ]

    print("Same concept 'Machine Learning' in different languages:")
    for i, word in enumerate(multilingual_words):
        lang_names = ["Chinese", "English", "French", "German", "Latin"]
        print(f"  {lang_names[i]:10}: {word}")

    print("\nGenerating embeddings for multilingual words...")
    ml_word_embeddings = model.encode(multilingual_words, convert_to_tensor=True)
    ml_word_similarities = util.cos_sim(ml_word_embeddings, ml_word_embeddings)

    print("\nSimilarity Matrix (5 languages × 5 languages):")
    print("-" * 80)
    # Print header
    lang_short = ["CN", "EN", "FR", "DE", "LA"]
    print(f"{'':15}", end="")
    for lang in lang_short:
        print(f"{lang:7}", end="")
    print()

    # Print each row
    for i, lang in enumerate(lang_short):
        print(f"{lang:5} {multilingual_words[i][:8]:8}", end="")
        for j in range(len(multilingual_words)):
            print(f"{ml_word_similarities[i][j].item():7.3f}", end="")
        print()

    print("\nKey observations:")
    print(f"  Chinese vs English:  {ml_word_similarities[0][1]:.4f} (cross-language, same concept)")
    print(f"  English vs German:   {ml_word_similarities[1][3]:.4f} (both Germanic languages)")
    print(f"  English vs Latin:    {ml_word_similarities[1][4]:.4f} (modern vs ancient)")
    print(f"  French vs German:    {ml_word_similarities[2][3]:.4f} (European languages)")
    print(f"  French vs Latin:     {ml_word_similarities[2][4]:.4f} (French derived from Latin)")

    # Summary
    print("\n" + "=" * 80)
    print("✓ Basic usage example completed!")
    print("=" * 80)
    print("\nKey Findings:")
    print("  1. Embeddings for words, phrases, and sentences all have 4096 dimensions")
    print("  2. Longer texts provide more context and richer semantic information")
    print("  3. Texts at different granularities on the same topic (word→phrase→sentence) have high semantic similarity")
    print("  4. The model can capture semantic information at all granularities from words to sentences")
    print("  5. The model supports 100+ languages including Chinese, English, French, German, Latin, etc.")
    print("  6. Same concepts across different languages show high semantic similarity")
    print("=" * 80)


if __name__ == "__main__":
    main()
