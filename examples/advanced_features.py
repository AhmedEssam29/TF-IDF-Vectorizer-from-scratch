#!/usr/bin/env python3
"""
Advanced Features Demo for TF-IDF Implementation

Demonstrates advanced configuration and analysis features.

Author: Ahmed Essam Abd Elgwad
Date: August 2025
"""

from tfidf import TFIDFVectorizer, TFIDFAnalyzer
import numpy as np

def main():
    """Advanced features demonstration."""
    print("TF-IDF Advanced Features Demo")
    print("Author: Ahmed Essam Abd Elgwad")
    print("=" * 50)
    
    # More complex documents
    documents = [
        "Machine learning algorithms require large datasets for training effective models",
        "Deep learning neural networks can automatically extract features from raw data",
        "Natural language processing enables computers to understand and generate human text",
        "Computer vision systems use convolutional neural networks for image recognition",
        "Reinforcement learning agents learn optimal strategies through trial and error",
        "Data preprocessing is crucial for building robust machine learning pipelines",
        "Feature engineering helps improve model performance and interpretability",
        "Cross-validation techniques ensure models generalize well to unseen data"
    ]
    
    print(f"Analyzing {len(documents)} technical documents...")
    
    # Advanced configuration
    vectorizer = TFIDFVectorizer(
        lowercase=True,
        remove_punctuation=True,
        min_df=2,  # Ignore terms appearing in < 2 documents
        max_df=0.7,  # Ignore terms appearing in > 70% of documents
        stop_words=['and', 'the', 'for', 'to', 'of', 'is', 'are', 'can', 'use', 'well']
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"\nVocabulary size: {len(vectorizer.get_feature_names())}")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    
    # Vocabulary statistics
    stats = TFIDFAnalyzer.get_vocabulary_stats(vectorizer)
    print(f"\nVocabulary Statistics:")
    print(f"  Total terms: {stats['vocab_size']}")
    print(f"  Average IDF: {stats['avg_idf']:.4f}")
    print(f"  Rarest terms: {stats['max_idf_terms']}")
    print(f"  Common terms: {stats['min_idf_terms']}")
    
    # Document similarity analysis
    print(f"\nDocument Similarity Analysis:")
    for i in range(min(3, len(documents))):  # Analyze first 3 documents
        similar_docs = TFIDFAnalyzer.find_similar_documents(tfidf_matrix, i, top_k=2)
        print(f"  Doc {i} most similar to:")
        for doc_idx, similarity in similar_docs:
            print(f"    Doc {doc_idx}: {similarity:.4f}")
    
    # Top terms analysis
    print(f"\nTop Terms Analysis:")
    for i in range(min(4, len(documents))):
        top_terms = TFIDFAnalyzer.get_top_terms(
            tfidf_matrix, vectorizer.get_feature_names(), i, top_k=5
        )
        print(f"  Doc {i}: {[(term, f'{score:.3f}') for term, score in top_terms]}")
    
    # Transform new document
    print(f"\nTransforming New Document:")
    new_doc = ["Advanced machine learning techniques for natural language understanding"]
    new_tfidf = vectorizer.transform(new_doc)
    
    new_top_terms = TFIDFAnalyzer.get_top_terms(
        new_tfidf, vectorizer.get_feature_names(), 0, top_k=5
    )
    print(f"  New document top terms: {[(term, f'{score:.3f}') for term, score in new_top_terms]}")
    
    # Find most similar existing document
    similarities = []
    for i in range(len(documents)):
        sim = TFIDFAnalyzer.document_similarity(
            np.vstack([tfidf_matrix, new_tfidf]), i, len(documents)
        )
        similarities.append((i, sim))
    
    best_match = max(similarities, key=lambda x: x[1])
    print(f"  Most similar to Doc {best_match[0]} (similarity: {best_match[1]:.4f})")
    print(f"  Which is: '{documents[best_match[0]][:60]}...'")
    
    print("\nAdvanced demo completed!")

if __name__ == "__main__":
    main()