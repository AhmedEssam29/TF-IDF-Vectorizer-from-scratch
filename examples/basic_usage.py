#!/usr/bin/env python3
"""
Basic Usage Example for TF-IDF Implementation

This example demonstrates fundamental usage patterns.

Author: Ahmed Essam Abd Elgwad
Date: August 2025
"""

from tfidf import TFIDFVectorizer, TFIDFAnalyzer

def main():
    """Basic usage demonstration."""
    print("TF-IDF Basic Usage Example")
    print("Author: Ahmed Essam Abd Elgwad")
    print("=" * 50)
    
    # Sample documents
    documents = [
        "I love machine learning and artificial intelligence",
        "Python is a great programming language for data science",
        "Machine learning algorithms can solve complex problems",
        "Data science combines statistics and programming skills"
    ]
    
    print(f"Analyzing {len(documents)} documents...")
    
    # Initialize vectorizer
    vectorizer = TFIDFVectorizer(
        lowercase=True,
        remove_punctuation=True,
        stop_words=['and', 'is', 'a', 'for', 'can']
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"\nVocabulary: {vectorizer.get_feature_names()}")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    
    # Show top terms for each document
    print("\nTop terms per document:")
    for i, doc in enumerate(documents):
        top_terms = TFIDFAnalyzer.get_top_terms(
            tfidf_matrix, vectorizer.get_feature_names(), i, top_k=3
        )
        print(f"Doc {i}: {[term for term, _ in top_terms]}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()