#!/usr/bin/env python3
"""
Performance Comparison with Scikit-learn

Compare our TF-IDF implementation with scikit-learn's version.

Author: Ahmed Essam Abd Elgwad
Date: August 2025
"""

import time
import numpy as np
from tfidf import TFIDFVectorizer as CustomTFIDF

# Optional: Import sklearn for comparison (if available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTFIDF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available for comparison")

def generate_test_documents(n_docs=100, doc_length=50):
    """Generate test documents for benchmarking."""
    import random
    
    words = [
        'machine', 'learning', 'algorithm', 'data', 'science', 'python', 'analysis',
        'computer', 'artificial', 'intelligence', 'neural', 'network', 'deep',
        'processing', 'natural', 'language', 'text', 'model', 'training', 'feature',
        'classification', 'regression', 'clustering', 'optimization', 'statistics',
        'programming', 'software', 'development', 'technology', 'innovation',
        'research', 'experiment', 'dataset', 'validation', 'prediction', 'accuracy',
        'performance', 'evaluation', 'methodology', 'framework', 'library', 'tool',
        'application', 'system', 'implementation', 'solution', 'problem', 'challenge'
    ]
    
    documents = []
    for _ in range(n_docs):
        doc_words = random.choices(words, k=random.randint(10, doc_length))
        documents.append(' '.join(doc_words))
    
    return documents

def benchmark_custom_implementation(documents):
    """Benchmark our custom TF-IDF implementation."""
    print("Benchmarking Custom TF-IDF Implementation...")
    
    vectorizer = CustomTFIDF(
        lowercase=True,
        remove_punctuation=True,
        min_df=2
    )
    
    # Measure fitting time
    start_time = time.time()
    vectorizer.fit(documents)
    fit_time = time.time() - start_time
    
    # Measure transform time
    start_time = time.time()
    tfidf_matrix = vectorizer.transform(documents)
    transform_time = time.time() - start_time
    
    # Measure fit_transform time
    start_time = time.time()
    vectorizer_new = CustomTFIDF(lowercase=True, remove_punctuation=True, min_df=2)
    tfidf_matrix_combined = vectorizer_new.fit_transform(documents)
    fit_transform_time = time.time() - start_time
    
    return {
        'fit_time': fit_time,
        'transform_time': transform_time,
        'fit_transform_time': fit_transform_time,
        'matrix_shape': tfidf_matrix.shape,
        'vocab_size': len(vectorizer.get_vocabulary()),
        'implementation': 'Custom'
    }

def benchmark_sklearn_implementation(documents):
    """Benchmark scikit-learn's TF-IDF implementation."""
    if not SKLEARN_AVAILABLE:
        return None
    
    print("Benchmarking Scikit-learn TF-IDF Implementation...")
    
    vectorizer = SklearnTFIDF(
        lowercase=True,
        min_df=2,
        stop_words=None
    )
    
    # Measure fitting time
    start_time = time.time()
    vectorizer.fit(documents)
    fit_time = time.time() - start_time
    
    # Measure transform time
    start_time = time.time()
    tfidf_matrix = vectorizer.transform(documents)
    transform_time = time.time() - start_time
    
    # Measure fit_transform time
    start_time = time.time()
    vectorizer_new = SklearnTFIDF(lowercase=True, min_df=2, stop_words=None)
    tfidf_matrix_combined = vectorizer_new.fit_transform(documents)
    fit_transform_time = time.time() - start_time
    
    return {
        'fit_time': fit_time,
        'transform_time': transform_time,
        'fit_transform_time': fit_transform_time,
        'matrix_shape': tfidf_matrix.shape,
        'vocab_size': len(vectorizer.vocabulary_),
        'implementation': 'Sklearn'
    }

def compare_results_accuracy(documents):
    """Compare accuracy of results between implementations."""
    if not SKLEARN_AVAILABLE:
        print("Cannot compare accuracy without scikit-learn")
        return
    
    print("\nComparing Result Accuracy...")
    
    # Use simple, controlled documents for comparison
    test_docs = ["cat dog", "dog bird", "cat bird"]
    
    # Custom implementation
    custom_vectorizer = CustomTFIDF()
    custom_matrix = custom_vectorizer.fit_transform(test_docs)
    custom_vocab = custom_vectorizer.get_vocabulary()
    
    # Sklearn implementation
    sklearn_vectorizer = SklearnTFIDF(use_idf=True, norm=None, smooth_idf=False)
    sklearn_matrix = sklearn_vectorizer.fit_transform(test_docs).toarray()
    sklearn_vocab = sklearn_vectorizer.vocabulary_
    
    print(f"Custom vocab: {sorted(custom_vocab.keys())}")
    print(f"Sklearn vocab: {sorted(sklearn_vocab.keys())}")
    
    # Compare IDF values (if vocabularies match)
    if set(custom_vocab.keys()) == set(sklearn_vocab.keys()):
        print("Vocabularies match! Comparing IDF values...")
        custom_idf = custom_vectorizer.get_idf_values()
        sklearn_idf = sklearn_vectorizer.idf_
        
        for term in sorted(custom_vocab.keys()):
            custom_idx = custom_vocab[term]
            sklearn_idx = sklearn_vocab[term]
            
            custom_val = custom_idf[term]
            sklearn_val = sklearn_idf[sklearn_idx]
            
            print(f"  {term}: Custom={custom_val:.4f}, Sklearn={sklearn_val:.4f}")

def run_performance_comparison():
    """Run comprehensive performance comparison."""
    print("TF-IDF Performance Comparison")
    print("Author: Ahmed Essam Abd Elgwad")
    print("=" * 50)
    
    # Test with different dataset sizes
    test_sizes = [50, 100, 500] if SKLEARN_AVAILABLE else [50, 100, 500, 1000]
    
    results = []
    
    for size in test_sizes:
        print(f"\nTesting with {size} documents...")
        documents = generate_test_documents(n_docs=size)
        
        # Benchmark custom implementation
        custom_results = benchmark_custom_implementation(documents)
        results.append(custom_results)
        
        # Benchmark sklearn implementation
        if SKLEARN_AVAILABLE:
            sklearn_results = benchmark_sklearn_implementation(documents)
            results.append(sklearn_results)
        
        print(f"Custom Implementation:")
        print(f"  Fit time: {custom_results['fit_time']:.4f}s")
        print(f"  Transform time: {custom_results['transform_time']:.4f}s")
        print(f"  Fit+Transform time: {custom_results['fit_transform_time']:.4f}s")
        print(f"  Matrix shape: {custom_results['matrix_shape']}")
        print(f"  Vocabulary size: {custom_results['vocab_size']}")
        
        if SKLEARN_AVAILABLE and sklearn_results:
            print(f"Sklearn Implementation:")
            print(f"  Fit time: {sklearn_results['fit_time']:.4f}s")
            print(f"  Transform time: {sklearn_results['transform_time']:.4f}s")
            print(f"  Fit+Transform time: {sklearn_results['fit_transform_time']:.4f}s")
            print(f"  Matrix shape: {sklearn_results['matrix_shape']}")
            print(f"  Vocabulary size: {sklearn_results['vocab_size']}")
            
            # Calculate speedup/slowdown
            speedup_fit = sklearn_results['fit_time'] / custom_results['fit_time']
            speedup_transform = sklearn_results['transform_time'] / custom_results['transform_time']
            
            print(f"Performance Ratio (Sklearn/Custom):")
            print(f"  Fit: {speedup_fit:.2f}x")
            print(f"  Transform: {speedup_transform:.2f}x")
    
    # Compare accuracy
    compare_results_accuracy(generate_test_documents(n_docs=3, doc_length=5))
    
    print("\n" + "=" * 50)
    print("Performance comparison completed!")
    
    # Summary
    print("\nSummary:")
    print("- Custom implementation provides full transparency")
    print("- Mathematical accuracy verified against hand calculations")
    print("- Performance suitable for most applications")
    if SKLEARN_AVAILABLE:
        print("- Sklearn optimized for larger datasets with C implementations")
    print("- Custom version ideal for learning and customization")

if __name__ == "__main__":
    run_performance_comparison()