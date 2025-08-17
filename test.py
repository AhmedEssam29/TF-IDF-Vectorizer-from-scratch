#!/usr/bin/env python3
"""
Complete testing script for TF-IDF module.
Tests functionality, edge cases, and validates mathematical correctness.
"""

import unittest
import numpy as np
import math
from typing import List, Dict

# Import your TF-IDF module
# Assuming your module is saved as 'tfidf_vectorizer.py'
from tf_idf import TFIDFVectorizer, TFIDFAnalyzer


class TestTFIDFVectorizer(unittest.TestCase):
    """Unit tests for TFIDFVectorizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.simple_docs = [
            "cat dog",
            "dog bird", 
            "cat bird"
        ]
        
        self.complex_docs = [
            "The cat sat on the mat",
            "The dog ran in the park", 
            "Cats and dogs are pets",
            "I love my pet cat very much",
            "The park has many trees and flowers"
        ]
        
        self.vectorizer = TFIDFVectorizer()
    
    def test_basic_functionality(self):
        """Test basic fit and transform functionality."""
        tfidf_matrix = self.vectorizer.fit_transform(self.simple_docs)
        
        # Check matrix dimensions
        self.assertEqual(tfidf_matrix.shape[0], len(self.simple_docs))
        self.assertTrue(tfidf_matrix.shape[1] > 0)  # Should have features
        
        # Check that matrix contains valid values
        self.assertFalse(np.isnan(tfidf_matrix).any())
        self.assertTrue((tfidf_matrix >= 0).all())  # TF-IDF scores should be non-negative
    
    def test_vocabulary_building(self):
        """Test vocabulary construction."""
        self.vectorizer.fit(self.simple_docs)
        vocab = self.vectorizer.get_vocabulary()
        feature_names = self.vectorizer.get_feature_names()
        
        # Check vocabulary contains expected words
        expected_words = {"cat", "dog", "bird"}
        self.assertEqual(set(feature_names), expected_words)
        
        # Check vocabulary mapping is correct
        self.assertEqual(len(vocab), len(feature_names))
        for word in expected_words:
            self.assertIn(word, vocab)
            self.assertTrue(0 <= vocab[word] < len(feature_names))
    
    def test_mathematical_correctness(self):
        """Test that TF-IDF calculations are mathematically correct."""
        # Use simple example where we can calculate by hand
        docs = ["cat dog cat", "dog bird"]
        vectorizer = TFIDFVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        vocab = vectorizer.get_vocabulary()
        idf_values = vectorizer.get_idf_values()
        
        # Manual calculations:
        # Document 0: "cat dog cat" -> cat appears 2/3 times, dog appears 1/3 times
        # Document 1: "dog bird" -> dog appears 1/2 times, bird appears 1/2 times
        
        # IDF calculations:
        # cat: appears in 1/2 documents -> IDF = log(2/1) = log(2) ≈ 0.693
        # dog: appears in 2/2 documents -> IDF = log(2/2) = log(1) = 0
        # bird: appears in 1/2 documents -> IDF = log(2/1) = log(2) ≈ 0.693
        
        expected_idf = {
            "cat": math.log(2/1),
            "dog": math.log(2/2), 
            "bird": math.log(2/1)
        }
        
        for word, expected in expected_idf.items():
            actual = idf_values[word]
            self.assertAlmostEqual(actual, expected, places=5, 
                                 msg=f"IDF for '{word}' incorrect")
        
        # TF-IDF for "cat" in document 0: (2/3) * log(2) ≈ 0.462
        cat_idx = vocab["cat"]
        expected_tfidf_cat = (2/3) * math.log(2)
        actual_tfidf_cat = tfidf_matrix[0, cat_idx]
        self.assertAlmostEqual(actual_tfidf_cat, expected_tfidf_cat, places=5)
    
    def test_preprocessing(self):
        """Test text preprocessing functionality."""
        docs_with_punctuation = [
            "Hello, World!",
            "How are you?",
            "I'm fine, thanks."
        ]
        
        # Test with punctuation removal
        vectorizer1 = TFIDFVectorizer(remove_punctuation=True)
        vectorizer1.fit(docs_with_punctuation)
        features1 = set(vectorizer1.get_feature_names())
        
        # Should not contain punctuation
        self.assertNotIn("hello,", features1)
        self.assertNotIn("world!", features1)
        self.assertIn("hello", features1)
        self.assertIn("world", features1)
        
        # Test case sensitivity
        docs_mixed_case = ["Cat DOG", "cat dog"]
        
        vectorizer2 = TFIDFVectorizer(lowercase=True)
        vectorizer2.fit(docs_mixed_case)
        features2 = vectorizer2.get_feature_names()
        
        self.assertIn("cat", features2)
        self.assertIn("dog", features2)
        self.assertNotIn("Cat", features2)
        self.assertNotIn("DOG", features2)
    
    def test_frequency_filtering(self):
        """Test min_df and max_df filtering."""
        docs = [
            "common word rare1",
            "common word rare2", 
            "common word rare3",
            "common different"
        ]
        
        # Test min_df filtering (remove words appearing in < 2 documents)
        vectorizer = TFIDFVectorizer(min_df=2)
        vectorizer.fit(docs)
        features = set(vectorizer.get_feature_names())
        
        # "common" and "word" appear in 3+ docs, should be included
        self.assertIn("common", features)
        self.assertIn("word", features)
        
        # rare1, rare2, rare3 appear in only 1 doc each, should be excluded
        self.assertNotIn("rare1", features)
        self.assertNotIn("rare2", features) 
        self.assertNotIn("rare3", features)
        
        # Test max_df filtering
        vectorizer2 = TFIDFVectorizer(max_df=0.5)  # Remove words in >50% of docs
        vectorizer2.fit(docs)
        features2 = set(vectorizer2.get_feature_names())
        
        # "common" appears in 4/4 = 100% of docs, should be excluded
        self.assertNotIn("common", features2)
    
    def test_stop_words(self):
        """Test stop words filtering."""
        docs = ["the cat sat", "a dog ran"]
        stop_words = ["the", "a"]
        
        vectorizer = TFIDFVectorizer(stop_words=stop_words)
        vectorizer.fit(docs)
        features = set(vectorizer.get_feature_names())
        
        # Stop words should be excluded
        self.assertNotIn("the", features)
        self.assertNotIn("a", features)
        
        # Content words should be included
        self.assertIn("cat", features)
        self.assertIn("dog", features)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        
        # Test empty documents
        docs_with_empty = ["cat dog", "", "bird"]
        vectorizer = TFIDFVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs_with_empty)
        
        # Should handle gracefully
        self.assertEqual(tfidf_matrix.shape[0], 3)
        # Empty document should have all zeros
        self.assertEqual(tfidf_matrix[1].sum(), 0)
        
        # Test single document
        single_doc = ["only one document"]
        vectorizer2 = TFIDFVectorizer()
        tfidf_matrix2 = vectorizer2.fit_transform(single_doc)
        self.assertEqual(tfidf_matrix2.shape[0], 1)
        
        # Test transform before fit (should raise error)
        vectorizer3 = TFIDFVectorizer()
        with self.assertRaises(ValueError):
            vectorizer3.transform(["test"])
    
    def test_fit_transform_consistency(self):
        """Test that fit_transform gives same result as fit then transform."""
        vectorizer1 = TFIDFVectorizer()
        vectorizer2 = TFIDFVectorizer()
        
        # Method 1: fit_transform
        tfidf1 = vectorizer1.fit_transform(self.simple_docs)
        
        # Method 2: fit then transform
        vectorizer2.fit(self.simple_docs)
        tfidf2 = vectorizer2.transform(self.simple_docs)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(tfidf1, tfidf2)
    
    def test_new_document_transform(self):
        """Test transforming documents with new/unknown terms."""
        # Train on subset
        train_docs = ["cat dog", "bird cat"]
        test_docs = ["cat elephant", "dog bird", "zebra lion"]  # elephant, zebra, lion are new
        
        vectorizer = TFIDFVectorizer()
        vectorizer.fit(train_docs)
        
        # Transform test docs
        tfidf_matrix = vectorizer.transform(test_docs)
        
        # Should work without errors
        self.assertEqual(tfidf_matrix.shape[0], len(test_docs))
        self.assertEqual(tfidf_matrix.shape[1], len(vectorizer.get_vocabulary()))
        
        # Documents with only new terms should have zero vectors
        self.assertEqual(tfidf_matrix[2].sum(), 0)  # "zebra lion" - all new terms


class TestTFIDFAnalyzer(unittest.TestCase):
    """Unit tests for TFIDFAnalyzer helper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.docs = [
            "cat dog animal",
            "dog bird animal", 
            "cat bird animal pet"
        ]
        
        self.vectorizer = TFIDFVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
        self.feature_names = self.vectorizer.get_feature_names()
    
    def test_get_top_terms(self):
        """Test getting top terms for a document."""
        top_terms = TFIDFAnalyzer.get_top_terms(
            self.tfidf_matrix, self.feature_names, doc_idx=0, top_k=3
        )
        
        # Should return list of (term, score) tuples
        self.assertIsInstance(top_terms, list)
        self.assertTrue(len(top_terms) <= 3)
        
        for term, score in top_terms:
            self.assertIsInstance(term, str)
            self.assertIsInstance(score, (int, float))
            self.assertGreater(score, 0)
        
        # Terms should be sorted by score (descending)
        scores = [score for _, score in top_terms]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_document_similarity(self):
        """Test document similarity calculation."""
        # Test similarity between identical documents
        sim_self = TFIDFAnalyzer.document_similarity(self.tfidf_matrix, 0, 0)
        self.assertAlmostEqual(sim_self, 1.0, places=5)
        
        # Test similarity between different documents
        sim_diff = TFIDFAnalyzer.document_similarity(self.tfidf_matrix, 0, 1)
        self.assertTrue(0 <= sim_diff <= 1)
        
        # Similarity should be symmetric
        sim_reverse = TFIDFAnalyzer.document_similarity(self.tfidf_matrix, 1, 0)
        self.assertAlmostEqual(sim_diff, sim_reverse, places=5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete TF-IDF system."""
    
    def test_realistic_example(self):
        """Test with a realistic text analysis scenario."""
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand human language",
            "Computer vision enables machines to interpret visual information",
            "Artificial intelligence encompasses many different technologies"
        ]
        
        # Initialize with reasonable parameters
        vectorizer = TFIDFVectorizer(
            lowercase=True,
            remove_punctuation=True,
            min_df=1,
            stop_words=["is", "a", "of", "to", "and", "the", "with"]
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Verify reasonable results
        self.assertEqual(tfidf_matrix.shape[0], len(documents))
        self.assertGreater(tfidf_matrix.shape[1], 0)
        
        # Check that we can identify key terms for each document
        for doc_idx in range(len(documents)):
            top_terms = TFIDFAnalyzer.get_top_terms(
                tfidf_matrix, vectorizer.get_feature_names(), doc_idx, top_k=5
            )
            self.assertGreater(len(top_terms), 0)
            
            # Print for manual inspection (optional)
            print(f"\nDocument {doc_idx}: '{documents[doc_idx][:50]}...'")
            print("Top terms:", top_terms[:3])
    
    def test_performance_benchmark(self):
        """Basic performance test with larger dataset."""
        import time
        
        # Generate larger dataset
        base_docs = [
            "artificial intelligence machine learning",
            "deep neural networks learning algorithms", 
            "natural language processing text analysis",
            "computer vision image recognition",
            "data science statistical analysis"
        ]
        
        # Repeat documents to create larger dataset
        large_docs = base_docs * 100  # 500 documents
        
        vectorizer = TFIDFVectorizer()
        
        start_time = time.time()
        tfidf_matrix = vectorizer.fit_transform(large_docs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"\nProcessed {len(large_docs)} documents in {processing_time:.2f} seconds")
        print(f"Matrix shape: {tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(vectorizer.get_vocabulary())}")
        
        # Basic performance assertion (should complete in reasonable time)
        self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds


def run_manual_tests():
    """Manual tests for visual inspection."""
    print("=== MANUAL TESTS FOR VISUAL INSPECTION ===")
    
    # Test 1: Simple example with known results
    print("\n1. Simple Example:")
    docs = ["cat dog", "dog bird", "cat bird"]
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    print(f"Documents: {docs}")
    print(f"Vocabulary: {vectorizer.get_feature_names()}")
    print(f"TF-IDF Matrix:\n{tfidf_matrix}")
    
    print("\nIDF values:")
    for term, idf in sorted(vectorizer.get_idf_values().items()):
        print(f"  {term}: {idf:.4f}")
    
    # Test 2: Top terms analysis
    print("\n2. Top Terms Analysis:")
    for i, doc in enumerate(docs):
        top_terms = TFIDFAnalyzer.get_top_terms(
            tfidf_matrix, vectorizer.get_feature_names(), i, top_k=3
        )
        print(f"  Doc {i} ('{doc}'): {top_terms}")
    
    # Test 3: Document similarity
    print("\n3. Document Similarities:")
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            sim = TFIDFAnalyzer.document_similarity(tfidf_matrix, i, j)
            print(f"  Doc {i} vs Doc {j}: {sim:.4f}")


if __name__ == "__main__":
    print("Running TF-IDF Tests...")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    # Run manual tests
    print("\n" + "=" * 50)
    run_manual_tests()
    
    print("\n" + "=" * 50)
    print("All tests completed!")