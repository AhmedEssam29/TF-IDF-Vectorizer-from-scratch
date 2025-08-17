#!/usr/bin/env python3
"""
Integration Tests for TF-IDF System

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT
"""

import unittest
import time
from tfidf import TFIDFVectorizer, TFIDFAnalyzer

__author__ = "Ahmed Essam Abd Elgwad"


class TestIntegration(unittest.TestCase):
    """Integration tests for complete TF-IDF system."""
    
    def test_realistic_scenario(self):
        """Test with realistic text analysis scenario."""
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand human language",
            "Computer vision enables machines to interpret visual information",
            "Artificial intelligence encompasses many different technologies"
        ]
        
        vectorizer = TFIDFVectorizer(
            lowercase=True,
            remove_punctuation=True,
            stop_words=["is", "a", "of", "to", "and", "the", "with"]
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Verify results
        self.assertEqual(tfidf_matrix.shape[0], len(documents))
        self.assertGreater(tfidf_matrix.shape[1], 0)
        
        # Test analysis functions
        for doc_idx in range(len(documents)):
            top_terms = TFIDFAnalyzer.get_top_terms(
                tfidf_matrix, vectorizer.get_feature_names(), doc_idx, top_k=5
            )
            self.assertGreater(len(top_terms), 0)
    
    def test_performance_benchmark(self):
        """Test performance with larger dataset."""
        base_docs = [
            "artificial intelligence machine learning",
            "deep neural networks learning algorithms", 
            "natural language processing text analysis",
            "computer vision image recognition",
            "data science statistical analysis"
        ]
        
        large_docs = base_docs * 100  # 500 documents
        vectorizer = TFIDFVectorizer()
        
        start_time = time.time()
        tfidf_matrix = vectorizer.fit_transform(large_docs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time
        self.assertLess(processing_time, 10.0)
        self.assertEqual(tfidf_matrix.shape[0], len(large_docs))


if __name__ == '__main__':
    unittest.main()