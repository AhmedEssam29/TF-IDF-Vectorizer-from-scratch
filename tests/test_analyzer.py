#!/usr/bin/env python3
"""
Unit Tests for TFIDFAnalyzer

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT
"""

import unittest
import numpy as np
from tfidf import TFIDFVectorizer, TFIDFAnalyzer

__author__ = "Ahmed Essam Abd Elgwad"


class TestTFIDFAnalyzer(unittest.TestCase):
    """Test cases for TFIDFAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.docs = ["cat dog animal", "dog bird animal", "cat bird animal pet"]
        self.vectorizer = TFIDFVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
        self.feature_names = self.vectorizer.get_feature_names()
    
    def test_get_top_terms(self):
        """Test top terms extraction."""
        top_terms = TFIDFAnalyzer.get_top_terms(
            self.tfidf_matrix, self.feature_names, doc_idx=0, top_k=3
        )
        
        self.assertIsInstance(top_terms, list)
        self.assertTrue(len(top_terms) <= 3)
        
        for term, score in top_terms:
            self.assertIsInstance(term, str)
            self.assertIsInstance(score, (int, float))
            self.assertGreater(score, 0)
    
    def test_document_similarity(self):
        """Test document similarity calculation."""
        # Test self-similarity
        sim_self = TFIDFAnalyzer.document_similarity(self.tfidf_matrix, 0, 0)
        self.assertAlmostEqual(sim_self, 1.0, places=5)
        
        # Test similarity range
        sim_diff = TFIDFAnalyzer.document_similarity(self.tfidf_matrix, 0, 1)
        self.assertTrue(0 <= sim_diff <= 1)
        
        # Test symmetry
        sim_reverse = TFIDFAnalyzer.document_similarity(self.tfidf_matrix, 1, 0)
        self.assertAlmostEqual(sim_diff, sim_reverse, places=5)


if __name__ == '__main__':
    unittest.main()