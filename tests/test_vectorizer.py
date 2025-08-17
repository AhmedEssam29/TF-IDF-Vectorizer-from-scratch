#!/usr/bin/env python3
"""
Unit Tests for TFIDFVectorizer

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT
"""

import unittest
import numpy as np
import math
from tfidf import TFIDFVectorizer

__author__ = "Ahmed Essam Abd Elgwad"


class TestTFIDFVectorizer(unittest.TestCase):
    """Test cases for TFIDFVectorizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_docs = ["cat dog", "dog bird", "cat bird"]
        self.vectorizer = TFIDFVectorizer()
    
    def test_basic_functionality(self):
        """Test basic fit and transform functionality."""
        tfidf_matrix = self.vectorizer.fit_transform(self.simple_docs)
        
        self.assertEqual(tfidf_matrix.shape[0], len(self.simple_docs))
        self.assertTrue(tfidf_matrix.shape[1] > 0)
        self.assertFalse(np.isnan(tfidf_matrix).any())
        self.assertTrue((tfidf_matrix >= 0).all())
    
    def test_mathematical_correctness(self):
        """Test mathematical accuracy of TF-IDF calculations."""
        docs = ["cat dog cat", "dog bird"]
        vectorizer = TFIDFVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        vocab = vectorizer.get_vocabulary()
        idf_values = vectorizer.get_idf_values()
        
        # Manual IDF calculations
        expected_idf = {
            "cat": math.log(2/1),    # appears in 1/2 documents
            "dog": math.log(2/2),    # appears in 2/2 documents  
            "bird": math.log(2/1)    # appears in 1/2 documents
        }
        
        for word, expected in expected_idf.items():
            actual = idf_values[word]
            self.assertAlmostEqual(actual, expected, places=5)
        
        # Test specific TF-IDF calculation
        cat_idx = vocab["cat"]
        expected_tfidf_cat = (2/3) * math.log(2)  # TF * IDF
        actual_tfidf_cat = tfidf_matrix[0, cat_idx]
        self.assertAlmostEqual(actual_tfidf_cat, expected_tfidf_cat, places=5)


if __name__ == '__main__':
    unittest.main()