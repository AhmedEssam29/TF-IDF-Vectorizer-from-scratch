#!/usr/bin/env python3
"""
Unit Tests for Utility Functions

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT
"""

import unittest
from tfidf.utils import TextPreprocessor, VocabularyBuilder

__author__ = "Ahmed Essam Abd Elgwad"


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "Hello, World! 123"
        
        # Test lowercase
        cleaned = TextPreprocessor.clean_text(text, lowercase=True)
        self.assertIn("hello", cleaned)
        
        # Test punctuation removal
        cleaned = TextPreprocessor.clean_text(text, remove_punctuation=True)
        self.assertNotIn(",", cleaned)
        self.assertNotIn("!", cleaned)
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "hello world test"
        tokens = TextPreprocessor.tokenize(text)
        
        self.assertEqual(tokens, ["hello", "world", "test"])
        
        # Test with stop words
        stop_words = {"the", "hello"}
        tokens = TextPreprocessor.tokenize("hello the world", stop_words)
        self.assertEqual(tokens, ["world"])


class TestVocabularyBuilder(unittest.TestCase):
    """Test cases for VocabularyBuilder."""
    
    def test_build_vocabulary(self):
        """Test vocabulary building."""
        docs = ["hello world", "world peace", "hello peace"]
        vocab = VocabularyBuilder.build_vocabulary_from_documents(docs)
        
        expected_words = {"hello", "world", "peace"}
        self.assertEqual(set(vocab.keys()), expected_words)


if __name__ == '__main__':
    unittest.main()