#!/usr/bin/env python3
"""
TF-IDF Vectorizer Implementation from Scratch

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT

A complete implementation of Term Frequency-Inverse Document Frequency (TF-IDF) 
vectorizer built from scratch using pure Python and NumPy.
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Union
import numpy as np

__author__ = "Ahmed Essam Abd Elgwad"
__version__ = "1.0.0"


class TFIDFVectorizer:
    """
    A complete TF-IDF implementation for text analysis.
    
    Author: Ahmed Essam Abd Elgwad
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 min_df: int = 1,
                 max_df: float = 1.0,
                 stop_words: List[str] = None):
        """Initialize the TF-IDF vectorizer."""
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = set(stop_words) if stop_words else set()
        
        # Will be populated during fitting
        self.vocabulary_ = {}
        self.idf_values_ = {}
        self.n_docs_ = 0
        self.feature_names_ = []
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing and cleaning."""
        if self.lowercase:
            text = text.lower()
            
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
        
        tokens = text.strip().split()
        
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        return tokens
    
    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """Build vocabulary from documents with frequency filtering."""
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = self._preprocess_text(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        n_docs = len(documents)
        vocabulary = {}
        idx = 0
        
        for term, freq in doc_freq.items():
            if freq >= self.min_df and freq <= (self.max_df * n_docs):
                vocabulary[term] = idx
                idx += 1
                
        return vocabulary
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate Term Frequency for a document."""
        if not tokens:
            return {}
            
        token_count = Counter(tokens)
        doc_length = len(tokens)
        
        tf_scores = {}
        for token, count in token_count.items():
            if token in self.vocabulary_:
                tf_scores[token] = count / doc_length
                
        return tf_scores
    
    def _calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        """Calculate Inverse Document Frequency for all terms."""
        n_docs = len(documents)
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = set(self._preprocess_text(doc))
            for token in tokens:
                if token in self.vocabulary_:
                    doc_freq[token] += 1
        
        idf_values = {}
        for term in self.vocabulary_:
            if doc_freq[term] > 0:
                idf_values[term] = math.log(n_docs / doc_freq[term])
            else:
                idf_values[term] = 0
                
        return idf_values
    
    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """Learn vocabulary and IDF from training documents."""
        self.n_docs_ = len(documents)
        
        self.vocabulary_ = self._build_vocabulary(documents)
        self.feature_names_ = sorted(self.vocabulary_.keys(), 
                                   key=lambda x: self.vocabulary_[x])
        
        self.idf_values_ = self._calculate_idf(documents)
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors."""
        if not self.vocabulary_:
            raise ValueError("Vectorizer has not been fitted yet. Call fit() first.")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._preprocess_text(doc)
            tf_scores = self._calculate_tf(tokens)
            
            for term, tf in tf_scores.items():
                if term in self.vocabulary_:
                    feature_idx = self.vocabulary_[term]
                    idf = self.idf_values_[term]
                    tfidf_matrix[doc_idx, feature_idx] = tf * idf
        
        return tfidf_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Learn vocabulary and transform documents in one step."""
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary terms)."""
        return self.feature_names_
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get the fitted vocabulary."""
        return self.vocabulary_
    
    def get_idf_values(self) -> Dict[str, float]:
        """Get IDF values for all terms."""
        return self.idf_values_