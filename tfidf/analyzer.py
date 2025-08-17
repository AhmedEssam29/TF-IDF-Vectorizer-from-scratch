#!/usr/bin/env python3
"""
TF-IDF Analysis Utilities

Helper functions for analyzing TF-IDF results including document similarity
and top terms extraction.

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Union

__author__ = "Ahmed Essam Abd Elgwad"


class TFIDFAnalyzer:
    """
    Helper class for analyzing TF-IDF results.
    
    Author: Ahmed Essam Abd Elgwad
    """
    
    @staticmethod
    def get_top_terms(tfidf_matrix: np.ndarray, 
                     feature_names: List[str], 
                     doc_idx: int, 
                     top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top k terms for a specific document.
        
        Author: Ahmed Essam Abd Elgwad
        """
        doc_scores = tfidf_matrix[doc_idx]
        top_indices = doc_scores.argsort()[::-1][:top_k]
        
        return [(feature_names[idx], doc_scores[idx]) 
                for idx in top_indices if doc_scores[idx] > 0]
    
    @staticmethod
    def document_similarity(tfidf_matrix: np.ndarray, 
                          doc1_idx: int, 
                          doc2_idx: int) -> float:
        """
        Calculate cosine similarity between two documents.
        
        Author: Ahmed Essam Abd Elgwad
        """
        vec1 = tfidf_matrix[doc1_idx]
        vec2 = tfidf_matrix[doc2_idx]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def get_vocabulary_stats(vectorizer) -> Dict[str, Union[int, float, List[str]]]:
        """
        Get comprehensive vocabulary statistics.
        
        Author: Ahmed Essam Abd Elgwad
        """
        if not vectorizer.vocabulary_:
            raise ValueError("Vectorizer must be fitted first")
        
        idf_values = vectorizer.get_idf_values()
        
        vocab_size = len(vectorizer.vocabulary_)
        avg_idf = np.mean(list(idf_values.values()))
        
        sorted_terms = sorted(idf_values.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'vocab_size': vocab_size,
            'avg_idf': avg_idf,
            'max_idf_terms': [term for term, _ in sorted_terms[:5]],
            'min_idf_terms': [term for term, _ in sorted_terms[-5:]]
        }
    
    @staticmethod
    def find_similar_documents(tfidf_matrix: np.ndarray, 
                              doc_idx: int, 
                              top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar documents to a given document.
        
        Author: Ahmed Essam Abd Elgwad
        """
        similarities = []
        
        for i in range(tfidf_matrix.shape[0]):
            if i != doc_idx:
                sim = TFIDFAnalyzer.document_similarity(tfidf_matrix, doc_idx, i)
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def get_term_frequency_stats(vectorizer) -> Dict[str, Union[int, float]]:
        """
        Get term frequency statistics across the corpus.
        
        Author: Ahmed Essam Abd Elgwad
        """
        if not vectorizer.vocabulary_:
            raise ValueError("Vectorizer must be fitted first")
        
        idf_values = vectorizer.get_idf_values()
        n_docs = vectorizer.n_docs_
        
        # Calculate document frequencies from IDF values
        doc_freqs = {}
        for term, idf in idf_values.items():
            if idf > 0:
                doc_freqs[term] = int(n_docs / math.exp(idf))
            else:
                doc_freqs[term] = n_docs
        
        return {
            'total_terms': len(vectorizer.vocabulary_),
            'avg_doc_freq': np.mean(list(doc_freqs.values())),
            'max_doc_freq': max(doc_freqs.values()),
            'min_doc_freq': min(doc_freqs.values())
        }