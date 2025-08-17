#!/usr/bin/env python3
"""
Utility Functions for TF-IDF Implementation

Common utility functions for text preprocessing and vocabulary building.

Author: Ahmed Essam Abd Elgwad
Date: August 2025
License: MIT
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Set

__author__ = "Ahmed Essam Abd Elgwad"


class TextPreprocessor:
    """
    Text preprocessing utilities.
    
    Author: Ahmed Essam Abd Elgwad
    """
    
    @staticmethod
    def clean_text(text: str, 
                   lowercase: bool = True,
                   remove_punctuation: bool = True,
                   remove_numbers: bool = False) -> str:
        """
        Clean and normalize text.
        
        Author: Ahmed Essam Abd Elgwad
        """
        if lowercase:
            text = text.lower()
        
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def tokenize(text: str, stop_words: Set[str] = None) -> List[str]:
        """
        Tokenize text into words.
        
        Author: Ahmed Essam Abd Elgwad
        """
        tokens = text.split()
        
        if stop_words:
            tokens = [token for token in tokens if token not in stop_words]
        
        return [token for token in tokens if token]  # Remove empty strings
    
    @staticmethod
    def get_default_stop_words() -> Set[str]:
        """
        Get a default set of English stop words.
        
        Author: Ahmed Essam Abd Elgwad
        """
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by',
            'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of',
            'on', 'that', 'the', 'to', 'was', 'will', 'with', 'the',
            'this', 'but', 'they', 'have', 'had', 'what', 'said', 'each',
            'which', 'their', 'time', 'if', 'up', 'out', 'many', 'then',
            'them', 'can', 'she', 'may', 'or', 'more', 'these', 'so',
            'some', 'very', 'when', 'much', 'would', 'how', 'your', 'now',
            'than', 'first', 'been', 'call', 'who', 'oil', 'its', 'long',
            'down', 'day', 'did', 'get', 'has', 'her', 'his', 'how', 'man',
            'new', 'now', 'old', 'see', 'two', 'way', 'were', 'been'
        }


class VocabularyBuilder:
    """
    Vocabulary building utilities.
    
    Author: Ahmed Essam Abd Elgwad
    """
    
    @staticmethod
    def build_vocabulary_from_documents(documents: List[str],
                                      preprocessor_func=None,
                                      min_df: int = 1,
                                      max_df: float = 1.0) -> Dict[str, int]:
        """
        Build vocabulary from a list of documents.
        
        Author: Ahmed Essam Abd Elgwad
        """
        if preprocessor_func is None:
            preprocessor_func = lambda x: x.lower().split()
        
        # Count document frequency
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = set(preprocessor_func(doc))
            for token in tokens:
                doc_freq[token] += 1
        
        # Filter by frequency
        n_docs = len(documents)
        vocabulary = {}
        idx = 0
        
        for term, freq in doc_freq.items():
            if min_df <= freq <= (max_df * n_docs):
                vocabulary[term] = idx
                idx += 1
        
        return vocabulary
    
    @staticmethod
    def get_vocabulary_statistics(vocabulary: Dict[str, int],
                                 documents: List[str],
                                 preprocessor_func=None) -> Dict:
        """
        Get statistics about vocabulary coverage.
        
        Author: Ahmed Essam Abd Elgwad
        """
        if preprocessor_func is None:
            preprocessor_func = lambda x: x.lower().split()
        
        total_tokens = 0
        covered_tokens = 0
        
        for doc in documents:
            tokens = preprocessor_func(doc)
            total_tokens += len(tokens)
            covered_tokens += len([t for t in tokens if t in vocabulary])
        
        return {
            'vocab_size': len(vocabulary),
            'total_tokens': total_tokens,
            'covered_tokens': covered_tokens,
            'coverage_ratio': covered_tokens / total_tokens if total_tokens > 0 else 0
        }


def load_text_file(filepath: str, encoding: str = 'utf-8') -> str:
    """
    Load text from a file.
    
    Author: Ahmed Essam Abd Elgwad
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {str(e)}")


def save_vocabulary(vocabulary: Dict[str, int], filepath: str):
    """
    Save vocabulary to a file.
    
    Author: Ahmed Essam Abd Elgwad
    """
    import json
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise Exception(f"Error saving vocabulary to {filepath}: {str(e)}")


def load_vocabulary(filepath: str) -> Dict[str, int]:
    """
    Load vocabulary from a file.
    
    Author: Ahmed Essam Abd Elgwad
    """
    import json
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading vocabulary from {filepath}: {str(e)}")