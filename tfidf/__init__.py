"""
TF-IDF from Scratch

A complete implementation of Term Frequency-Inverse Document Frequency
vectorization built from scratch in Python.

Author: Ahmed Essam Abd Elgwad
Email: your.email@example.com
License: MIT
Date: August 2025
"""

from .vectorizer import TFIDFVectorizer
from .analyzer import TFIDFAnalyzer
from .utils import TextPreprocessor, VocabularyBuilder

__version__ = "1.0.0"
__author__ = "Ahmed Essam Abd Elgwad"
__email__ = "your.email@example.com"
__license__ = "MIT"

__all__ = [
    "TFIDFVectorizer", 
    "TFIDFAnalyzer", 
    "TextPreprocessor", 
    "VocabularyBuilder"
]