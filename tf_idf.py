import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Union
import numpy as np

class TFIDFVectorizer:
    """
    A complete TF-IDF implementation for text analysis.
    
    Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic
    that reflects how important a word is to a document in a collection of documents.
    
    Mathematical formulas:
    - TF(t,d) = count(t,d) / |d|
    - IDF(t,D) = log(|D| / |{d ∈ D : t ∈ d}|)
    - TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 min_df: int = 1,
                 max_df: float = 1.0,
                 stop_words: List[str] = None):
        """
        Initialize the TF-IDF vectorizer.
        
        Parameters:
        -----------
        lowercase : bool, default=True
            Convert all text to lowercase
        remove_punctuation : bool, default=True
            Remove punctuation from text
        min_df : int, default=1
            Ignore terms that appear in fewer than min_df documents
        max_df : float, default=1.0
            Ignore terms that appear in more than max_df proportion of documents
        stop_words : List[str], optional
            List of words to ignore
        """
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
        """
        Preprocess text by tokenizing and cleaning.
        
        Parameters:
        -----------
        text : str
            Input text to preprocess
            
        Returns:
        --------
        List[str] : List of processed tokens
        """
        if self.lowercase:
            text = text.lower()
            
        if self.remove_punctuation:
            # Remove punctuation and extra whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        tokens = text.strip().split()
        
        # Remove stop words
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        return tokens
    
    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from documents with frequency filtering.
        
        Parameters:
        -----------
        documents : List[str]
            List of document strings
            
        Returns:
        --------
        Dict[str, int] : Vocabulary mapping word to index
        """
        # Count document frequency for each term
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = self._preprocess_text(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Filter terms based on document frequency
        n_docs = len(documents)
        vocabulary = {}
        idx = 0
        
        for term, freq in doc_freq.items():
            # Apply min_df and max_df filtering
            if freq >= self.min_df and freq <= (self.max_df * n_docs):
                vocabulary[term] = idx
                idx += 1
                
        return vocabulary
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate Term Frequency for a document.
        
        TF(t,d) = count(t,d) / |d|
        
        Parameters:
        -----------
        tokens : List[str]
            List of tokens in document
            
        Returns:
        --------
        Dict[str, float] : Term frequencies
        """
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
        """
        Calculate Inverse Document Frequency for all terms.
        
        IDF(t,D) = log(|D| / |{d ∈ D : t ∈ d}|)
        
        Parameters:
        -----------
        documents : List[str]
            List of document strings
            
        Returns:
        --------
        Dict[str, float] : IDF values for each term
        """
        n_docs = len(documents)
        doc_freq = defaultdict(int)
        
        # Count document frequency for each term in vocabulary
        for doc in documents:
            tokens = set(self._preprocess_text(doc))
            for token in tokens:
                if token in self.vocabulary_:
                    doc_freq[token] += 1
        
        # Calculate IDF
        idf_values = {}
        for term in self.vocabulary_:
            if doc_freq[term] > 0:
                idf_values[term] = math.log(n_docs / doc_freq[term])
            else:
                idf_values[term] = 0
                
        return idf_values
    
    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Learn vocabulary and IDF from training documents.
        
        Parameters:
        -----------
        documents : List[str]
            List of document strings to learn from
            
        Returns:
        --------
        self : TFIDFVectorizer
            Returns the instance itself
        """
        self.n_docs_ = len(documents)
        
        # Build vocabulary
        self.vocabulary_ = self._build_vocabulary(documents)
        self.feature_names_ = sorted(self.vocabulary_.keys(), 
                                   key=lambda x: self.vocabulary_[x])
        
        # Calculate IDF values
        self.idf_values_ = self._calculate_idf(documents)
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors.
        
        Parameters:
        -----------
        documents : List[str]
            Documents to transform
            
        Returns:
        --------
        np.ndarray : TF-IDF matrix (n_docs, n_features)
        """
        if not self.vocabulary_:
            raise ValueError("Vectorizer has not been fitted yet. Call fit() first.")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        
        # Initialize TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._preprocess_text(doc)
            tf_scores = self._calculate_tf(tokens)
            
            # Calculate TF-IDF for each term
            for term, tf in tf_scores.items():
                if term in self.vocabulary_:
                    feature_idx = self.vocabulary_[term]
                    idf = self.idf_values_[term]
                    tfidf_matrix[doc_idx, feature_idx] = tf * idf
        
        return tfidf_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Learn vocabulary and transform documents in one step.
        
        Parameters:
        -----------
        documents : List[str]
            Documents to learn from and transform
            
        Returns:
        --------
        np.ndarray : TF-IDF matrix
        """
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary terms).
        
        Returns:
        --------
        List[str] : List of feature names
        """
        return self.feature_names_
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Get the fitted vocabulary.
        
        Returns:
        --------
        Dict[str, int] : Vocabulary mapping
        """
        return self.vocabulary_
    
    def get_idf_values(self) -> Dict[str, float]:
        """
        Get IDF values for all terms.
        
        Returns:
        --------
        Dict[str, float] : IDF values
        """
        return self.idf_values_


# Example usage and testing
if __name__ == "__main__":
    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "Cats and dogs are pets",
        "I love my pet cat very much",
        "The park has many trees and flowers"
    ]
    
    # Initialize and fit the vectorizer
    vectorizer = TFIDFVectorizer(lowercase=True, remove_punctuation=True)
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print("Vocabulary:", vectorizer.get_feature_names())
    print("\nTF-IDF Matrix shape:", tfidf_matrix.shape)
    print("\nTF-IDF Matrix:")
    print(tfidf_matrix)
    
    # Show IDF values
    print("\nIDF values:")
    for term, idf in sorted(vectorizer.get_idf_values().items()):
        print(f"{term}: {idf:.4f}")
    
    # Transform new documents
    new_docs = ["The cat loves the park"]
    new_tfidf = vectorizer.transform(new_docs)
    print(f"\nNew document TF-IDF: {new_tfidf}")


class TFIDFAnalyzer:
    """
    Helper class for analyzing TF-IDF results.
    """
    
    @staticmethod
    def get_top_terms(tfidf_matrix: np.ndarray, 
                     feature_names: List[str], 
                     doc_idx: int, 
                     top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top k terms for a specific document.
        
        Parameters:
        -----------
        tfidf_matrix : np.ndarray
            TF-IDF matrix
        feature_names : List[str]
            Feature names (vocabulary)
        doc_idx : int
            Document index
        top_k : int, default=10
            Number of top terms to return
            
        Returns:
        --------
        List[Tuple[str, float]] : List of (term, score) pairs
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
        
        Parameters:
        -----------
        tfidf_matrix : np.ndarray
            TF-IDF matrix
        doc1_idx : int
            First document index
        doc2_idx : int
            Second document index
            
        Returns:
        --------
        float : Cosine similarity score
        """
        vec1 = tfidf_matrix[doc1_idx]
        vec2 = tfidf_matrix[doc2_idx]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)