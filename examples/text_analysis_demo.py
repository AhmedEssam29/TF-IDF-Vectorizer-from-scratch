#!/usr/bin/env python3
"""
Real-World Text Analysis Demo

Demonstrates TF-IDF usage for practical text analysis tasks.

Author: Ahmed Essam Abd Elgwad
Date: August 2025
"""

from tfidf import TFIDFVectorizer, TFIDFAnalyzer
import numpy as np

def analyze_news_articles():
    """Analyze a collection of news articles."""
    print("News Articles Analysis")
    print("-" * 30)
    
    news_articles = [
        "Tech giants announce massive investments in artificial intelligence research and development",
        "Climate change summit reaches historic agreement on carbon emission reduction targets",
        "Stock market volatility continues as investors react to economic uncertainty and inflation fears",
        "Breakthrough in renewable energy technology promises cheaper and more efficient solar panels",
        "Cybersecurity experts warn of increasing threats from sophisticated ransomware attacks",
        "Healthcare innovation leads to new treatments for rare genetic diseases",
        "Space exploration mission successfully lands rover on Mars surface for geological studies",
        "Education sector embraces digital transformation with online learning platforms and tools"
    ]
    
    # Configure for news analysis
    vectorizer = TFIDFVectorizer(
        lowercase=True,
        remove_punctuation=True,
        min_df=1,
        stop_words=['and', 'the', 'of', 'to', 'for', 'in', 'on', 'with', 'as', 'by']
    )
    
    tfidf_matrix = vectorizer.fit_transform(news_articles)
    
    print(f"Analyzed {len(news_articles)} news articles")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names())}")
    
    # Extract key topics from each article
    print("\nKey Topics per Article:")
    topics = ["Technology", "Climate", "Finance", "Energy", "Security", "Healthcare", "Space", "Education"]
    
    for i, (article, topic) in enumerate(zip(news_articles, topics)):
        top_terms = TFIDFAnalyzer.get_top_terms(
            tfidf_matrix, vectorizer.get_feature_names(), i, top_k=4
        )
        print(f"  {topic}: {[term for term, _ in top_terms]}")
    
    # Find related articles
    print("\nRelated Articles Analysis:")
    tech_articles = [0, 4]  # Tech and cybersecurity
    for i in tech_articles:
        similar = TFIDFAnalyzer.find_similar_documents(tfidf_matrix, i, top_k=2)
        print(f"  Article {i} relates to articles: {[idx for idx, _ in similar]}")
    
    return vectorizer, tfidf_matrix, news_articles

def analyze_product_reviews():
    """Analyze product reviews to extract sentiments and features."""
    print("\nProduct Reviews Analysis")
    print("-" * 30)
    
    reviews = [
        "Amazing product with excellent build quality and fast performance",
        "Battery life is disappointing but camera quality is outstanding",
        "Great value for money with solid performance and reliable features",
        "Poor customer service experience and slow delivery times",
        "Innovative design with cutting-edge technology and premium materials",
        "Affordable price point but lacks advanced features and durability",
        "Exceptional user experience with intuitive interface and smooth operation",
        "Quality control issues and frequent software bugs affect usability"
    ]
    
    vectorizer = TFIDFVectorizer(
        lowercase=True,
        remove_punctuation=True,
        stop_words=['and', 'the', 'with', 'is', 'but', 'for']
    )
    
    tfidf_matrix = vectorizer.fit_transform(reviews)
    
    print(f"Analyzed {len(reviews)} product reviews")
    
    # Extract key aspects mentioned in reviews
    print("\nKey Aspects per Review:")
    for i, review in enumerate(reviews):
        top_terms = TFIDFAnalyzer.get_top_terms(
            tfidf_matrix, vectorizer.get_feature_names(), i, top_k=3
        )
        sentiment = "Positive" if any(word in review.lower() for word in 
                                    ['amazing', 'excellent', 'great', 'outstanding', 'exceptional']) else "Mixed/Negative"
        print(f"  Review {i} ({sentiment}): {[term for term, _ in top_terms]}")
    
    # Group similar reviews
    print("\nSimilar Review Groups:")
    processed = set()
    for i in range(len(reviews)):
        if i in processed:
            continue
        
        similar = TFIDFAnalyzer.find_similar_documents(tfidf_matrix, i, top_k=len(reviews)-1)
        group = [i]
        
        for doc_idx, similarity in similar:
            if similarity > 0.1 and doc_idx not in processed:  # Threshold for similarity
                group.append(doc_idx)
                processed.add(doc_idx)
        
        processed.add(i)
        if len(group) > 1:
            print(f"  Similar reviews: {group}")

def document_search_demo():
    """Demonstrate document search functionality."""
    print("\nDocument Search Demo")
    print("-" * 30)
    
    documents = [
        "Python programming language for data science and machine learning applications",
        "JavaScript web development with modern frameworks and responsive design",
        "Database management systems and SQL query optimization techniques",
        "Cloud computing platforms and distributed system architectures",
        "Mobile app development using React Native and Flutter frameworks",
        "DevOps practices including continuous integration and deployment pipelines",
        "Artificial intelligence algorithms for computer vision and natural language processing",
        "Cybersecurity best practices and network security implementation strategies"
    ]
    
    # Build search index
    vectorizer = TFIDFVectorizer(
        lowercase=True,
        remove_punctuation=True,
        stop_words=['and', 'for', 'with', 'using', 'the', 'of']
    )
    
    doc_vectors = vectorizer.fit_transform(documents)
    
    # Search queries
    queries = [
        "machine learning Python",
        "web development JavaScript",
        "cloud distributed systems",
        "mobile app React"
    ]
    
    print("Search Results:")
    for query in queries:
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities with all documents
        similarities = []
        for i in range(len(documents)):
            sim = TFIDFAnalyzer.document_similarity(
                np.vstack([doc_vectors, query_vector]), i, len(documents)
            )
            similarities.append((i, sim))
        
        # Sort by similarity and get top 2 results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:2]
        
        print(f"\n  Query: '{query}'")
        for doc_idx, similarity in top_results:
            if similarity > 0:
                print(f"    Doc {doc_idx} (score: {similarity:.3f}): {documents[doc_idx][:50]}...")

def main():
    """Run all text analysis demos."""
    print("Real-World Text Analysis Demo")
    print("Author: Ahmed Essam Abd Elgwad")
    print("=" * 50)
    
    # Run different analysis scenarios
    analyze_news_articles()
    analyze_product_reviews()
    document_search_demo()
    
    print("\n" + "=" * 50)
    print("Text analysis demo completed!")

if __name__ == "__main__":
    main()