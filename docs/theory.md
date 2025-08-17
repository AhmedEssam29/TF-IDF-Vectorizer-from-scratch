# Mathematical Theory Behind TF-IDF

**Author:** Ahmed Essam Abd Elgwad

## Introduction

Term Frequency-Inverse Document Frequency (TF-IDF) is a fundamental technique in information retrieval and text mining. This document explains the mathematical foundations and intuition behind our implementation.

## Core Mathematical Concepts

### 1. Term Frequency (TF)

Term Frequency measures how frequently a term appears in a document relative to the total number of terms in that document.

Where:
- `t` = term (word)
- `d` = document
- `count(t,d)` = number of times term t appears in document d
- `|d|` = total number of terms in document d

**Intuition:** Common words in a document should have higher weights.

### 2. Inverse Document Frequency (IDF)

IDF measures how rare or common a term is across the entire document collection.

Where:
- `D` = collection of all documents
- `|D|` = total number of documents
- `|{d ∈ D : t ∈ d}|` = number of documents containing term t

**Intuition:** Rare words are more informative than common words.

### 3. TF-IDF Score

The final TF-IDF score combines both measures:

**Intuition:** Words that are frequent in a specific document but rare across the collection are most important.

## Implementation Details

### Logarithmic IDF

We use natural logarithm in our IDF calculation to:
1. Reduce the effect of very large document collections
2. Ensure positive values for terms appearing in multiple documents
3. Provide smooth scaling of importance

### Normalization Considerations

Our implementation uses raw TF-IDF scores without additional normalization. For document similarity tasks, cosine similarity provides implicit L2 normalization.

## Mathematical Examples

### Example 1: Simple Calculation

Documents:
- Doc 1: "cat dog cat"
- Doc 2: "dog bird"

**Step 1: Calculate TF**
- Doc 1: TF(cat) = 2/3, TF(dog) = 1/3
- Doc 2: TF(dog) = 1/2, TF(bird) = 1/2

**Step 2: Calculate IDF**
- IDF(cat) = log(2/1) = 0.693 (appears in 1 document)
- IDF(dog) = log(2/2) = 0.000 (appears in 2 documents)
- IDF(bird) = log(2/1) = 0.693 (appears in 1 document)

**Step 3: Calculate TF-IDF**
- Doc 1: TF-IDF(cat) = (2/3) × 0.693 = 0.462
- Doc 1: TF-IDF(dog) = (1/3) × 0.000 = 0.000
- Doc 2: TF-IDF(dog) = (1/2) × 0.000 = 0.000
- Doc 2: TF-IDF(bird) = (1/2) × 0.693 = 0.347

## Advantages and Limitations

### Advantages
1. **Simple and interpretable:** Easy to understand and implement
2. **Language agnostic:** Works with any language
3. **Established baseline:** Well-studied and widely used
4. **Computationally efficient:** Linear time complexity

### Limitations
1. **Assumes term independence:** Ignores relationships between words
2. **No semantic understanding:** "car" and "automobile" treated as different
3. **Sensitive to document length:** Longer documents may be disadvantaged
4. **No context awareness:** Same word meaning different things in different contexts

## Extensions and Variations

### Sublinear TF Scaling

TF(t,d) = 1 + log(count(t,d))

### L2 Normalization
TF-IDF_normalized(t,d) = TF-IDF(t,d) / ||TF-IDF_vector(d)||_2

### Smooth IDF
IDF(t,D) = log(|D| / (1 + |{d ∈ D : t ∈ d}|)) + 1

## References


1. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval.
3. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval.

---

*This implementation by Ahmed Essam Abd Elgwad provides a complete, transparent TF-IDF system with mathematical accuracy and educational value.*