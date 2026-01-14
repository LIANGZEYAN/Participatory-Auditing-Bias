# Semantic-Based Reranking for ColBERT Rankings (Version 1)

This code implements semantic similarity-based reranking to improve ColBERT ranking quality without requiring click data.

## Requirements

```bash
pip install numpy pandas torch transformers
```

## Usage

### Basic usage (with defaults):
```bash
python ipssim_version_1.py
```
This will use the default ColBERT ranking file `colbert_rankings_default.csv` with top-k=5 and alpha=1.0.

### Custom parameters:
```bash
python ipssim_version_1.py <input_file> <top_k> <alpha_coef>
```

**Arguments:**
- `input_file`: Path to your ColBERT ranking CSV file (default: `colbert_rankings_default.csv`)
- `top_k`: Number of top documents for semantic similarity calculation (default: 5)
- `alpha_coef`: Weight coefficient for semantic similarity (default: 1.0)

**Optional flags:**
- `--output`: Output file path (default: `debiased_rankings_version1.csv`)

### Examples:

```bash
# Use your own ColBERT rankings with default parameters
python ipssim_version_1.py my_rankings.csv

# Customize top-k to 10
python ipssim_version_1.py my_rankings.csv 10

# Customize both top-k and alpha_coef
python ipssim_version_1.py my_rankings.csv 10 2.5

# Specify custom output file
python ipssim_version_1.py my_rankings.csv 5 1.0 --output my_output.csv
```

## Input Format

Your ColBERT ranking CSV file must contain the following columns:
- `qid`: Query ID
- `docno`: Document ID
- `score`: ColBERT relevance score
- `text`: Document text content

Example:
```csv
qid,docno,score,text
1,doc1,0.85,"This is document text..."
1,doc2,0.72,"Another document text..."
2,doc3,0.91,"Third document..."
```

## Output Format

The output CSV contains:
- `qid`: Query ID
- `docno`: Document ID
- `score`: Original ColBERT score
- `normalized_score`: Min-max normalized score (within each query)
- `semantic_sim`: Average semantic similarity with top-k documents
- `unbiased_score`: Final adjusted score
- `unbiased_rank`: New ranking based on unbiased score
- `text`: Document text

## Method Overview

This is the **SBR (Semantic-Based Reranking)** approach for scenarios without click data:

1. **Deduplication**: Removes duplicate documents within each query based on normalized text
2. **Top-K Selection**: Identifies top-k highest-scoring documents by ColBERT score for each query
3. **Semantic Similarity**: Computes cosine similarity between each document and the top-k documents using SimCSE embeddings
4. **Score Adjustment**: Adjusts document scores using semantic similarity:
   ```
   unbiased_score = normalized_score × (1 + alpha_coef × semantic_sim)
   ```
5. **Re-ranking**: Sorts documents by unbiased score to produce final rankings

## Formula Explanation

The core formula is:
```
Score_SBR(d) = Score_ColBERT(d) × (1 + α × AvgSim(d, D_top))
```

Where:
- `Score_ColBERT(d)`: Normalized ColBERT score for document d
- `AvgSim(d, D_top)`: Average semantic similarity between document d and top-k documents
- `α` (alpha_coef): Weight parameter controlling the influence of semantic similarity

**Effect**:
- When `α = 0`: Original ColBERT ranking (no adjustment)
- When `α > 0`: Documents semantically similar to top-k get boosted
- Higher `α`: Stronger emphasis on semantic diversity

## Parameters Explained

- **top_k**: Defines the reference set of top documents for similarity calculation. Typical values: 5-10.
- **alpha_coef**: Controls how much semantic similarity affects the final score. Typical values: 0.5-2.5.

## Default Dataset

If you don't have your own ColBERT rankings, you can use our provided default dataset `colbert_rankings_default.csv`.

## Citation

[Paper citation information will be added after review]
