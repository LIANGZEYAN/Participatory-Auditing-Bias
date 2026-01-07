# Submission ID xxxxx
Implementation of "Submission ID xxx" (Anonymous Submission).

# IPS-based Debiasing for ColBERT Rankings (Version 1)

This code implements an IPS-inspired debiasing method, combined with semantic similarity adjustment, to enhance ranking quality.

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
python ipssim_version_1.py <input_file> <top_k> <alpha>
```

**Arguments:**
- `input_file`: Path to your ColBERT ranking CSV file (default: `colbert_rankings_default.csv`)
- `top_k`: Number of top documents for semantic similarity calculation (default: 5)
- `alpha`: Debiasing strength parameter (default: 1.0)

**Optional flag:**
- `--output`: Output file path (default: `debiased_rankings_version1.csv`)

### Examples:

```bash
# Use your own ColBERT rankings with default parameters
python ipssim_version_1.py my_rankings.csv

# Customise top-k to 10
python ipssim_version_1.py my_rankings.csv 10

# Customise both top-k and alpha
python ipssim_version_1.py my_rankings.csv 10 0.8

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
qid, docno, score, text
1,doc1,0.85, "This is document text..."
1,doc2,0.72, "Another document text..."
2,doc3,0.91, "Third document..."
```

## Output Format

The output CSV contains:
- `qid`: Query ID
- `docno`: Document ID
- `score`: Original ColBERT score
- `normalized_score`: Min-max normalised score (within each query)
- `semantic_sim`: Average semantic similarity with top-k documents
- `unbiased_score`: Final debiased score
- `unbiased_rank`: New ranking based on unbiased score
- `text`: Document text

## Method Overview

1. **Deduplication**: Removes duplicate documents within each query based on normalised text
2. **Top-K Selection**: Identifies top-k highest-scoring documents by ColBERT score for each query
3. **Semantic Similarity**: Computes cosine similarity between each document and the top-k documents using SimCSE embeddings
4. **Debiasing**: Combines normalised ColBERT score and semantic similarity:
   ```
   unbiased_score = normalized_score / alpha + semantic_sim
   ```
5. **Re-ranking**: Sorts documents by unbiased score to produce final rankings

## Default Dataset

If you don't have your own ColBERT rankings, you can use our provided default dataset `colbert_rankings_default.csv`.
