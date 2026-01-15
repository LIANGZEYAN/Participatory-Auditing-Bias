# Strategic Document Selection for Ranking Evaluation

This script merges biased (original) and debiased rankings to create a strategic document set for evaluation.

## Requirements

```bash
pip install numpy pandas torch transformers scipy
```

## Usage

### Basic usage (with defaults):
```bash
python strategy_merge_version1.py
```
This will use default files: `colbert_rankings_original.csv` (biased) and `debiased_rankings_version1.csv` (debiased).

### Custom files:
```bash
python strategy_merge_version1.py <biased_file> <debiased_file>
```

**Arguments:**
- `biased_file`: Path to biased (original ColBERT) ranking CSV (default: `colbert_rankings_original.csv`)
- `debiased_file`: Path to debiased ranking CSV (default: `debiased_rankings_version1.csv`)

**Optional flags:**
- `--top_k`: Number of documents to select from each ranking (default: 4)
- `--qrels`: Path to qrels CSV file with relevance judgments (optional)
- `--output`: Output file path (default: `strategic_selection_results_version1.csv`)

### Examples:

```bash
# Use your own ranking files with default parameters
python strategy_merge_version1.py my_biased.csv my_debiased.csv

# Customize top-k to 5
python strategy_merge_version1.py my_biased.csv my_debiased.csv --top_k 5

# Provide qrels and custom output
python strategy_merge_version1.py my_biased.csv my_debiased.csv --qrels my_qrels.csv --output merged.csv
```

## Input Format

### Biased Ranking CSV
Must contain columns:
- `qid`: Query ID
- `query`: Query text
- `docno`: Document ID
- `text`: Document text
- `biased_rank`: Rank position in biased ranking

### Debiased Ranking CSV
Must contain columns:
- `qid`: Query ID
- `query`: Query text
- `docno`: Document ID
- `text`: Document text
- `unbiased_rank`: Rank position in debiased ranking

### Qrels CSV (Optional)
If provided, must contain columns:
- `qid`: Query ID
- `docno`: Document ID
- `label`: Relevance label (0=non-relevant, 1+=relevant)

If not provided, the script creates dummy qrels assuming top-10 documents are relevant.

## Output Format

The output CSV contains selected documents with:
- `qid`: Query ID
- `query`: Query text
- `docno`: Document ID
- `text`: Document text
- `biased_rank`: Rank in biased ranking
- `debiased_rank`: Rank in debiased ranking
- `semantic_sim`: Semantic similarity score (for easy negatives)
- `source`: Selection source ("top from biased", "top from debiased", "easy negative")
- `from`: Ranking source ("biased" or "debiased")
- `selected_in_turn`: Selection order (1-9)
- `label`: Relevance label (if qrels provided)

## Selection Strategy

For each query, the script selects **2×top_k + 1** documents (default: 9 documents):

### Step 1: Top-k from Biased Ranking (default: 4 documents)
- Selects top-k documents ranked highest in the biased (original) ranking
- These represent what the original ranker considers most relevant

### Step 2: Top-k from Debiased Ranking (default: 4 documents)
- Selects top-k documents ranked highest in the debiased ranking
- **Excludes documents already selected in Step 1** (no duplicates)
- These represent documents that were "under-valued" by the biased ranker

### Step 3: Easy Negative (1 document)
- Selects one document with:
  1. **Lowest semantic similarity** to the 8 documents selected in Steps 1-2
  2. **Relevance label = 0** (non-relevant, if qrels provided)
- This serves as a control/negative example for evaluation

### Semantic Similarity Calculation
- Uses SimCSE-BERT embeddings (`princeton-nlp/sup-simcse-bert-base-uncased`)
- For each document, computes average cosine similarity with selected documents
- Documents with lower similarity are more dissimilar/diverse

## Method Overview

1. **Load Rankings**: Load biased and debiased ranking files
2. **Validate Format**: Check required columns exist
3. **Load/Create Qrels**: Load qrels if provided, otherwise create dummy qrels
4. **For Each Query**:
   - Select top-k from biased ranking
   - Select top-k from debiased ranking (excluding duplicates)
   - Compute semantic similarity for remaining documents
   - Select easy negative (lowest similarity + label=0)
5. **Save Results**: Output merged selection to CSV

## Use Case

This strategic selection is designed for:
- **Audit-based evaluation** of ranking systems
- **Human evaluation** studies comparing biased vs debiased rankings
- **Identifying under-valued documents** that debiasing surfaced
- **Controlled experiments** with positive and negative examples

The 3×3 grid layout (9 documents total) is suitable for presenting to human evaluators for preference judgments.

## Citation

[Paper citation information will be added after review]
