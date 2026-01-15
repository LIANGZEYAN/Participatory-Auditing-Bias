# SBR (Semantic-Based Reranking) for ColBERT Rankings

This repository implements the SBR (Semantic-Based Reranking) method to improve ColBERT ranking quality through semantic similarity adjustment, followed by strategic document selection for evaluation.

## Requirements

```bash
pip install numpy pandas torch transformers scipy
```

## Workflow

The complete workflow consists of two steps:

### Step 1: Generate SBR Rankings
Apply semantic-based reranking to ColBERT results.

### Step 2: Strategic Document Selection
Merge ColBERT and SBR rankings to select documents for evaluation.

---

## Step 1: SBR Reranking (`ipssim_version_1.py`)

Applies semantic similarity adjustment to ColBERT rankings without requiring click data.

### Usage

```bash
# Basic usage with defaults
python ipssim_version_1.py

# Custom parameters
python ipssim_version_1.py <input_file> <top_k> <alpha_coef> --output <output_file>
```

**Parameters:**
- `input_file`: ColBERT ranking CSV (default: `colbert_rankings_default.csv`)
- `top_k`: Number of top documents for similarity calculation (default: 5)
- `alpha_coef`: Semantic similarity weight (default: 1.0)
- `--output`: Output file path (default: `debiased_rankings_version1.csv`)

### Input Format

CSV with columns: `qid, docno, score, text`

### Output Format

CSV with columns: `qid, docno, score, normalized_score, semantic_sim, sbr_score, sbr_rank, text`

### Method

1. Remove duplicate documents within each query
2. Identify top-k documents by ColBERT score
3. Compute semantic similarity using SimCSE embeddings
4. Apply SBR formula: `sbr_score = normalized_score × (1 + alpha × semantic_sim)`
5. Re-rank documents by SBR score

---

## Step 2: Strategic Selection (`strategy_merge_version1.py`)

Merges ColBERT and SBR rankings to select documents for evaluation.

### Usage

```bash
# Basic usage with defaults
python strategy_merge_version1.py

# Custom files
python strategy_merge_version1.py <colbert_file> <sbr_file> --top_k <k> --output <output_file>
```

**Parameters:**
- `colbert_file`: ColBERT ranking CSV (default: `colbert_rankings_original.csv`)
- `sbr_file`: SBR ranking CSV from Step 1 (default: `sbr_rankings_version1.csv`)
- `--top_k`: Documents to select from each ranking (default: 4)
- `--qrels`: Optional qrels file for relevance judgments
- `--output`: Output file path (default: `strategic_selection_results_version1.csv`)

### Input Format

**ColBERT ranking:** `qid, query, docno, text, colbert_rank`  
**SBR ranking:** `qid, query, docno, text, sbr_rank`

### Output Format

CSV with: `qid, query, docno, text, colbert_rank, sbr_rank, semantic_sim, source, from, selected_in_turn, label`

### Selection Strategy

For each query, selects **9 documents** (suitable for 3×3 grid presentation):

1. **Top-4 from ColBERT**: Documents ranked highest by ColBERT
2. **Top-4 from SBR**: Documents ranked highest by SBR (excluding duplicates)
3. **1 Easy Negative**: Document with lowest semantic similarity and label=0

This creates a balanced set for human evaluation comparing ColBERT vs SBR rankings.

---

## Quick Start Example

```bash
# Step 1: Generate SBR rankings
python ipssim_version_1.py colbert_results.csv 5 1.0 --output sbr_results.csv

# Step 2: Select documents for evaluation
python strategy_merge_version1.py colbert_results.csv sbr_results.csv --top_k 4 --output evaluation_set.csv
```

## Key Formula

The SBR score formula (from paper):

```
Score_SBR(d) = Score_ColBERT(d) × (1 + α × AvgSim(d, D_top))
```

Where:
- `Score_ColBERT(d)`: Normalized ColBERT score
- `AvgSim(d, D_top)`: Average semantic similarity with top-k documents
- `α` (alpha_coef): Weight parameter controlling diversity

## Notes

- **Step 1** requires only ColBERT rankings (no click data needed)
- **Step 2** creates dummy qrels if none provided (assumes top-10 are relevant)
- Both scripts use SimCSE-BERT for semantic similarity computation
- GPU acceleration available if CUDA is installed

## Citation

[Paper citation information will be added after review]
