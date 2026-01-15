import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine


def strategic_document_selection(biased_ranking_df, debiased_ranking_df, qid, qrels_df=None, top_k=4, 
                                model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    """
    Strategic document selection: picks top documents from both rankings and adds a dissimilar negative example.
    
    Selection strategy:
    1. Select top-k documents from biased ranking
    2. Select top-k documents from debiased ranking (excluding duplicates)
    3. Select 1 easy negative document (lowest semantic similarity, label=0)
    
    Total: 2*top_k + 1 documents (default: 4 + 4 + 1 = 9 documents)
    """
    # Filter data for the specified qid
    biased_df = biased_ranking_df[biased_ranking_df['qid'] == qid].copy()
    debiased_df = debiased_ranking_df[debiased_ranking_df['qid'] == qid].copy()
    
    if biased_df.empty or debiased_df.empty:
        raise ValueError(f"qid={qid} is missing in one/both rankings")
    
    query_text = biased_df['query'].iloc[0]
    
    # Sort by respective rank columns
    biased_sorted = biased_df.sort_values('biased_rank').reset_index(drop=True)
    debiased_sorted = debiased_df.sort_values('unbiased_rank').reset_index(drop=True)
    
    selected_docs = []
    selected_docnos = set()
    
    # Step 1: Select top-k documents from biased ranking
    for i in range(min(top_k, len(biased_sorted))):
        doc = biased_sorted.iloc[i]
        
        # Find debiased rank if exists
        debiased_rank = np.nan
        matching_debiased = debiased_df[debiased_df['docno'] == doc['docno']]
        if not matching_debiased.empty:
            debiased_rank = matching_debiased.iloc[0]['unbiased_rank']
        
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': doc['docno'],
            'text': doc['text'],
            'biased_rank': doc['biased_rank'],
            'debiased_rank': debiased_rank,
            'source': "top from biased",
            'from': "biased",
            'selected_in_turn': len(selected_docs) + 1
        })
        selected_docnos.add(doc['docno'])
    
    # Step 2: Select top-k documents from debiased ranking (skip duplicates)
    debiased_idx = 0
    selected_from_debiased = 0
    
    while selected_from_debiased < top_k and debiased_idx < len(debiased_sorted):
        doc = debiased_sorted.iloc[debiased_idx]
        debiased_idx += 1
        
        if doc['docno'] in selected_docnos:
            continue
        
        # Find biased rank if exists
        biased_rank = np.nan
        matching_biased = biased_df[biased_df['docno'] == doc['docno']]
        if not matching_biased.empty:
            biased_rank = matching_biased.iloc[0]['biased_rank']
            
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': doc['docno'],
            'text': doc['text'],
            'biased_rank': biased_rank,
            'debiased_rank': doc['unbiased_rank'],
            'source': "top from debiased",
            'from': "debiased",
            'selected_in_turn': len(selected_docs) + 1
        })
        selected_docnos.add(doc['docno'])
        selected_from_debiased += 1
    
    # If no remaining documents, return what we have
    remaining_docs = debiased_sorted[~debiased_sorted['docno'].isin(selected_docnos)].copy()
    if remaining_docs.empty:
        return pd.DataFrame(selected_docs)
    
    # Step 3: Compute semantic similarity for easy negative selection
    temp_df = pd.DataFrame(selected_docs)
    all_docs_df = pd.concat([remaining_docs, temp_df]).reset_index(drop=True)
    
    try:
        docs_with_sim = add_semantic_similarity_hf(
            all_docs_df,
            qid_col="qid",
            text_col="text",
            rank_col="debiased_rank" if "debiased_rank" in all_docs_df.columns else "unbiased_rank",
            top_k=min(len(temp_df), 5),
            model_name=model_name,
            batch_size=32
        )
    except Exception as e:
        print(f"Error computing semantic similarity: {str(e)}")
        return pd.DataFrame(selected_docs)
    
    # Step 4: Find easy negative from remaining documents
    neg_candidates_df = docs_with_sim[~docs_with_sim['docno'].isin(selected_docnos)].copy()
    
    # Filter for label=0 documents if qrels provided
    if qrels_df is not None and not qrels_df.empty and 'qid' in qrels_df.columns:
        query_qrels = qrels_df[qrels_df['qid'] == qid]
        
        if not query_qrels.empty:
            neg_candidates_df = neg_candidates_df.merge(
                query_qrels[['docno', 'label']], 
                on='docno',
                how='left'
            )
            neg_candidates_df['label'] = neg_candidates_df['label'].fillna(0)
            neg_candidates_rel0 = neg_candidates_df[neg_candidates_df['label'] == 0]
            
            if not neg_candidates_rel0.empty:
                neg_candidates_df = neg_candidates_rel0
    
    # Select document with lowest semantic similarity
    if not neg_candidates_df.empty:
        neg_candidates_df = neg_candidates_df.sort_values('semantic_sim')
        neg_doc = neg_candidates_df.iloc[0]
        
        # Find biased rank
        biased_rank = np.nan
        matching_biased = biased_df[biased_df['docno'] == neg_doc['docno']]
        if not matching_biased.empty:
            biased_rank = matching_biased.iloc[0]['biased_rank']
        
        selected_docs.append({
            'qid': qid,
            'query': query_text,
            'docno': neg_doc['docno'],
            'text': neg_doc['text'],
            'biased_rank': biased_rank,
            'debiased_rank': neg_doc['debiased_rank'] if 'debiased_rank' in neg_doc else neg_doc['unbiased_rank'],
            'semantic_sim': neg_doc['semantic_sim'],
            'source': "easy negative",
            'from': "debiased",
            'selected_in_turn': len(selected_docs) + 1,
            'label': neg_doc['label'] if 'label' in neg_doc else np.nan
        })
    
    return pd.DataFrame(selected_docs)


def add_semantic_similarity_hf(
    df,
    qid_col="qid",
    text_col="text",
    rank_col="biased_rank",
    top_k=5,
    model_name="princeton-nlp/sup-simcse-bert-base-uncased",
    batch_size=32
):
    """
    Compute semantic similarity between each document and top-k reference documents.
    
    For each query:
    1. Generate embeddings for all documents using SimCSE
    2. Select top-k documents as reference set (by rank_col)
    3. For each document, compute average cosine similarity with reference set
       - If document is in top-k: exclude itself from calculation
       - If document is not in top-k: compute with all top-k documents
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer and model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Compute embeddings for all documents in batches
    print("Computing embeddings for all documents...")
    texts = df[text_col].tolist()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            embeddings.extend(batch_embeddings)
    df["embedding"] = list(embeddings)
    
    def cosine_sim(a, b):
        return 1 - cosine(a, b)
    
    semantic_sim_dict = {}
    
    # Group by qid and calculate semantic similarity
    for qid_val, group in df.groupby(qid_col):
        valid_rank_rows = group[group[rank_col].notna()]
        if valid_rank_rows.empty:
            group_sorted = group
        else:
            group_sorted = group.sort_values(by=rank_col, ascending=True)
            
        top_k_group = group_sorted.head(top_k)
        top_k_indices = set(top_k_group.index)
        top_k_embs = np.stack(top_k_group["embedding"].values)
        
        for idx, row in group.iterrows():
            doc_emb = row["embedding"]
            
            if idx in top_k_indices:
                # Exclude itself from reference set
                mask = [i != idx for i in top_k_group.index]
                if sum(mask) == 0:
                    avg_sim = 0.0
                else:
                    ref_embs = np.stack(top_k_group.iloc[np.where(mask)[0]]["embedding"].values)
                    sims = [cosine_sim(doc_emb, ref_emb) for ref_emb in ref_embs]
                    avg_sim = np.mean(sims)
            else:
                # Calculate similarity with all top-k
                sims = [cosine_sim(doc_emb, top_emb) for top_emb in top_k_embs]
                avg_sim = np.mean(sims)
            
            semantic_sim_dict[idx] = avg_sim
    
    df["semantic_sim"] = df.index.map(semantic_sim_dict)
    print("Semantic similarity computed and stored in 'semantic_sim'.")
    
    return df


def process_all_queries(biased_ranking_df, debiased_ranking_df, qrels_df=None, top_k=4, 
                       model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    """
    Process all query IDs using the strategic document selection approach.
    
    Returns a DataFrame with selected documents for all queries.
    """
    # Validate data format
    biased_required_columns = ['qid', 'query', 'docno', 'text', 'biased_rank']
    debiased_required_columns = ['qid', 'query', 'docno', 'text', 'unbiased_rank']
    
    missing_cols = [col for col in biased_required_columns if col not in biased_ranking_df.columns]
    if missing_cols:
        raise ValueError(f"biased ranking missing columns: {', '.join(missing_cols)}")
        
    missing_cols = [col for col in debiased_required_columns if col not in debiased_ranking_df.columns]
    if missing_cols:
        raise ValueError(f"unbiased ranking missing columns: {', '.join(missing_cols)}")
    
    # Validate or create dummy qrels
    if qrels_df is not None and not qrels_df.empty:
        qrels_required_columns = ['qid', 'docno', 'label']
        missing_cols = [col for col in qrels_required_columns if col not in qrels_df.columns]
        if missing_cols:
            print(f"Warning: qrels missing columns: {', '.join(missing_cols)}. Creating dummy qrels.")
            qrels_df = get_dummy_qrels_data(biased_ranking_df, debiased_ranking_df)
    else:
        print("No qrels provided. Creating dummy qrels data.")
        qrels_df = get_dummy_qrels_data(biased_ranking_df, debiased_ranking_df)
    
    # Get common query IDs
    all_qids = sorted(set(biased_ranking_df['qid'].unique()) & set(debiased_ranking_df['qid'].unique()))
    
    if not all_qids:
        raise ValueError("no same qid in two rankings")
    
    print(f"Found {len(all_qids)} query IDs")
    
    # Process each query
    all_selected_docs = []
    
    for qid in all_qids:
        try:
            selected_docs = strategic_document_selection(
                biased_ranking_df, 
                debiased_ranking_df, 
                qid, 
                qrels_df=qrels_df,
                top_k=top_k,
                model_name=model_name
            )
            all_selected_docs.append(selected_docs)
            print(f"Successfully processed ID {qid}, selected {len(selected_docs)} documents")
        except Exception as e:
            print(f"Error processing ID {qid}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not all_selected_docs:
        raise ValueError("no successful processing for any qid")
    
    # Merge all results
    result_df = pd.concat(all_selected_docs, ignore_index=True)
    
    # Print statistics
    from_counts = result_df['from'].value_counts()
    source_counts = result_df['source'].value_counts()
    
    print("\nAll selected documents distribution:")
    print(from_counts)
    print("\nAll selected documents source distribution:")
    print(source_counts)
    
    return result_df


def get_dummy_qrels_data(biased_ranking_df, debiased_ranking_df):
    """
    Create dummy qrels assuming top-10 documents from either ranking are relevant (label=1).
    Others are non-relevant (label=0).
    """
    all_qids = sorted(set(biased_ranking_df['qid'].unique()) & set(debiased_ranking_df['qid'].unique()))
    rows = []
    
    for qid in all_qids:
        biased_docs = biased_ranking_df[biased_ranking_df['qid'] == qid].sort_values('biased_rank')
        debiased_docs = debiased_ranking_df[debiased_ranking_df['qid'] == qid].sort_values('unbiased_rank')
        
        # Top 10 from either ranking are considered relevant
        top_biased_docnos = set(biased_docs.head(10)['docno'])
        top_debiased_docnos = set(debiased_docs.head(10)['docno'])
        relevant_docnos = top_biased_docnos.union(top_debiased_docnos)
        
        all_docnos = set(biased_docs['docno']).union(set(debiased_docs['docno']))
        
        for docno in all_docnos:
            rows.append({
                'qid': qid,
                'docno': docno,
                'label': 1 if docno in relevant_docnos else 0
            })
    
    return pd.DataFrame(rows)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Strategic document selection merging biased and debiased rankings"
    )
    parser.add_argument(
        "biased_file",
        nargs="?",
        default="colbert_rankings_original.csv",
        help="Path to biased (original ColBERT) ranking CSV. Columns: qid, query, docno, text, biased_rank"
    )
    parser.add_argument(
        "debiased_file",
        nargs="?",
        default="debiased_rankings_version1.csv",
        help="Path to debiased ranking CSV. Columns: qid, query, docno, text, unbiased_rank"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Number of top documents to select from each ranking. Default: 4"
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help="Path to qrels CSV file (optional). Columns: qid, docno, label"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="strategic_selection_results_version1.csv",
        help="Output file path. Default: strategic_selection_results_version1.csv"
    )
    
    args = parser.parse_args()
    
    print("Loading ranking data...")
    
    # Load biased ranking
    biased_df = pd.read_csv(args.biased_file)
    required_cols = ['qid', 'query', 'docno', 'text', 'biased_rank']
    if not all(col in biased_df.columns for col in required_cols):
        raise ValueError(f"Biased ranking CSV must contain columns: {required_cols}")
    
    # Load debiased ranking
    debiased_df = pd.read_csv(args.debiased_file)
    required_cols = ['qid', 'query', 'docno', 'text', 'unbiased_rank']
    if not all(col in debiased_df.columns for col in required_cols):
        raise ValueError(f"Debiased ranking CSV must contain columns: {required_cols}")
    
    print(f"Loaded biased ranking: {len(biased_df)} documents for {biased_df['qid'].nunique()} queries")
    print(f"Loaded debiased ranking: {len(debiased_df)} documents for {debiased_df['qid'].nunique()} queries")
    
    # Load qrels if provided
    qrels_df = None
    if args.qrels:
        qrels_df = pd.read_csv(args.qrels)
        print(f"Loaded qrels: {len(qrels_df)} judgments")
    
    # Process all queries
    print(f"\nProcessing with top_k={args.top_k}...")
    result_df = process_all_queries(
        biased_df, 
        debiased_df, 
        qrels_df=qrels_df,
        top_k=args.top_k,
        model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    )
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    result_df.to_csv(args.output, index=False)
    
    print("\n=== Processing Complete ===")
    print(f"Total selected documents: {len(result_df)}")
    print(f"Queries processed: {result_df['qid'].nunique()}")
    print(f"Average documents per query: {len(result_df) / result_df['qid'].nunique():.1f}")


if __name__ == "__main__":
    main()
