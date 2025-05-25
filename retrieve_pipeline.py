import os
import gzip
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from ir_datasets_longeval import load

MODEL_DENSE = 'intfloat/e5-base-v2'
MODEL_RERANK = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
BATCH_SIZE = 16
TOP_K = 100

def get_documents(snapshot):
    documents = []
    doc_ids = []
    for doc in snapshot.docs_iter():
        text = (doc.title or '') + ' ' + (doc.abstract or '')
        if text.strip():
            documents.append("passage: " + text.strip().lower())
            doc_ids.append(doc.doc_id)
    return documents, doc_ids

def get_queries(snapshot):
    return pd.DataFrame([
        {'qid': q.query_id, 'query': "query: " + q.text.strip().lower()}
        for q in snapshot.queries_iter()
    ])

def generate_run(snapshot, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.txt.gz"
    if run_path.exists():
        print(f"Skipping existing {run_path}")
        return

    print(f"\nProcessing snapshot: {snapshot.get_snapshot()}")

    embed_path = output_dir / 'doc_embeddings.npy'
    ids_path = output_dir / 'doc_ids.npy'

    dense_model = SentenceTransformer(MODEL_DENSE, device=device)

    if embed_path.exists() and ids_path.exists():
        print("Loading precomputed document embeddings...")
        doc_embeddings = np.load(embed_path)
        doc_ids = np.load(ids_path)
    else:
        print("Encoding documents...")
        documents, doc_ids = get_documents(snapshot)
        doc_embeddings = []
        for i in tqdm(range(0, len(documents), BATCH_SIZE)):
            batch = documents[i:i + BATCH_SIZE]
            emb = dense_model.encode(batch, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=False)
            doc_embeddings.append(emb)
        doc_embeddings = np.vstack(doc_embeddings)
        doc_ids = np.array(doc_ids)
        np.save(embed_path, doc_embeddings)
        np.save(ids_path, doc_ids)
        print("Embeddings saved.")

    queries_df = get_queries(snapshot)
    results = []

    print("Dense retrieval...")
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
        qid = row['qid']
        qtext = row['query']
        qvec = dense_model.encode(qtext, convert_to_tensor=True, device=device)
        scores = util.cos_sim(qvec, torch.tensor(doc_embeddings).to(device))[0]
        topk = torch.topk(scores, k=min(TOP_K, len(doc_embeddings)))
        for rank, (score, idx) in enumerate(zip(topk.values, topk.indices)):
            results.append({'qid': qid, 'docid': doc_ids[idx], 'score': score.item(), 'rank': rank})

    print("Reranking with CrossEncoder...")
    rerank_model = CrossEncoder(MODEL_RERANK, device=device)
    reranked = []

    doc_text_map = {doc.doc_id: (doc.title or '') + ' ' + (doc.abstract or '') for doc in snapshot.docs_iter()}
    query_map = dict(zip(queries_df['qid'], queries_df['query']))

    for qid in queries_df['qid']:
        query = query_map[qid].replace("query: ", "")
        candidates = [r for r in results if r['qid'] == qid]
        pairs = [[query, doc_text_map.get(r['docid'], '')] for r in candidates]
        if not pairs:
            continue
        scores = rerank_model.predict(pairs, batch_size=32)

        sorted_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        for rank, (cand, score) in enumerate(sorted_docs):
            reranked.append({
                'qid': qid,
                'docid': cand['docid'],
                'rank': rank + 1,
                'score': float(score),
                'run_tag': 'e5-reranked'
            })

    print(f"Writing TREC run file to {run_path}")
    with gzip.open(run_path, 'wt') as f:
        for row in reranked:
            f.write(f"{row['qid']} Q0 {row['docid']} {row['rank']} {row['score']} {row['run_tag']}\n")

def main(dataset_id, output_dir, index_dir):
    dataset = load(dataset_id)
    snapshots = dataset.get_datasets() or [dataset]
    for snapshot in snapshots:
        generate_run(snapshot, Path(output_dir) / snapshot.get_snapshot())

    if os.path.exists('ir-metadata.yml'):
        shutil.copy('ir-metadata.yml', Path(output_dir) / 'ir-metadata.yml')

if __name__ == '__main__':
    import click

    @click.command()
    @click.option('--dataset', required=True, type=str, help='IR dataset identifier')
    @click.option('--output', required=True, type=click.Path(), help='Output directory')
    @click.option('--index', required=True, type=click.Path(), help='Index directory (unused but required)')
    def run(dataset, output, index):
        global device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        main(dataset, output, index)

    run()
