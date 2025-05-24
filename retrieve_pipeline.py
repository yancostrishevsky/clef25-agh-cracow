import os
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from pathlib import Path
import shutil
import subprocess
from sentence_transformers import SentenceTransformer, CrossEncoder
from pyserini.search import SimpleSearcher
from ir_datasets_longeval import load

DENSE_MODEL = 'malteos/scincl'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

BATCH_SIZE = 16
TOP_K_BM25 = 100
TOP_K_DENSE = 100
TOP_K_RERANK = 100

def export_docs_to_jsonl(snapshot, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'docs.jsonl')
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping export.")
        return out_path
    print("Exporting documents to JSONL for BM25 indexing...")
    with open(out_path, 'w', encoding='utf-8') as f:
        for doc in tqdm(snapshot.docs_iter(), desc="Export docs"):
            contents = ((doc.title or '') + ' ' + (doc.abstract or '')).strip()
            json.dump({'id': doc.doc_id, 'contents': contents}, f, ensure_ascii=False)
            f.write('\n')
    return out_path


def build_bm25_index(doc_jsonl_path, bm25_index_dir):
    if os.path.exists(bm25_index_dir) and os.listdir(bm25_index_dir):
        print(f"BM25 index already exists at {bm25_index_dir}, skipping indexing.")
        return
    print("Building BM25 index using Pyserini...")
    subprocess.run([
        "python", "-m", "pyserini.index",
        "-collection", "JsonCollection",
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "8",
        "-input", os.path.dirname(doc_jsonl_path),
        "-index", bm25_index_dir,
        "-storePositions", "-storeDocvectors", "-storeRaw"
    ], check=True)

def get_documents(jsonl_dir):
    doc_ids = []
    doc_texts = []
    jsonl_path = os.path.join(jsonl_dir, 'docs.jsonl')
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            doc_id = str(d['id'])
            text = ((d.get('title') or '') + ' [SEP] ' + (d.get('abstract') or '')).strip()
            doc_ids.append(doc_id)
            doc_texts.append(text)
    return doc_ids, doc_texts

def get_queries(snapshot):
    return pd.DataFrame([
        {'qid': q.query_id, 'query': q.text.strip()}
        for q in snapshot.queries_iter()
    ])

def compute_dense_embeddings(model, doc_texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Dense embeddings"):
        batch = doc_texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False, device=device)
        embeddings.append(emb)
    return np.vstack(embeddings)

def bm25_searcher(index_dir):
    return SimpleSearcher(index_dir)

def bm25_top_k(searcher, query, k):
    hits = searcher.search(query, k)
    return [(hit.docid, hit.score) for hit in hits]

def dense_top_k(query_encoder, query, doc_emb, doc_ids, top_k=TOP_K_DENSE):
    q_emb = query_encoder.encode("query: "+query.strip().lower(), convert_to_tensor=True, device=device)
    doc_emb_tensor = torch.tensor(doc_emb).to(device)
    scores = torch.matmul(doc_emb_tensor, q_emb)
    topk = torch.topk(scores, k=top_k)
    return [(doc_ids[idx], scores[idx].item()) for idx in topk.indices.cpu().numpy()]

def rerank_with_crossencoder(rerank_model, query, doc_texts, batch_size=16):
    pairs = [[query, doc] for doc in doc_texts]
    scores = rerank_model.predict(pairs, batch_size=batch_size)
    return scores

def main(dataset_id, output_dir, index_dir):
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load(dataset_id)
    snapshots = dataset.get_datasets() or [dataset]
    for snapshot in snapshots:
        snapshot_id = snapshot.get_snapshot() if hasattr(snapshot, "get_snapshot") else "default"
        print(f"\n--- Processing snapshot: {snapshot_id} ---")

        jsonl_dir = os.path.join("documents_jsonl", snapshot_id)
        doc_jsonl_path = export_docs_to_jsonl(snapshot, jsonl_dir)

        bm25_index_dir = os.path.join("bm25_index", snapshot_id)
        build_bm25_index(doc_jsonl_path, bm25_index_dir)

        doc_ids, doc_texts = get_documents(jsonl_dir)

        emb_path = os.path.join(output_dir, snapshot_id, 'doc_embeddings.npy')
        ids_path = os.path.join(output_dir, snapshot_id, 'doc_ids.npy')
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        dense_model = SentenceTransformer(DENSE_MODEL, device=device)
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            print("Loading precomputed document embeddings...")
            doc_emb = np.load(emb_path)
            doc_ids_arr = np.load(ids_path)
        else:
            print("Generating document embeddings...")
            doc_emb = compute_dense_embeddings(dense_model, doc_texts)
            doc_ids_arr = np.array(doc_ids)
            np.save(emb_path, doc_emb)
            np.save(ids_path, doc_ids_arr)

        queries_df = get_queries(snapshot)

        bm25 = bm25_searcher(bm25_index_dir)

        print("Loading cross-encoder:", RERANK_MODEL)
        rerank_model = CrossEncoder(RERANK_MODEL, device=device)

        output_path = os.path.join(output_dir, snapshot_id, 'run.txt.gz')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("Searching and reranking...")
        with gzip.open(output_path, 'wt', encoding='utf-8') as outf:
            for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Queries"):
                qid, query = row['qid'], row['query']

                bm25_candidates = bm25_top_k(bm25, query, TOP_K_BM25)
                dense_candidates = dense_top_k(dense_model, query, doc_emb, doc_ids_arr, TOP_K_DENSE)

                docid_set = set()
                candidates = []
                for did, score in bm25_candidates + dense_candidates:
                    if did not in docid_set:
                        docid_set.add(did)
                        candidates.append((did, score))
                docs_for_rerank = []
                didx_map = {}
                for i, (did, _) in enumerate(candidates[:TOP_K_RERANK]):
                    idx = np.where(doc_ids_arr == did)[0]
                    if len(idx) == 0:
                        docs_for_rerank.append("")
                    else:
                        docs_for_rerank.append(doc_texts[idx[0]])
                    didx_map[i] = did

                rerank_scores = rerank_with_crossencoder(rerank_model, query, docs_for_rerank)
                ranked = sorted(zip(didx_map.values(), rerank_scores), key=lambda x: x[1], reverse=True)
                for rank, (docid, score) in enumerate(ranked):
                    outf.write(f"{qid} Q0 {docid} {rank+1} {score} scincl-bm25-hybrid\n")

        print(f"saved to: {output_path}")

if __name__ == '__main__':
    import click
    @click.command()
    @click.option('--dataset', required=True, type=str, help='IR dataset identifier')
    @click.option('--output', required=True, type=click.Path(), help='Output directory')
    @click.option('--index', required=True, type=click.Path(), help='Index directory (unused, for compatibility)')
    def run(dataset, output, index):
        main(dataset, output, index)
    run()
