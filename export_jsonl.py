import os
import json
from ir_datasets_longeval import load
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()

dataset = load(args.dataset)
os.makedirs(args.output_dir, exist_ok=True)
jsonl_path = os.path.join(args.output_dir, 'docs.jsonl')

with open(jsonl_path, 'w', encoding='utf-8') as f:
    for doc in tqdm(dataset.docs_iter(), desc="doc export"):
        json.dump({
            'id': doc.doc_id,
            'title': doc.title or "",
            'abstract': doc.abstract or ""
        }, f, ensure_ascii=False)
        f.write('\n')
print(f'Saved: {jsonl_path}')
