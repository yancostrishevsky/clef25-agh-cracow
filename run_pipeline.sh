#!/bin/bash
set -e

DATASET=longeval-sci/2024-11
OUTPUT=output
INDEX=indexes

echo "=== [1/3] Start pipeline ==="
python3 retrieve_pipeline.py --dataset $DATASET --output $OUTPUT --index $INDEX

if [ -f ir-metadata.yml ]; then
  cp ir-metadata.yml $OUTPUT/
fi

echo "=== [3/3] TREC : $OUTPUT/run.txt.gz ==="
