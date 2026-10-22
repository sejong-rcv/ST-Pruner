"""
Build `MME/mme_answers.json` (question_id, question, answer) used by the MME scorer.

The ground truth is taken from the `lmms-lab/MME` dataset on Hugging Face, which
contains the same 2,374 question-answer pairs as the official MME release.

    pip install pyarrow
    python scripts/prepare_mme_answers.py --out ./playground/data/eval/MME/mme_answers.json
"""

import os
import glob
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./playground/data/eval/MME/mme_answers.json")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    path = snapshot_download("lmms-lab/MME", repo_type="dataset", allow_patterns=["data/*.parquet"])
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "data", "*.parquet"))):
        rows += pq.read_table(f, columns=["question_id", "question", "answer"]).to_pylist()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=4)
    print(f"Saved {args.out} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
