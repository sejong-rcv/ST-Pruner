"""
Convert VQAv2 test-dev answers (jsonl) into the EvalAI submission format.

The submission covers the full test2015 question list; questions outside test-dev
are filled with "". Answers are normalized with EvalAIAnswerProcessor.

    python scripts/convert_vqav2_for_submission.py \
        --result-file results/llava-v1.5-7b/vqav2/stpruner_64.jsonl \
        --upload-file results/llava-v1.5-7b/vqav2/upload/stpruner_64.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--upload-file", required=True)
    parser.add_argument("--test-split", default="./playground/data/eval/vqav2/llava_vqav2_mscoco_test2015.jsonl")
    args = parser.parse_args()

    processor = EvalAIAnswerProcessor()
    preds = {}
    for line in open(args.result_file):
        item = json.loads(line)
        preds[item["question_id"]] = item["text"]

    test_split = [json.loads(line) for line in open(args.test_split)]
    answers = [{"question_id": x["question_id"],
                "answer": processor(preds[x["question_id"]]) if x["question_id"] in preds else ""}
               for x in test_split]

    os.makedirs(os.path.dirname(os.path.abspath(args.upload_file)), exist_ok=True)
    with open(args.upload_file, "w") as f:
        json.dump(answers, f)
    print(f"Saved {args.upload_file} ({len(preds)} answered / {len(answers)} questions)")


if __name__ == "__main__":
    main()
