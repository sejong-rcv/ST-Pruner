"""
Convert VizWiz test answers (jsonl) into the EvalAI submission format.

    python scripts/convert_vizwiz_for_submission.py \
        --result-file results/llava-v1.5-7b/vizwiz/stpruner_64.jsonl \
        --upload-file results/llava-v1.5-7b/vizwiz/upload/stpruner_64.json
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
    parser.add_argument("--test-split", default="./playground/data/eval/vizwiz/llava_test.jsonl")
    args = parser.parse_args()

    processor = EvalAIAnswerProcessor()
    results = {}
    for line in open(args.result_file):
        item = json.loads(line)
        results[item["question_id"]] = item["text"]

    test_split = [json.loads(line) for line in open(args.test_split)]
    answers = []
    for x in test_split:
        assert x["question_id"] in results, f"missing answer for question {x['question_id']}"
        answers.append({"image": x["image"], "answer": processor(results[x["question_id"]])})

    os.makedirs(os.path.dirname(os.path.abspath(args.upload_file)), exist_ok=True)
    with open(args.upload_file, "w") as f:
        json.dump(answers, f)
    print(f"Saved {args.upload_file} ({len(answers)} answers)")


if __name__ == "__main__":
    main()
