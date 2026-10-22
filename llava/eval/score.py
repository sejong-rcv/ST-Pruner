"""
Score an answers file produced by `llava.eval.infer`.

    python -m llava.eval.score --benchmark pope \
        --result-file results/llava-v1.5-7b/pope/stpruner_64.jsonl

The report is printed and written next to the answers file as `*_result.txt`.
VQAv2 has no public test-dev answers: convert it with
`scripts/convert_vqav2_for_submission.py` and submit to EvalAI.
"""

import argparse

from llava.eval.scorers import SCORERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./playground/data/eval")
    args = parser.parse_args()

    if args.benchmark not in SCORERS:
        print(f"[{args.benchmark}] no local scorer: use the submission converter in scripts/ (see EVAL.md).")
        return
    _, report = SCORERS[args.benchmark](args.result_file, args.data_dir)
    print(report)
    with open(args.result_file.replace(".jsonl", "_result.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
