#!/usr/bin/env python
"""Video QA scoring with the GPT-3.5-turbo judge (Video-ChatGPT protocol used by Video-LLaVA).

The system/user prompts and the score parsing follow Video-LLaVA's
`videollava/eval/video/eval_video_qa.py`. This version calls the API in parallel,
re-matches the reference answer by question_id (--gt-file), and can resume.

Accuracy = yes / (yes + no); the average score is also reported.
API key: environment variable OPENAI_API_KEY or --key-file.

    python video_llava/eval_gpt_judge.py \
        --pred results/Video-LLaVA-7B/MSVD/stpruner_136/merge.jsonl \
        --gt-file ./playground/data/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json \
        --out results/Video-LLaVA-7B/MSVD/stpruner_136/judge_gpt.json --workers 8
"""
import os, sys, json, ast, re, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed


SYS_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative "
    "outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if "
    "they match meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

def user_prompt(q, a, pred):
    return (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {q}\n"
        f"Correct Answer: {a}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer "
        "value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys "
        "'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of "
        "'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
    )


def parse_judge(text):
    text = (text or "").strip()
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        try:
            d = ast.literal_eval(m.group(0))
            p = str(d.get("pred", "")).strip().lower()
            if p in ("yes", "no"):
                return p, float(d.get("score", 0))
        except Exception:
            pass
    mp = re.search(r"['\"]?pred['\"]?\s*:\s*['\"]?(yes|no)['\"]?", text, re.I)
    ms = re.search(r"['\"]?score['\"]?\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if mp:
        s = float(ms.group(1)) if ms else (3.0 if mp.group(1).lower() == "yes" else 0.0)
        return mp.group(1).lower(), s
    low = text.lower()
    if "yes" in low[:30]: return "yes", 3.0
    if "no" in low[:30]:  return "no", 0.0
    return None, 0.0


def get_key(args):
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return k
    if args.key_file and os.path.isfile(args.key_file):
        return open(args.key_file).read().strip()
    sys.exit("Set OPENAI_API_KEY or pass --key-file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-file", default="", help="test_a.json; re-match reference answers by question_id (recommended)")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N samples (0: all)")
    ap.add_argument("--workers", type=int, default=8, help="number of parallel API calls")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--key-file", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred) if l.strip()]
    if args.gt_file:
        gt = json.load(open(args.gt_file))
        gtmap = {x["question_id"]: x["answer"] for x in gt}
        fixed = miss = 0
        for r in rows:
            if r["id"] in gtmap:
                if r["answer"] != gtmap[r["id"]]: fixed += 1
                r["answer"] = gtmap[r["id"]]
            else: miss += 1
        print(f"[gt-remap] re-matched answers (changed {fixed}, missing {miss})")
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[judge] {len(rows)} samples  model={args.model}  workers={args.workers}")

    from openai import OpenAI
    client = OpenAI(api_key=get_key(args))

    # Rate-limit state: remaining requests from the response headers + daily-limit flag
    import threading as _th
    rl = {"remaining": None, "rpd_dead": False}
    rl_lock = _th.Lock()

    def judge_one(r):
        # Stop calling the API once the daily request limit is exhausted
        with rl_lock:
            if rl["rpd_dead"]:
                return r, None, 0.0, "SKIP:RPD_DEAD"
        for attempt in range(4):
            try:
                # with_raw_response gives access to the rate-limit headers
                raw = client.chat.completions.with_raw_response.create(
                    model=args.model, temperature=0.0, max_tokens=32,
                    messages=[{"role": "system", "content": SYS_PROMPT},
                              {"role": "user", "content": user_prompt(r["question"], r["answer"], r["pred"])}])
                rem = raw.headers.get("x-ratelimit-remaining-requests")
                if rem is not None:
                    with rl_lock:
                        rl["remaining"] = int(rem)
                comp = raw.parse()
                txt = comp.choices[0].message.content
                p, s = parse_judge(txt)
                return r, p, s, txt[:80]
            except Exception as e:
                es = str(e)
                # Daily limit (RPD) exhausted: skip the remaining samples, resume later
                if "requests per day" in es or "(RPD)" in es:
                    with rl_lock:
                        rl["rpd_dead"] = True
                    return r, None, 0.0, "ERR:RPD_EXHAUSTED"
                # Transient errors (e.g. 429 per-minute limit): retry with backoff
                if attempt == 3:
                    return r, None, 0.0, f"ERR:{es}"[:80]
                time.sleep(2 ** attempt + 1)

    # Resume: skip ids that already have a valid judgment in the progress jsonl
    import threading
    jsonl_path = args.out + "l" if args.out.endswith(".json") else args.out + ".jsonl"
    done_ids = set()
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("judge_pred") in ("yes", "no"):
                        done_ids.add(d["id"])
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} judged samples loaded from {jsonl_path}")
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"[judge] {len(todo)} remaining (done {len(done_ids)}) -> {jsonl_path}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lock = threading.Lock()
    fout = open(jsonl_path, "a")
    done_cnt = [0]

    def judge_and_write(r):
        r2, p, s, raw = judge_one(r)
        # Skipped samples are not written, so they are judged on the next run
        if raw in ("SKIP:RPD_DEAD", "ERR:RPD_EXHAUSTED"):
            return p
        rec = {"id": r2["id"], "question": r2["question"], "answer": r2["answer"],
               "pred": r2["pred"], "judge_pred": p, "judge_score": s, "raw": raw}
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush(); os.fsync(fout.fileno())
            done_cnt[0] += 1
            if done_cnt[0] % 100 == 0:
                print(f"  {done_cnt[0]}/{len(todo)} judged (remaining requests: {rl['remaining']})", flush=True)
        return p

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(judge_and_write, todo))
    fout.close()
    if rl["rpd_dead"]:
        print(f"[stopped] daily request limit reached after {done_cnt[0]} samples; rerun later to resume")

    # Final summary over the progress jsonl, one record per id (valid judgments first)
    by_id = {}
    with open(jsonl_path) as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            prev = by_id.get(d["id"])
            if prev is None or (prev["judge_pred"] not in ("yes", "no") and d["judge_pred"] in ("yes", "no")):
                by_id[d["id"]] = d
    results = list(by_id.values())
    yes = sum(1 for d in results if d["judge_pred"] == "yes")
    no  = sum(1 for d in results if d["judge_pred"] == "no")
    bad = len(results) - yes - no
    score_sum = sum(d["judge_score"] for d in results if d["judge_pred"] in ("yes", "no"))
    n_valid = yes + no
    summary = {"pred_path": args.pred, "n": len(rows), "n_valid": n_valid, "n_unparsed": bad,
               "accuracy": yes / max(n_valid, 1) * 100, "avg_score": score_sum / max(n_valid, 1),
               "judge": args.model, "n_judged": len(results)}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)
    print(f"\n=== {os.path.basename(os.path.dirname(args.pred))} ({args.model}) ===")
    print(f"Accuracy = {summary['accuracy']:.2f}%   avg score = {summary['avg_score']:.3f}   "
          f"(valid {n_valid}/{len(rows)}, unparsed {bad})")
    print(f"→ {args.out}  (+ {jsonl_path})")


if __name__ == "__main__":
    main()
