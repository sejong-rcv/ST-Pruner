"""
Local scorers. Each follows the official LLaVA-1.5 evaluation logic of its benchmark
and returns (metrics: dict, report: str). See EVAL.md for the required files.
"""

import os
import re
import json
from collections import defaultdict

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator, EvalAIAnswerProcessor


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------
# GQA: exact match on testdev-balanced
# ----------------------------------------------------------------------------

def eval_gqa(result_file, data_dir):
    questions = json.load(open(os.path.join(data_dir, "gqa/testdev_balanced_questions.json")))
    results = _load_jsonl(result_file)
    correct = skipped = 0
    for r in results:
        qid = str(r["question_id"])
        if qid not in questions:
            skipped += 1
            continue
        gt = questions[qid]["answer"].lower().strip()
        pred = r["text"].rstrip('.').lower().strip()
        correct += int(pred == gt)
    total = len(results) - skipped
    acc = 100 * correct / total if total else 0.0
    return {"accuracy": acc}, f"GQA accuracy: {acc:.2f} ({correct}/{total})"


# ----------------------------------------------------------------------------
# POPE: F1 per sampling strategy, averaged
# ----------------------------------------------------------------------------

def _pope_normalize(text):
    if text.find('.') != -1:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        return 'no'
    return 'yes'


def eval_pope(result_file, data_dir):
    ann_dir = os.path.join(data_dir, "pope/coco")
    questions = {q["question_id"]: q for q in _load_jsonl(os.path.join(data_dir, "pope/llava_pope_test.jsonl"))}
    answers = _load_jsonl(result_file)
    metrics, lines, f1s = {}, [], []
    for fname in sorted(os.listdir(ann_dir)):
        if not (fname.startswith("coco_pope_") and fname.endswith(".json")):
            continue
        cat = fname[len("coco_pope_"):-len(".json")]
        cat_answers = [a for a in answers if questions[a["question_id"]]["category"] == cat]
        labels = [a["label"] for a in _load_jsonl(os.path.join(ann_dir, fname))]
        preds = [_pope_normalize(a["text"]) for a in cat_answers]
        tp = fp = tn = fn = 0
        for pred, gt in zip(preds, labels):
            if pred == "yes" and gt == "yes":
                tp += 1
            elif pred == "yes" and gt == "no":
                fp += 1
            elif pred == "no" and gt == "no":
                tn += 1
            elif pred == "no" and gt == "yes":
                fn += 1
        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics[cat] = {"acc": 100 * acc, "f1": 100 * f1}
        f1s.append(f1)
        lines.append(f"  {cat:12s} acc={100 * acc:.2f} f1={100 * f1:.2f}")
    metrics["avg_f1"] = 100 * sum(f1s) / len(f1s)
    return metrics, "\n".join(lines + [f"POPE avg F1: {metrics['avg_f1']:.2f}"])


# ----------------------------------------------------------------------------
# TextVQA: soft accuracy over 10 answers
# ----------------------------------------------------------------------------

def _textvqa_question(prompt):
    if prompt.startswith('OCR tokens: '):
        match = re.search(r"Question: (.*?) Short answer:", prompt, re.DOTALL)
        return match.group(1).lower()
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            return prompt.split('\n')[1].lower()
        return prompt.split('\n')[0].lower()
    return prompt.split('\n')[0].lower()


def eval_textvqa(result_file, data_dir):
    annotations = json.load(open(os.path.join(data_dir, "textvqa/TextVQA_0.5.1_val.json")))["data"]
    ann_map = {(a["image_id"], a["question"].lower()): a["answers"] for a in annotations}
    pred_list = []
    for r in _load_jsonl(result_file):
        key = (r["question_id"], _textvqa_question(r["prompt"]))
        if key in ann_map:
            pred_list.append({"pred_answer": r["text"], "gt_answers": ann_map[key]})
    acc = 100 * TextVQAAccuracyEvaluator().eval_pred_list(pred_list) if pred_list else 0.0
    return {"accuracy": acc}, f"TextVQA accuracy: {acc:.2f} (n={len(pred_list)})"


# ----------------------------------------------------------------------------
# MME: acc + acc+ per subtask, summed into Perception / Cognition
# ----------------------------------------------------------------------------

MME_PERCEPTION = {"existence", "count", "position", "color", "posters",
                  "celebrity", "scene", "landmark", "artwork", "OCR"}
MME_COGNITION = {"commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"}


def _mme_parse(text):
    text = text.strip().lower()
    if text in ["yes", "no"]:
        return text
    prefix = text[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def _mme_qid(qid):
    # Some subtasks store images under "<subtask>/images/"
    parts = qid.split("/")
    if len(parts) == 3 and parts[1] == "images":
        return parts[0] + "/" + parts[2]
    return qid


def eval_mme(result_file, data_dir):
    gt_list = json.load(open(os.path.join(data_dir, "MME/mme_answers.json")))
    gt_map = {(g["question_id"], g["question"]): g["answer"] for g in gt_list}
    cat_img = defaultdict(lambda: defaultdict(list))
    for r in _load_jsonl(result_file):
        qid = _mme_qid(r["question_id"])
        prompt = r["prompt"].replace("Answer the question using a single word or phrase.", "").strip()
        gt = (gt_map.get((qid, prompt + " Please answer yes or no.")) or
              gt_map.get((qid, prompt + "  Please answer yes or no.")) or
              gt_map.get((qid, prompt)))
        if gt is None:
            continue
        cat_img[qid.split("/")[0]][qid].append((gt.lower(), _mme_parse(r["text"])))

    scores, lines = {}, []
    for cat, img_dict in sorted(cat_img.items()):
        pairs = [p for v in img_dict.values() for p in v]
        acc = sum(g == p for g, p in pairs) / len(pairs)
        # acc+: both questions of an image answered correctly
        acc_plus = sum(sum(g == p for g, p in v) == 2 for v in img_dict.values()) / len(img_dict)
        scores[cat] = 100 * (acc + acc_plus)
        lines.append(f"  {cat:24s} {scores[cat]:.2f}")
    perception = sum(v for k, v in scores.items() if k in MME_PERCEPTION)
    cognition = sum(v for k, v in scores.items() if k in MME_COGNITION)
    metrics = {"perception": perception, "cognition": cognition}
    report = [f"MME Perception: {perception:.2f}", f"MME Cognition: {cognition:.2f}",
              f"MME Total: {perception + cognition:.2f}"]
    return metrics, "\n".join(lines + report)


# ----------------------------------------------------------------------------
# ScienceQA
# ----------------------------------------------------------------------------

SQA_OPTIONS = ["A", "B", "C", "D", "E"]


def _sqa_parse(text):
    if text in SQA_OPTIONS:
        return text
    if len(text) >= 3 and text[0] in SQA_OPTIONS and text[1:3] == ". ":
        return text[0]
    res = re.findall(r'The answer is ([A-Z]).', text)
    return res[0] if len(res) == 1 else "FAILED"


def eval_sqa(result_file, data_dir):
    base = os.path.join(data_dir, "scienceqa")
    problems = json.load(open(os.path.join(base, "problems.json")))
    split_ids = json.load(open(os.path.join(base, "pid_splits.json")))["test"]
    results = {str(r["question_id"]): r for r in _load_jsonl(result_file)}
    correct = total = img_correct = img_total = 0
    for pid in split_ids:
        if pid not in results:
            continue
        prob = problems[pid]
        answer = _sqa_parse(results[pid]["text"])
        pred_idx = SQA_OPTIONS.index(answer) if answer in SQA_OPTIONS[:len(prob["choices"])] else -1
        hit = int(pred_idx == prob["answer"])
        total += 1
        correct += hit
        if prob.get("image"):
            img_total += 1
            img_correct += hit
    acc = 100 * correct / total if total else 0.0
    img_acc = 100 * img_correct / img_total if img_total else 0.0
    report = (f"ScienceQA accuracy: {acc:.2f} ({correct}/{total})\n"
              f"ScienceQA IMG accuracy: {img_acc:.2f} ({img_correct}/{img_total})")
    return {"accuracy": acc, "img_accuracy": img_acc}, report


# ----------------------------------------------------------------------------
# VizWiz: VQA accuracy, min(1, #matches / 3) averaged over the 10 leave-one-out subsets
# ----------------------------------------------------------------------------

def eval_vizwiz(result_file, data_dir):
    gt_file = os.path.join(data_dir, "vizwiz/vizwiz_answer.json")
    if not os.path.exists(gt_file):
        return {}, ("VizWiz: no local annotation file (vizwiz/vizwiz_answer.json). "
                    "Convert with scripts/convert_vizwiz_for_submission.py and submit to EvalAI.")
    gt = json.load(open(gt_file))     # list aligned with question_id
    processor = EvalAIAnswerProcessor()
    scores = []
    for r in _load_jsonl(result_file):
        pred = processor(r["text"])
        answers = [processor(a["answer"]) for a in gt[r["question_id"]]["answers"]]
        accs = []
        for i in range(len(answers)):
            match = sum(1 for j, a in enumerate(answers) if j != i and a == pred)
            accs.append(min(1.0, match / 3.0))
        scores.append(sum(accs) / len(accs))
    acc = 100 * sum(scores) / len(scores)
    return {"accuracy": acc}, f"VizWiz accuracy: {acc:.2f} (n={len(scores)})"


# ----------------------------------------------------------------------------
# MMBench dev: CircularEval (a question counts only if all circular copies are correct)
# ----------------------------------------------------------------------------

def _eval_mmbench_tsv(result_file, tsv_file, name):
    import pandas as pd
    df = pd.read_csv(tsv_file, sep="\t")
    gt_map = dict(zip(df["index"].astype(str), df["answer"].astype(str)))
    pred_map = {}
    for r in _load_jsonl(result_file):
        qid = str(r["question_id"])
        if qid in gt_map:
            pred_map[qid] = next((ch for ch in r["text"].strip() if ch in "ABCD"), "")

    correct = total = 0
    cat_total, cat_correct = defaultdict(int), defaultdict(int)
    for _, row in df[df["index"] < 1000000].iterrows():
        idx = str(row["index"])
        group = [idx] + [str(1000000 * m + int(idx)) for m in range(1, 4)
                         if str(1000000 * m + int(idx)) in gt_map]
        if any(pred_map.get(q) is None for q in group):
            continue
        cat = str(row.get("category", ""))
        total += 1
        cat_total[cat] += 1
        if all(pred_map[q] == gt_map[q].strip().upper() for q in group):
            correct += 1
            cat_correct[cat] += 1
    acc = 100 * correct / total if total else 0.0
    lines = [f"  {c}: {cat_correct[c]}/{cat_total[c]}" for c in sorted(cat_total)]
    return {"accuracy": acc}, "\n".join(lines + [f"{name} circular accuracy: {acc:.2f} ({correct}/{total})"])


def eval_mmbench(result_file, data_dir):
    return _eval_mmbench_tsv(result_file, os.path.join(data_dir, "mmbench/mmbench_dev_20230712.tsv"), "MMBench")


def eval_mmbench_cn(result_file, data_dir):
    return _eval_mmbench_tsv(result_file, os.path.join(data_dir, "mmbench/mmbench_dev_cn_20231003.tsv"), "MMBench-CN")


SCORERS = {
    "gqa":        eval_gqa,
    "pope":       eval_pope,
    "textvqa":    eval_textvqa,
    "mme":        eval_mme,
    "sqa":        eval_sqa,
    "vizwiz":     eval_vizwiz,
    "mmbench":    eval_mmbench,
    "mmbench_cn": eval_mmbench_cn,
}
