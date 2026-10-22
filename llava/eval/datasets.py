"""
Benchmark loaders. The layout under `--data-dir` (default: playground/data/eval)
follows LLaVA-1.5; see EVAL.md.

Every loader returns a list of rows:
    {"id": question id, "question": full prompt text (instruction included),
     "category": str, "image_path": str or "", "_b64": base64 image (MMBench only)}
"""

import os
import json
import math

from PIL import Image


def split_chunk(lst, n, k):
    """Split `lst` into n chunks and return the k-th one."""
    chunk_size = math.ceil(len(lst) / n)
    return lst[k * chunk_size: (k + 1) * chunk_size]


def _load_llava_jsonl(jsonl, img_dir):
    rows = []
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "id":         d["question_id"],
                "question":   d["text"],
                "category":   d.get("category", "default"),
                "image_path": os.path.join(img_dir, d["image"]),
            })
    return rows


def load_vqav2(data_dir):
    return _load_llava_jsonl(os.path.join(data_dir, "vqav2/llava_vqav2_mscoco_test-dev2015.jsonl"),
                             os.path.join(data_dir, "vqav2/test2015"))


def load_gqa(data_dir):
    return _load_llava_jsonl(os.path.join(data_dir, "gqa/llava_gqa_testdev_balanced.jsonl"),
                             os.path.join(data_dir, "gqa/data/images"))


def load_vizwiz(data_dir):
    return _load_llava_jsonl(os.path.join(data_dir, "vizwiz/llava_test.jsonl"),
                             os.path.join(data_dir, "vizwiz/test"))


def load_textvqa(data_dir):
    return _load_llava_jsonl(os.path.join(data_dir, "textvqa/llava_textvqa_val_v051_ocr.jsonl"),
                             os.path.join(data_dir, "textvqa/train_images"))


def load_pope(data_dir):
    return _load_llava_jsonl(os.path.join(data_dir, "pope/llava_pope_test.jsonl"),
                             os.path.join(data_dir, "pope/val2014"))


def load_mme(data_dir):
    return _load_llava_jsonl(os.path.join(data_dir, "MME/llava_mme.jsonl"),
                             os.path.join(data_dir, "MME/MME_Benchmark_release_version"))


def load_sqa(data_dir):
    """ScienceQA test set (llava_test_CQM-A.json, 4,241 questions; text-only questions run without an image)."""
    data = json.load(open(os.path.join(data_dir, "scienceqa/llava_test_CQM-A.json")))
    img_dir = os.path.join(data_dir, "scienceqa/images/test")
    rows = []
    for item in data:
        qs = item["conversations"][0]["value"].replace("<image>", "").strip()
        qs = qs + "\nAnswer with the option's letter from the given choices directly."
        rows.append({
            "id":         item["id"],
            "question":   qs,
            "category":   "scienceqa",
            "image_path": os.path.join(img_dir, item["image"]) if "image" in item else "",
        })
    return rows


def _load_mmbench_tsv(tsv_path, lang):
    """MMBench dev TSV. Circular copies (index >= 1e6) reference the base64 image of their source row."""
    import pandas as pd

    def is_none(v):
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
        return isinstance(v, str) and v.lower() in ("nan", "none")

    df = pd.read_csv(tsv_path, sep="\t")
    idx_to_b64 = {str(r["index"]): str(r["image"]) for _, r in df.iterrows() if len(str(r["image"])) >= 100}

    rows = []
    for _, row in df.iterrows():
        img_val = str(row["image"])
        b64 = img_val if len(img_val) >= 100 else idx_to_b64.get(img_val, "")

        question = row["question"]
        if not is_none(row.get("hint", "")):
            question = str(row["hint"]) + "\n" + question
        for c in ["A", "B", "C", "D"]:
            if c in row and not is_none(row[c]):
                question = question + "\n" + c + ". " + str(row[c])
        if lang == "cn":
            question += "\n请直接回答选项字母。"
        else:
            question += "\nAnswer with the option's letter from the given choices directly."

        rows.append({
            "id":         int(row["index"]),
            "question":   question,
            "category":   str(row.get("category", "")),
            "image_path": "",
            "_b64":       b64,
        })
    return rows


def load_mmbench(data_dir):
    return _load_mmbench_tsv(os.path.join(data_dir, "mmbench/mmbench_dev_20230712.tsv"), "en")


def load_mmbench_cn(data_dir):
    return _load_mmbench_tsv(os.path.join(data_dir, "mmbench/mmbench_dev_cn_20231003.tsv"), "cn")


BENCHMARKS = {
    "vqav2":      load_vqav2,
    "gqa":        load_gqa,
    "vizwiz":     load_vizwiz,
    "sqa":        load_sqa,
    "textvqa":    load_textvqa,
    "pope":       load_pope,
    "mme":        load_mme,
    "mmbench":    load_mmbench,
    "mmbench_cn": load_mmbench_cn,
}


def load_image(row):
    """Return the PIL image of a row (None for text-only ScienceQA questions)."""
    if row.get("image_path"):
        return Image.open(row["image_path"]).convert("RGB")
    if row.get("_b64"):
        import base64
        import io
        b64 = row["_b64"] + "=" * (-len(row["_b64"]) % 4)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return None
