"""
Video-LLaVA zero-shot video QA inference with ST-Pruner.

Adapted from Video-LLaVA's `videollava/eval/video/run_inference_video_qa.py`; the only
changes are the pruner installation and the per-question answer lookup.

    python video_llava/infer_prune.py \
        --model_path LanguageBind/Video-LLaVA-7B --cache_dir ./cache_dir \
        --video_dir ./playground/data/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/videos \
        --gt_file_question ./playground/data/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_q.json \
        --gt_file_answers  ./playground/data/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json \
        --output_dir results/Video-LLaVA-7B/MSVD/stpruner_136 --output_name 1_0 \
        --method stpruner --per-frame-keep 17
"""

import math
import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_prune
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model


def get_chunk(lst, n, k):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)][k]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--video_dir', required=True, help='directory containing the video files')
    parser.add_argument('--gt_file_question', required=True)
    parser.add_argument('--gt_file_answers', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_name', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--skip-first", dest="skip_first", type=int, default=0,
                        help="resume: skip the first N questions of this chunk and append")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--model_base', default=None, type=str)
    parser.add_argument("--seed", type=int, default=0, help="seed for the randomized SVD")
    # ST-Pruner
    parser.add_argument("--method", default="stpruner", choices=["none", "spectral", "semantic", "stpruner"],
                        help="none: full 8x256 tokens | spectral: spectral-only | stpruner: ours")
    parser.add_argument("--per-frame-keep", type=int, default=17,
                        help="tokens kept per 256-token frame (17 -> 136 tokens over 8 frames)")
    parser.add_argument("--rank", type=int, default=0, help="spectral rank R (0: R = N/8 = 32)")
    parser.add_argument("--sv-power", type=float, default=0.5)
    parser.add_argument("--decomp", default="lowrank", choices=["lowrank", "full"])
    return parser.parse_args()


def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if getattr(model, "_st_set_question", None) is not None:
        model._st_set_question(qs)
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN] * 8) + '\n' + qs

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


def run_inference(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    if args.method != "none":
        text_encoder = None
        if args.method != "spectral":
            vt = model.get_model().get_video_tower()
            text_encoder = video_prune.LanguageBindTextEncoder(
                vt.video_tower_name, args.cache_dir, args.device, next(model.parameters()).dtype)
        video_prune.install_video_pruner(model, args.method, args.per_frame_keep, rank=args.rank,
                                         sv_power=args.sv_power, decomp=args.decomp,
                                         text_encoder=text_encoder)
        print(f"[ST-Pruner] method={args.method}, {args.per_frame_keep} tokens/frame "
              f"({8 * args.per_frame_keep} video tokens)")

    gt_questions = get_chunk(json.load(open(args.gt_file_question)), args.num_chunks, args.chunk_idx)
    gt_answers = {x["question_id"]: x["answer"] for x in json.load(open(args.gt_file_answers))}

    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    if args.skip_first > 0:
        gt_questions = gt_questions[args.skip_first:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    for sample in tqdm(gt_questions):
        qid, question = sample['question_id'], sample['question']
        sample_set = {'id': qid, 'question': question, 'answer': gt_answers[qid]}
        for fmt in video_formats:
            video_path = os.path.join(args.video_dir, f"{sample['video_name']}{fmt}")
            if os.path.exists(video_path):
                sample_set['pred'] = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                ans_file.write(json.dumps(sample_set) + "\n")
                ans_file.flush()
                break
    ans_file.close()


if __name__ == "__main__":
    run_inference(parse_args())
