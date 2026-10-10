"""Convert data/archive/converted_apc/*.jsonl -> data/<task>/<dataset>/{train,dev,test}.txt

Output line format expected by utils/data_utils_mix.py:
    sentence####[['aspect', 'category', 'sentiment', 'opinion'], ...]
Implicit aspect/opinion are written as 'NULL'.

Optionally registers the dataset's category token ids in utils/force_tokens.json
(required when constraint_decoding=True).
"""
import argparse
import json
import os
import re
from collections import Counter

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(text)


def convert_split(src_path, dst_path, stats):
    n_lines = 0
    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            ex = json.loads(raw)
            words = tokenize(ex["text"].replace("####", " "))
            if not words or not ex["quads"]:
                stats["skipped"] += 1
                continue
            sent = " ".join(words)
            quads = []
            for q in ex["quads"]:
                aspect = "NULL" if q["aspect_term"] is None else " ".join(tokenize(q["aspect_term"]))
                opinion = "NULL" if q["opinion_term"] is None else " ".join(tokenize(q["opinion_term"]))
                category = q["aspect_category"].strip().lower()
                sentiment = q["sentiment"].strip().lower()
                if aspect != "NULL" and aspect.lower() not in sent.lower():
                    stats["aspect_not_in_sent"] += 1
                if opinion != "NULL" and opinion.lower() not in sent.lower():
                    stats["opinion_not_in_sent"] += 1
                stats["categories"][category] += 1
                stats["sentiments"][sentiment] += 1
                quad = [aspect, category, sentiment, opinion]
                if quad not in quads:
                    quads.append(quad)
            fout.write(f"{sent}####{quads!r}\n")
            n_lines += 1
    return n_lines


def update_force_tokens(force_tokens_path, dataset, categories, model_name_or_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    ids = []
    for cat in categories:
        ids.extend(tokenizer(cat, add_special_tokens=False)["input_ids"])
    with open(force_tokens_path, "r", encoding="utf-8") as f:
        force_tokens = json.load(f)
    force_tokens["cate_tokens"][dataset] = sorted(set(ids))
    with open(force_tokens_path, "w", encoding="utf-8") as f:
        json.dump(force_tokens, f, indent=2)
    sent_ids = {w: tokenizer(w, add_special_tokens=False)["input_ids"] for w in ["positive", "negative", "neutral"]}
    print(f"cate_tokens[{dataset}] = {force_tokens['cate_tokens'][dataset]}")
    print(f"sentiment token ids (must match modeling_t5.py SP list): {sent_ids}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default="./data/archive/converted_apc")
    parser.add_argument("--task_name", default="acos")
    parser.add_argument("--dataset", default="hotel")
    parser.add_argument("--model_name_or_path", default=None,
                        help="If given, write category token ids into utils/force_tokens.json")
    parser.add_argument("--force_tokens_path", default="./utils/force_tokens.json")
    args = parser.parse_args()

    dst_dir = os.path.join("./data", args.task_name, args.dataset)
    os.makedirs(dst_dir, exist_ok=True)
    stats = {"skipped": 0, "aspect_not_in_sent": 0, "opinion_not_in_sent": 0,
             "categories": Counter(), "sentiments": Counter()}
    for split in ["train", "dev", "test"]:
        src = os.path.join(args.src_dir, f"{split}.jsonl")
        dst = os.path.join(dst_dir, f"{split}.txt")
        n = convert_split(src, dst, stats)
        print(f"{split}: {n} lines -> {dst}")
    print(f"skipped={stats['skipped']} aspect_not_in_sent={stats['aspect_not_in_sent']} "
          f"opinion_not_in_sent={stats['opinion_not_in_sent']}")
    print(f"categories={dict(stats['categories'])}")
    print(f"sentiments={dict(stats['sentiments'])}")

    categories = sorted(stats["categories"])
    print(f"Add to utils/const.py cate list for '{args.dataset}': {categories}")
    if args.model_name_or_path:
        update_force_tokens(args.force_tokens_path, args.dataset, categories, args.model_name_or_path)


if __name__ == "__main__":
    main()
