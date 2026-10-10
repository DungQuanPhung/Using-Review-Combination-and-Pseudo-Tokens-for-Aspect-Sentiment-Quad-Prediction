"""ACOS quad extraction with a local HuggingFace Qwen2.5 (decoder-only) model.

Modes
  do_train=True   : LoRA fine-tune on data/<task>/<dataset>/train.txt, then predict on test.txt
  do_train=False  : zero/few-shot prompting (n_shots in-context examples) or load a trained adapter

Reuses utils.parse_utils.parse_aste / utils.eval_utils.score so Quad-F1 (+ implicit/explicit breakdown)
is computed exactly like the T5 pipeline.  Run:  python main_qwen.py args_qwen.json
"""
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)

from utils.const import cate_list
from utils.data_utils_mix import ABSA_format_multi, read_line_examples_from_file
from utils.eval_utils import score
from utils.parse_utils import parse_aste

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class QwenArgs:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Optional[str] = field(default=None, metadata={"help": "LoRA adapter dir to load when do_train=False"})
    task_name: str = "acos"
    dataset: str = "hotel"
    output_dir: str = "./outputs/qwen/hotel"
    do_train: bool = True
    do_predict: bool = True
    seed: int = 3407
    lowercase: bool = True
    order: str = field(default="[A] [O] [S] [C]", metadata={"help": "element order of the target sequence"})
    max_samples: int = field(default=-1, metadata={"help": "limit train sentences (smoke test)"})
    max_eval_samples: int = field(default=500, metadata={"help": "limit dev sentences used for eval_loss"})
    max_test_samples: int = field(default=-1, metadata={"help": "limit test sentences"})
    n_shots: int = field(default=0, metadata={"help": "in-context examples for prompting mode"})
    # LoRA / quantization
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    # training
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_length: int = field(default=384, metadata={"help": "max prompt+target tokens for training"})
    max_new_tokens: int = 160
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 20
    eval_strategy: str = field(default="epoch", metadata={"help": "epoch / no"})

    def __post_init__(self):
        base = f"./data/{self.task_name}/{self.dataset}"
        self.train_path = f"{base}/train.txt"
        self.valid_path = f"{base}/dev.txt"
        self.test_path = f"{base}/test.txt"


SYSTEM_PROMPT = (
    "You are an aspect-based sentiment analysis system. Extract ALL sentiment quadruplets from the review sentence.\n"
    "Each quadruplet has: aspect term (A), opinion term (O), sentiment polarity (S), aspect category (C).\n"
    "- A and O must be copied verbatim from the sentence; write null if the aspect or opinion is implicit.\n"
    "- S is one of: positive, negative, neutral.\n"
    "- C is one of: {categories}.\n"
    "Output format (one line, no explanation): {order_example}\n"
    "Separate multiple quadruplets with [SSEP]."
)


def build_messages(sent, categories, order, shots=None):
    order_example = " ".join(f"{e} <{e[1].lower()}>" for e in order.split())
    system = SYSTEM_PROMPT.format(categories=", ".join(categories), order_example=order_example)
    messages = [{"role": "system", "content": system}]
    for shot_sent, shot_target in shots or []:
        messages.append({"role": "user", "content": f"Sentence: {shot_sent}"})
        messages.append({"role": "assistant", "content": shot_target})
    messages.append({"role": "user", "content": f"Sentence: {sent}"})
    return messages


def load_split(path, args, limit=-1):
    sents, labels, _ = read_line_examples_from_file(path, lowercase=args.lowercase, silence=False)
    if limit is not None and limit > 0:
        sents, labels = sents[:limit], labels[:limit]
    sents = [" ".join(w) for w in sents]
    targets = [ABSA_format_multi(lab, args.order) for lab in labels]
    return sents, targets


class SFTDataset(Dataset):
    def __init__(self, tokenizer, sents, targets, categories, args):
        self.items = []
        n_trunc = 0
        for sent, target in zip(sents, targets):
            prompt = tokenizer.apply_chat_template(build_messages(sent, categories, args.order),
                                                   tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + len(target_ids) > args.max_length:
                n_trunc += 1
                prompt_ids = prompt_ids[: max(args.max_length - len(target_ids), 16)]
            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            self.items.append({"input_ids": ids, "labels": labels})
        if n_trunc:
            logger.warning(f"{n_trunc} examples truncated to max_length={args.max_length}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attn.append([1] * len(x["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    kwargs = {"dtype": dtype, "device_map": {"": 0} if torch.cuda.is_available() else None}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model, args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=args.lora_target_modules.split(","), task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def generate_all(model, tokenizer, sents, categories, args, shots=None):
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    outputs = []
    bs = args.per_device_eval_batch_size
    device = next(model.parameters()).device
    for i in range(0, len(sents), bs):
        prompts = [tokenizer.apply_chat_template(build_messages(s, categories, args.order, shots),
                                                 tokenize=False, add_generation_prompt=True)
                   for s in sents[i:i + bs]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        if (i // bs) % 20 == 0:
            logger.info(f"generated {min(i + bs, len(sents))}/{len(sents)}")
    return outputs


def evaluate_generations(preds, targets, output_json_path):
    all_preds = [parse_aste(p) for p in preds]
    all_labels = [parse_aste(t) for t in targets]
    data_args = SimpleNamespace(output_json_path=output_json_path)
    return score(all_preds, all_labels, data_args)


def main():
    parser = HfArgumentParser(QwenArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    categories = cate_list[args.dataset]

    model, tokenizer = load_model_and_tokenizer(args)
    run_info = {
        "model_name_or_path": args.model_name_or_path, "task_name": args.task_name, "dataset": args.dataset,
        "mode": "lora_finetune" if args.do_train else ("adapter" if args.adapter_path else f"{args.n_shots}-shot"),
        "load_in_4bit": args.load_in_4bit, "lora_r": args.lora_r, "order": args.order, "seed": args.seed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    shots = None
    if args.do_train:
        train_sents, train_targets = load_split(args.train_path, args, args.max_samples)
        dev_sents, dev_targets = load_split(args.valid_path, args, args.max_eval_samples)
        model = attach_lora(model, args)
        run_info["n_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_ds = SFTDataset(tokenizer, train_sents, train_targets, categories, args)
        dev_ds = SFTDataset(tokenizer, dev_sents, dev_targets, categories, args) if args.eval_strategy != "no" else None
        run_info["n_train_examples"] = len(train_ds)

        targs = TrainingArguments(
            output_dir=args.output_dir, seed=args.seed,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate,
            warmup_steps=args.warmup_ratio,  # transformers>=5: float in [0,1) = ratio
            lr_scheduler_type=args.lr_scheduler_type, weight_decay=args.weight_decay,
            bf16=args.bf16, fp16=not args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=args.logging_steps,
            eval_strategy=args.eval_strategy, save_strategy=args.eval_strategy,
            load_best_model_at_end=args.eval_strategy != "no", metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=1,
            report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                          data_collator=PadCollator(tokenizer.pad_token_id), processing_class=tokenizer)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        train_result = trainer.train()
        train_time = time.perf_counter() - t0
        trainer.save_model(args.output_dir)
        run_info.update(train_result.metrics)
        run_info["train_time_sec"] = round(train_time, 2)
        run_info["train_time_hms"] = time.strftime("%H:%M:%S", time.gmtime(train_time))
        if torch.cuda.is_available():
            run_info["train_peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if dev_ds is not None:
            run_info["best_eval_loss"] = trainer.state.best_metric
        model = trainer.model
    else:
        if args.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
        elif args.n_shots > 0:
            train_sents, train_targets = load_split(args.train_path, args)
            rng = random.Random(args.seed)
            idx = rng.sample(range(len(train_sents)), args.n_shots)
            shots = [(train_sents[i], train_targets[i]) for i in idx]

    if args.do_predict:
        test_sents, test_targets = load_split(args.test_path, args, args.max_test_samples)
        t0 = time.perf_counter()
        preds = generate_all(model, tokenizer, test_sents, categories, args, shots)
        predict_time = time.perf_counter() - t0
        with open(os.path.join(args.output_dir, "test_predictions.txt"), "w", encoding="utf-8") as f:
            for s, p, t in zip(test_sents, preds, test_targets):
                f.write(f"{s}\t{p}\t{t}\n")
        metrics = evaluate_generations(preds, test_targets, os.path.join(args.output_dir, "pred_quads.json"))
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        metrics["test_predict_time_sec"] = round(predict_time, 2)
        metrics["test_n_sentences"] = len(test_sents)
        run_info.update(metrics)
        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
            logger.info("***** Test results *****")
            for k, v in sorted(metrics.items()):
                logger.info(f"{k} = {v}")
                f.write(f"{k} = {v}\n")

    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
