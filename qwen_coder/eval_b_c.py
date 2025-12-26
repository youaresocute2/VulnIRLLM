# 原文件基础上做了两处替换：
# 1) smart_code_pruning(..., window_size=3) -> 增加 keep_head/keep_tail
# 2) eos_token_id 使用 model.generation_config.eos_token_id

import os
import re
import csv
import time
import argparse
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from qwen_coder.utils.pruning_utils import smart_code_pruning
from qwen_coder.configs.vulnirllm_dataset import PROMPT_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset("json", data_files={"validation": args.valid_data_path})["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.checkpoint_dir)
    model.eval()

    preds = []
    targets = []

    for item in tqdm(ds, desc="Evaluating B/C"):
        raw_code = item.get("func", "")
        vulnir_ref = item.get("vulnir", "")
        target = int(item.get("target", 0))
        targets.append(target)

        limit_b = 6144 - 512
        limit_c = 6144 - 512

        code_b = smart_code_pruning(raw_code, vulnir_ref, limit_b, tokenizer, window_size=3, keep_head=8, keep_tail=6)
        code_c = smart_code_pruning(raw_code, vulnir_ref, limit_c, tokenizer, window_size=3, keep_head=8, keep_tail=6)

        prompt = PROMPT_DICT["task_b"].format(input=code_b)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=tokenizer.eos_token_id,
        )
        out_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = 1 if "vulnerable" in out_text.lower() else 0
        preds.append(pred)

    f1 = f1_score(targets, preds)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)

    out_dir = os.path.dirname(args.checkpoint_dir.rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"eval_bc_summary_{ts}.csv")

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["F1", "Accuracy", "Precision", "Recall", "Checkpoint"])
        w.writerow([f1, acc, prec, rec, os.path.basename(args.checkpoint_dir)])

    print(f"[B/C] Saved: {summary_path} | F1={f1:.4f} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f}")


if __name__ == "__main__":
    main()
