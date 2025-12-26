# 原文件基础上做了两处替换：
# 1) smart_code_pruning(...) 统一 stage1 大窗口
# 2) eos_token_id 使用 model.generation_config.eos_token_id

import os
import re
import csv
import argparse
from datetime import datetime

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from qwen_coder.utils.pruning_utils import smart_code_pruning
from qwen_coder.configs.vulnirllm_dataset import PROMPT_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset("json", data_files={"test": args.test_data_path})["test"]

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

    for item in tqdm(ds, desc="Testing"):
        raw_code = item.get("func", "")
        vulnir_ref = item.get("vulnir", "")
        target = int(item.get("target", 0))
        targets.append(target)

        # Budget heuristic
        limit_a = 8192 - 256
        code_a = smart_code_pruning(raw_code, vulnir_ref, limit_a, tokenizer, window_size=12, keep_head=20, keep_tail=15)

        prompt = PROMPT_DICT["task_a"].format(input=code_a)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=tokenizer.eos_token_id,
        )
        out_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        pred = 1 if "vulnerable" in out_text else 0
        preds.append(pred)

    # 你原来的 test 统计逻辑保持即可，这里略


if __name__ == "__main__":
    main()
