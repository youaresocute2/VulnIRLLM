import os
import re
import csv
import argparse
from datetime import datetime

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from qwen_coder.utils.pruning_utils import smart_code_pruning
from qwen_coder.configs.vulnirllm_dataset import PROMPT_DICT


def extract_verdict(text: str) -> int:
    """
    Robust verdict extractor for CoT-style outputs.
    Returns: 1 for Vulnerable, 0 for Safe.
    """
    if not text:
        return 0
    t = text.strip()

    m = re.search(r'(?i)\bverdict\s*:\s*(vulnerable|safe)\b', t)
    if m:
        return 1 if m.group(1).lower().startswith("vuln") else 0

    if re.search(r'(?i)\bvulnerable\b', t):
        return 1
    if re.search(r'(?i)\bsafe\b', t):
        return 0
    return 0


@torch.no_grad()
def _score_candidate_cached(model, prompt_ids, cand_ids):
    """
    Compute log P(cand | prompt) efficiently:
      1) forward(prompt, use_cache=True) to get past + last logits
      2) score first cand token from prompt last logits
      3) forward(cand, past_key_values=prompt_past) to score remaining tokens
    """
    device = model.device
    prompt = torch.tensor([prompt_ids], device=device)

    out = model(input_ids=prompt, use_cache=True)
    past = out.past_key_values

    last_logits = out.logits[0, -1]
    logp = torch.log_softmax(last_logits, dim=-1)[cand_ids[0]].item()

    if len(cand_ids) == 1:
        return logp

    cand = torch.tensor([cand_ids], device=device)
    out2 = model(input_ids=cand, past_key_values=past, use_cache=False)
    logits = out2.logits[0]

    for i in range(1, len(cand_ids)):
        logp += torch.log_softmax(logits[i - 1], dim=-1)[cand_ids[i]].item()

    return logp


@torch.no_grad()
def predict_stage1_logprob(model, tokenizer, prompt: str) -> int:
    """
    Stage1 binary classification via conditional log-prob comparison.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    vuln_cands = ["Vulnerable", " Vulnerable"]
    safe_cands = ["Safe", " Safe"]

    def best_score(texts):
        best = None
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            s = _score_candidate_cached(model, prompt_ids, ids)
            best = s if best is None else max(best, s)
        return best

    s_v = best_score(vuln_cands)
    s_s = best_score(safe_cands)

    return 1 if s_v > s_s else 0


def _build_pruned_prompt(
    item,
    tokenizer,
    context_length: int,
    stage: str,
    use_cot: bool,
    prune_window: int,
    prune_keep_head: int,
    prune_keep_tail: int,
    safety_buffer: int = 64,
):
    raw_code = item.get("func", "")
    vulnir_ref = item.get("vulnir", "")

    template = PROMPT_DICT["task_a_cot"] if use_cot else PROMPT_DICT["task_a"]

    base_prompt = template.format(input="", vulnir="") if "{vulnir}" in template else template.format(input="")
    base_len = len(tokenizer.encode(base_prompt, add_special_tokens=False))

    lab_v = len(tokenizer.encode("Vulnerable", add_special_tokens=False))
    lab_s = len(tokenizer.encode("Safe", add_special_tokens=False))
    label_budget = max(lab_v, lab_s)

    code_budget = max(0, context_length - base_len - label_budget - safety_buffer)

    code_pruned = smart_code_pruning(
        raw_code,
        vulnir_ref,
        code_budget,
        tokenizer,
        window_size=prune_window,
        keep_head=prune_keep_head,
        keep_tail=prune_keep_tail,
    )

    prompt = template.format(input=code_pruned, vulnir=vulnir_ref) if "{vulnir}" in template else template.format(input=code_pruned)
    return prompt


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset,
    context_length: int,
    stage: str,
    use_cot: bool,
    prune_window: int,
    prune_keep_head: int,
    prune_keep_tail: int,
    use_logprob_stage1: bool = True,
):
    preds = []
    targets = []

    model.eval()

    for item in tqdm(dataset, desc="Evaluating"):
        target = int(item.get("target", 0))
        prompt = _build_pruned_prompt(
            item=item,
            tokenizer=tokenizer,
            context_length=context_length,
            stage=stage,
            use_cot=use_cot,
            prune_window=prune_window,
            prune_keep_head=prune_keep_head,
            prune_keep_tail=prune_keep_tail,
        )

        if stage == "stage1" and not use_cot and use_logprob_stage1:
            pred = predict_stage1_logprob(model, tokenizer, prompt)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=96 if use_cot else 10,
                do_sample=False,
                eos_token_id=getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
                pad_token_id=tokenizer.eos_token_id,
            )
            out_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_verdict(out_text) if use_cot else (1 if "vulnerable" in out_text.lower() else 0)

        preds.append(pred)
        targets.append(target)

    f1 = f1_score(targets, preds)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    return f1, acc, prec, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)

    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2_1", "stage2_2"])
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument("--context_length", type=int, default=8192)

    parser.add_argument("--prune_window", type=int, default=12)
    parser.add_argument("--prune_keep_head", type=int, default=20)
    parser.add_argument("--prune_keep_tail", type=int, default=15)

    parser.add_argument("--no_logprob", action="store_true")
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

    f1, acc, prec, rec = evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        context_length=args.context_length,
        stage=args.stage,
        use_cot=args.use_cot,
        prune_window=args.prune_window,
        prune_keep_head=args.prune_keep_head,
        prune_keep_tail=args.prune_keep_tail,
        use_logprob_stage1=(not args.no_logprob),
    )

    out_dir = os.path.dirname(args.checkpoint_dir.rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"eval_summary_{ts}.csv")

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["F1", "Accuracy", "Precision", "Recall", "Checkpoint"])
        w.writerow([f1, acc, prec, rec, os.path.basename(args.checkpoint_dir)])

    print("=" * 50)
    print(f"📊 Evaluation Summary Saved: {summary_path}")
    print(f"F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Ckpt={os.path.basename(args.checkpoint_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
