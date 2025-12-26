import copy
import json
import torch
import random
import statistics

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from qwen_coder.utils.pruning_utils import smart_code_pruning, fallback_pruning, extract_critical_lines, \
    structured_ir_pruning

# ================================
# Templates
# ================================
PROMPT_DICT = {
    "task_a": (
        "<|im_start|>system\n"
        "You are a security expert specialized in detecting vulnerabilities in code.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Determine whether the following code is vulnerable.\n"
        "Only answer with one word: Vulnerable or Safe.\n\n"
        "Input Code:\n{input}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "task_a_cot": (
        "<|im_start|>system\n"
        "You are a security expert specialized in detecting vulnerabilities in code.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "You are given a code snippet and a vulnerability reference.\n"
        "Analyze if the code is vulnerable and explain briefly.\n"
        "Then output a strict verdict line: Verdict: Vulnerable or Verdict: Safe.\n\n"
        "Input Code:\n{input}\n\n"
        "Vulnerability Reference:\n{vulnir}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "task_b": (
        "<|im_start|>system\n"
        "You are a security expert.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Given the code below, identify the vulnerable line(s) and explain.\n\n"
        "Input Code:\n{input}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "task_c": (
        "<|im_start|>system\n"
        "You are a security expert.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Given the code below and a vulnerability reference, generate a vulnerability IR/trace.\n\n"
        "Input Code:\n{input}\n\n"
        "Vulnerability Reference:\n{vulnir}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


SAFETY_BUFFER = 50


class VulnirLLMDataset(Dataset):
    """
    PrimeVul JSONL fields expected:
      - func: str (function code)
      - target: int (1 vulnerable, 0 safe)
      - vulnir: str (optional, for guided pruning & stage2)
    """

    def __init__(self, data_path, tokenizer: AutoTokenizer, partition="train", dataset_config=None):
        self.data = []
        self.tokenizer = tokenizer
        self.split = partition

        self.stage = getattr(dataset_config, "stage", "stage1")
        self.max_total_tokens = getattr(dataset_config, "context_length", 6144)
        self.total_processed = 0

        # -------------------------------
        # Pruning controls (stage-aware)
        # -------------------------------
        if self.stage == "stage1":
            _default_window, _default_head, _default_tail = 12, 20, 15
        else:
            _default_window, _default_head, _default_tail = 3, 8, 6

        # User overrides (<=0 means use default)
        self._prune_window_override = int(getattr(dataset_config, "prune_window", 0))
        self._prune_head_override = int(getattr(dataset_config, "prune_keep_head", 0))
        self._prune_tail_override = int(getattr(dataset_config, "prune_keep_tail", 0))

        self.prune_window = int(_default_window if self._prune_window_override <= 0 else self._prune_window_override)
        self.prune_keep_head = int(_default_head if self._prune_head_override <= 0 else self._prune_head_override)
        self.prune_keep_tail = int(_default_tail if self._prune_tail_override <= 0 else self._prune_tail_override)
        self.prune_window = max(0, self.prune_window)
        self.prune_keep_head = max(0, self.prune_keep_head)
        self.prune_keep_tail = max(0, self.prune_keep_tail)

        # ChatML terminator: prefer <|im_end|> over eos_token (<|endoftext|> on Qwen2.5)
        self.im_end_token = "<|im_end|>"
        try:
            _ = self.tokenizer.encode(self.im_end_token, add_special_tokens=False)
        except Exception:
            self.im_end_token = self.tokenizer.eos_token

        # Prompt lengths (empty input) for token budgeting
        self.prompt_lens = {
            k: len(self.tokenizer.encode(v.format(input="", vulnir="") if "{vulnir}" in v else v.format(input=""),
                                         add_special_tokens=False))
            for k, v in PROMPT_DICT.items()
        }

        # Stage1 label budget (token-accurate)
        self.label_budget_stage1 = max(
            len(self.tokenizer.encode("Vulnerable" + self.im_end_token, add_special_tokens=False)),
            len(self.tokenizer.encode("Safe" + self.im_end_token, add_special_tokens=False))
        )

        # PrimeVul code-length statistics (reservoir sample in token space)
        self._len_sample_size = int(getattr(dataset_config, "len_sample_size", 512))
        self._code_toklen_samples = []
        self._seen_for_sample = 0

        loaded_count = 0
        skipped_count = 0

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 简单检查必要字段
                if "func" not in item:
                    skipped_count += 1
                    continue

                # Reservoir sample (token stats) - tokenize only when selected
                self._seen_for_sample += 1
                if self._len_sample_size > 0 and isinstance(item.get("func", None), str) and item["func"]:
                    if len(self._code_toklen_samples) < self._len_sample_size:
                        self._code_toklen_samples.append(
                            len(self.tokenizer.encode(item["func"], add_special_tokens=False))
                        )
                    else:
                        j = random.randint(0, self._seen_for_sample - 1)
                        if j < self._len_sample_size:
                            self._code_toklen_samples[j] = len(
                                self.tokenizer.encode(item["func"], add_special_tokens=False)
                            )

                self.data.append(item)
                loaded_count += 1
            except json.JSONDecodeError:
                skipped_count += 1
                continue

        print(f"--> Mode: {self.stage.upper()} | Loaded: {loaded_count} | Skipped (Error/Missing): {skipped_count}")

        # Report PrimeVul code token-length stats (sample-based)
        if self._code_toklen_samples:
            arr = sorted(self._code_toklen_samples)
            n = len(arr)
            mean = sum(arr) / n
            p50 = arr[n // 2]
            p90 = arr[min(n - 1, int(n * 0.90))]
            p99 = arr[min(n - 1, int(n * 0.99))]
            mx = arr[-1]
            print(f"--> PrimeVul(func) token-length stats (reservoir sample n={n}): "
                  f"mean={mean:.1f}, p50={p50}, p90={p90}, p99={p99}, max={mx}")

            # Auto-tune pruning window for stage1 if user didn't override
            # Heuristic: if most functions fit easily, we can afford larger windows.
            if self.stage == "stage1" and self._prune_window_override <= 0:
                if p90 <= int(self.max_total_tokens * 0.55):
                    self.prune_window = max(self.prune_window, 14)
                elif p90 <= int(self.max_total_tokens * 0.75):
                    self.prune_window = max(self.prune_window, 12)
                else:
                    self.prune_window = max(self.prune_window, 10)
                print(f"--> Auto-tuned stage1 prune_window={self.prune_window} "
                      f"(keep_head={self.prune_keep_head}, keep_tail={self.prune_keep_tail})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_code = item.get("func", "")
        target = int(item.get("target", 0))
        vulnir_ref = item.get("vulnir", "")

        # Output labels
        label = "Vulnerable" if target == 1 else "Safe"

        # =======================
        # Stage 1: Task A only
        # =======================
        if self.stage == "stage1":
            limit_a = self.max_total_tokens - self.prompt_lens["task_a"] - self.label_budget_stage1 - SAFETY_BUFFER
            code_a = smart_code_pruning(
                raw_code,
                vulnir_ref,
                limit_a,
                self.tokenizer,
                window_size=self.prune_window,
                keep_head=self.prune_keep_head,
                keep_tail=self.prune_keep_tail,
            )
            prompt = PROMPT_DICT["task_a"].format(input=code_a)
            return self._pack("task_a", prompt, label)

        # =======================
        # Stage 2.1: Expert Mode
        # =======================
        elif self.stage == "stage2_1":
            # Decide tasks (B/C) etc. (keep original behavior)
            # Here we keep your original logic; only pruning params are upgraded.
            for _ in range(3):
                pick_c = random.random() < 0.5
                if pick_c:
                    # Task C
                    limit_c_code = self.max_total_tokens - self.prompt_lens["task_c"] - 256 - SAFETY_BUFFER
                    code_c = smart_code_pruning(
                        raw_code,
                        vulnir_ref,
                        limit_c_code,
                        self.tokenizer,
                        window_size=self.prune_window,
                        keep_head=self.prune_keep_head,
                        keep_tail=self.prune_keep_tail,
                    )
                    vulnir_out = structured_ir_pruning(vulnir_ref, 256, self.tokenizer)
                    return self._pack("task_c", PROMPT_DICT["task_c"].format(input=code_c, vulnir=vulnir_ref), vulnir_out)
                else:
                    # Task B
                    limit_b = self.max_total_tokens - self.prompt_lens["task_b"] - 256 - SAFETY_BUFFER
                    code_b = smart_code_pruning(
                        raw_code,
                        vulnir_ref,
                        limit_b,
                        self.tokenizer,
                        window_size=self.prune_window,
                        keep_head=self.prune_keep_head,
                        keep_tail=self.prune_keep_tail,
                    )
                    vulnir_out = structured_ir_pruning(vulnir_ref, 256, self.tokenizer)
                    return self._pack("task_b", PROMPT_DICT["task_b"].format(input=code_b), vulnir_out)

            return self._pack("task_a", PROMPT_DICT["task_a"].format(input=""), "Safe")

        # =======================
        # Stage 2.2: Fusion Mode
        # =======================
        elif self.stage == "stage2_2":
            # Keep your original fusion behavior (task_a_cot + fused output)
            for _ in range(3):
                vulnir_str = vulnir_ref
                fused_output = f"Analysis:\n{vulnir_str}\n\nVerdict: {label}"

                limit_a = self.max_total_tokens - self.prompt_lens["task_a_cot"] - 512 - SAFETY_BUFFER
                code_a = smart_code_pruning(
                    raw_code,
                    vulnir_str,
                    limit_a,
                    self.tokenizer,
                    window_size=self.prune_window,
                    keep_head=self.prune_keep_head,
                    keep_tail=self.prune_keep_tail,
                )

                return self._pack(
                    "task_a_cot",
                    PROMPT_DICT["task_a_cot"].format(input=code_a, vulnir=vulnir_str),
                    fused_output
                )

            return self._pack("task_a", PROMPT_DICT["task_a"].format(input=""), "Safe")

        else:
            return self._pack("task_a", PROMPT_DICT["task_a"].format(input=""), "Safe")

    def _pack(self, task_type, prompt, output):
        # IMPORTANT: use ChatML message terminator
        completion = output + self.im_end_token

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        # Hard truncate (should be rare after correct budgeting)
        input_ids = input_ids[: self.max_total_tokens]
        labels = labels[: self.max_total_tokens]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "task_ids": task_type,
        }
