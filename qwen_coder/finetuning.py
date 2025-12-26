import argparse
import os
from copy import deepcopy

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from qwen_coder.configs.dataset_utils import make_supervised_data_module
from qwen_coder.configs.training import train_config
from qwen_coder.utils.pruning_utils import smart_code_pruning
from qwen_coder.valid_eval import evaluate as verdict_eval


def resolve_device_map(device: str):
    # Force single-GPU execution (cuda:0) for the refactored pipeline.
    if torch.cuda.is_available():
        return {"": device}
    return None


class AsyncEarlyStoppingCallback(TrainerCallback):
    def __init__(self, signal_file: str):
        self.signal_file = signal_file

    def on_step_end(self, args, state, control, **kwargs):
        if os.path.exists(self.signal_file):
            print("--> Early stop signal detected. Stopping training...")
            control.should_training_stop = True
        return control


class TaskDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        task_ids = [f.pop("task_ids", "task_a") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        batch["task_ids"] = task_ids
        return batch


class DeepDualTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_ids = inputs.pop("task_ids", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)


def clone_with_stage(cfg: train_config, stage: str) -> train_config:
    next_cfg = deepcopy(cfg)
    next_cfg.stage = stage
    next_cfg.__post_init__()
    return next_cfg


def attach_lora_adapter(model, cfg: train_config):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    if not cfg.use_peft:
        return model

    # Stage-specific LoRA footprint (joint stage uses a lighter adapter).
    lora_r = cfg.joint_lora_r if cfg.stage == "joint_finetune" else cfg.lora_r
    lora_alpha = cfg.joint_lora_alpha if cfg.stage == "joint_finetune" else cfg.lora_alpha
    adapter_name = {
        "aux_pretrain": cfg.aux_adapter_name,
        "task1_finetune": cfg.task1_adapter_name,
        "joint_finetune": cfg.joint_adapter_name,
    }.get(cfg.stage, "default")

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    model.train_adapter(adapter_name)
    model.print_trainable_parameters()
    return model


def build_base_model(cfg: train_config, merge_adapter_path: str = None):
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    device_map = resolve_device_map(cfg.device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=device_map,
        quantization_config=bnb_config if cfg.use_4bit else None,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=True,
    )

    if merge_adapter_path:
        print(f"--> Merging previous adapter from {merge_adapter_path} into base weights")
        merged = PeftModel.from_pretrained(model, merge_adapter_path)
        model = merged.merge_and_unload()

    return model


def evaluate_aux_tasks(model, tokenizer, cfg: train_config):
    if not getattr(cfg, "valid_data_path", None):
        return None

    ds = load_dataset("json", data_files={"validation": cfg.valid_data_path})["validation"]
    preds, targets = [], []

    model.eval()
    for item in ds:
        raw_code = item.get("func", "")
        vulnir_ref = item.get("vulnir", "")
        target = int(item.get("target", 0))
        targets.append(target)

        limit_ctx = cfg.context_length - 512
        code_b = smart_code_pruning(
            raw_code,
            vulnir_ref,
            limit_ctx,
            tokenizer,
            window_size=cfg.prune_window,
            keep_head=cfg.prune_keep_head,
            keep_tail=cfg.prune_keep_tail,
        )

        prompt = (
            "<|im_start|>system\nYou are a security expert.\n<|im_end|>\n"
            "<|im_start|>user\n"
            "Given the code below, identify the vulnerable line(s) and explain.\n\n"
            f"Input Code:\n{code_b}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=tokenizer.eos_token_id,
        )
        out_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(1 if "vulnerable" in out_text.lower() else 0)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "F1": f1_score(targets, preds),
        "Accuracy": accuracy_score(targets, preds),
        "Precision": precision_score(targets, preds, zero_division=0),
        "Recall": recall_score(targets, preds, zero_division=0),
    }


def evaluate_verdict(model, tokenizer, cfg: train_config, use_cot: bool):
    if not getattr(cfg, "valid_data_path", None):
        return None

    ds = load_dataset("json", data_files={"validation": cfg.valid_data_path})["validation"]
    f1, acc, prec, rec = verdict_eval(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        context_length=cfg.context_length,
        stage=cfg.stage,
        use_cot=use_cot,
        prune_window=cfg.prune_window,
        prune_keep_head=cfg.prune_keep_head,
        prune_keep_tail=cfg.prune_keep_tail,
        use_logprob_stage1=False,
    )
    return {
        "F1": f1,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
    }


def run_single_stage(cfg: train_config, merge_adapter_path: str = None):
    signal_file = os.path.join(cfg.output_dir, "early_stop_signal")
    if os.path.exists(signal_file):
        os.remove(signal_file)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = build_base_model(cfg, merge_adapter_path=merge_adapter_path)
    model = attach_lora_adapter(base_model, cfg)

    data_module = make_supervised_data_module(tokenizer=tokenizer, dataset_config=cfg)
    collator = TaskDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps if cfg.save_strategy == "steps" else None,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        optim=cfg.optim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        report_to="none",
    )

    trainer = DeepDualTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module.get("eval_dataset", None),
        data_collator=collator,
        callbacks=[AsyncEarlyStoppingCallback(signal_file)],
        tokenizer=tokenizer,
    )

    print(f"--> Starting Training Loop for stage: {cfg.stage}")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"--> Training Finished & Model Saved to {cfg.output_dir}.")

    if cfg.eval_after_stage and data_module.get("eval_dataset") is not None:
        eval_summaries = {}
        if cfg.stage == "aux_pretrain":
            eval_summaries["aux_tasks"] = evaluate_aux_tasks(model, tokenizer, cfg)
        elif cfg.stage == "task1_finetune":
            eval_summaries["task1"] = evaluate_verdict(model, tokenizer, cfg, use_cot=True)
        elif cfg.stage == "joint_finetune":
            eval_summaries["task1"] = evaluate_verdict(model, tokenizer, cfg, use_cot=True)
            eval_summaries["aux_tasks"] = evaluate_aux_tasks(model, tokenizer, cfg)

        if eval_summaries:
            print(f"--> Evaluation summaries for {cfg.stage}: {eval_summaries}")

    return cfg.output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--merge_adapter_path", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--run_pipeline", action="store_true", help="Run the full 3-stage pipeline sequentially.")
    args, unknown = parser.parse_known_args()

    cfg = train_config()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.num_train_epochs is not None:
        cfg.num_train_epochs = args.num_train_epochs
    if args.stage is not None:
        cfg.stage = args.stage
        cfg.__post_init__()
    if args.run_pipeline:
        cfg.run_pipeline = True

    if not torch.cuda.is_available():
        print("⚠️ CUDA is not available; training will fall back to CPU.")

    if cfg.run_pipeline:
        print("--> Running the serial single-GPU pipeline across 3 stages.")
        prev_adapter = None
        for stage in ["aux_pretrain", "task1_finetune", "joint_finetune"]:
            stage_cfg = clone_with_stage(cfg, stage)
            prev_adapter = run_single_stage(stage_cfg, merge_adapter_path=prev_adapter)
        return

    run_single_stage(cfg, merge_adapter_path=args.merge_adapter_path)


if __name__ == "__main__":
    main()
