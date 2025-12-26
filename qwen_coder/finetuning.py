import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from qwen_coder.configs.training import train_config
from qwen_coder.configs.vulnirllm_dataset import VulnirLLMDataset
from qwen_coder.configs.dataset_utils import make_supervised_data_module
from qwen_coder.utils.pruning_utils import smart_code_pruning

from torch import nn
from transformers import Trainer, TrainerCallback


def resolve_device_map():
    # Support single GPU / multi GPU
    if torch.cuda.is_available():
        return "auto"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--merge_adapter_path", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--stage", type=str, default=None)
    args, unknown = parser.parse_known_args()

    cfg = train_config()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.num_train_epochs is not None:
        cfg.num_train_epochs = args.num_train_epochs
    if args.stage is not None:
        cfg.stage = args.stage
        cfg.__post_init__()

    signal_file = os.path.join(cfg.output_dir, "early_stop_signal")
    if os.path.exists(signal_file):
        os.remove(signal_file)

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )

    device_map = resolve_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=device_map,
        quantization_config=bnb_config if cfg.use_4bit else None,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.stage == "stage2_2" and args.merge_adapter_path:
        base_adapter = PeftModel.from_pretrained(model, args.merge_adapter_path)
        model = base_adapter.merge_and_unload()
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
        if cfg.use_peft:
            peft_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.target_modules,
                lora_dropout=cfg.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                bias="none"
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

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
        tokenizer=tokenizer
    )

    print("--> Starting Training Loop...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print("--> Training Finished & Model Saved.")


if __name__ == "__main__":
    main()
