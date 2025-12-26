import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class train_config:
    # =====================================================
    # 1. 基础配置 (用户需修改这里)
    # =====================================================
    model_name: str = "/home/daiwenju/Qwen2.5-Coder-7B-Instruct"
    output_root: str = "vulnirllm_qwen7b"
    stage: str = "aux_pretrain"  # aux_pretrain | task1_finetune | joint_finetune

    # 数据路径
    train_data_path: str = "./dataset/primevul_train.jsonl"
    valid_data_path: str = "./dataset/primevul_valid.jsonl"

    # =====================================================
    # 2. 训练核心参数
    # =====================================================
    context_length: int = 6144
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_train_epochs: int = 7
    lr: float = 2e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # =====================================================
    # 3. 保存/日志
    # =====================================================
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 255
    save_total_limit: int = 10

    # =====================================================
    # 4. 量化/PEFT
    # =====================================================
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    joint_lora_r: int = 8
    joint_lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    aux_adapter_name: str = "aux_tasks"
    task1_adapter_name: str = "verdict_evidence"
    joint_adapter_name: str = "joint_refinement"

    # =====================================================
    # 5. 其他配置
    # =====================================================
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    num_workers_dataloader: int = 0
    device: str = "cuda:0"
    run_pipeline: bool = False
    eval_after_stage: bool = True

    # =====================================================
    # 6. 剪枝配置 (Stage1建议更大窗口)
    # =====================================================
    # <=0 表示自动/使用stage默认
    prune_window: int = -1
    prune_keep_head: int = -1
    prune_keep_tail: int = -1
    # 用于估计 PrimeVul 的 token 长度分布（reservoir sample）
    len_sample_size: int = 512

    # =====================================================
    # 7. 自动生成输出目录
    # =====================================================
    output_dir: str = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.output_root

        # 2. 根据 Stage 自动配置 Epoch, Output Dir, 和 Learning Rate
        if self.stage == "aux_pretrain":
            self.num_train_epochs = 7
            self.lr = 2e-5
            self.output_dir = os.path.join(self.output_root, "stage1_aux_tasks")

            # 辅助任务更大的剪枝窗口，保留更多上下文
            if self.prune_window <= 0: self.prune_window = 12
            if self.prune_keep_head <= 0: self.prune_keep_head = 20
            if self.prune_keep_tail <= 0: self.prune_keep_tail = 15

        elif self.stage == "task1_finetune":
            if self.prune_window <= 0: self.prune_window = 6
            if self.prune_keep_head <= 0: self.prune_keep_head = 10
            if self.prune_keep_tail <= 0: self.prune_keep_tail = 8
            self.num_train_epochs = 4
            self.lr = 2e-5
            self.output_dir = os.path.join(self.output_root, "stage2_task1")

        elif self.stage == "joint_finetune":
            if self.prune_window <= 0: self.prune_window = 6
            if self.prune_keep_head <= 0: self.prune_keep_head = 10
            if self.prune_keep_tail <= 0: self.prune_keep_tail = 8
            self.num_train_epochs = 3
            self.lr = 1e-5
            self.output_dir = os.path.join(self.output_root, "stage3_joint")

        else:
            raise ValueError(f"Unknown stage={self.stage}")
