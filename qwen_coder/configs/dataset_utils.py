import os
from typing import Dict, Optional

from qwen_coder.configs.vulnirllm_dataset import VulnirLLMDataset


def make_supervised_data_module(tokenizer, dataset_config) -> Dict[str, Optional[VulnirLLMDataset]]:
    """
    Build supervised datasets for the current stage.

    The training/eval split is controlled via ``train_data_path`` and ``valid_data_path``
    inside ``train_config``. Each stage reuses the same tokenizer but the dataset
    behavior (which tasks are emitted) is driven by ``dataset_config.stage``.
    """
    train_dataset = VulnirLLMDataset(
        data_path=dataset_config.train_data_path,
        tokenizer=tokenizer,
        partition="train",
        dataset_config=dataset_config,
    )

    eval_dataset = None
    if getattr(dataset_config, "valid_data_path", None) and os.path.exists(dataset_config.valid_data_path):
        eval_dataset = VulnirLLMDataset(
            data_path=dataset_config.valid_data_path,
            tokenizer=tokenizer,
            partition="validation",
            dataset_config=dataset_config,
        )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
