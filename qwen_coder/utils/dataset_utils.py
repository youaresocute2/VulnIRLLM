# qwen_coder/utils/dataset_utils.py
import torch
from qwen_coder.configs.vulnirllm_dataset import VulnIRLLMDataset


def get_vulnirllm_dataset(dataset_config, tokenizer, split):
    return VulnIRLLMDataset(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        partition=split
    )


DATASET_PREPROC = {
    "vulnirllm_dataset": get_vulnirllm_dataset
}


# [修复] 增加主脚本调用的接口名称，并适配参数顺序
def get_custom_dataset(dataset_config, tokenizer, split):
    """
    Wrapper to match the call signature in finetuning.py/async_eval.py:
    get_custom_dataset(cfg, tokenizer, split)
    """
    if dataset_config.dataset not in DATASET_PREPROC:
        raise NotImplementedError(f"Dataset '{dataset_config.dataset}' is not implemented.")

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        split
    )