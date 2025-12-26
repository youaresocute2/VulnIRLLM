import os
import argparse
import subprocess
import re
import pandas as pd
from datetime import datetime

# 默认配置路径（与 training.py 保持一致）
DEFAULT_ROOT = "vulnirllm_qwen7b"
DEFAULT_EXPERT_PATH = os.path.join(DEFAULT_ROOT, "stage2_1_expert/checkpoint-best")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto Evaluation Pipeline for VulnIR-LLM")
    parser.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2_1", "stage2_2"],
                        help="The training stage to evaluate.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_ROOT,
                        help="Root directory containing stage subfolders.")
    parser.add_argument("--expert_adapter_path", type=str, default=DEFAULT_EXPERT_PATH,
                        help="Path to Expert Adapter (Required for Stage 2.2).")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use.")
    return parser.parse_args()


def get_checkpoints(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return []
    ckpts = [d for d in os.listdir(ckpt_dir) if
             d.startswith("checkpoint-") and os.path.isdir(os.path.join(ckpt_dir, d))]
    # 按 step 数字排序
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    return ckpts


def parse_valid_eval_output(output_str):
    """从 valid_eval.py 的标准输出中提取指标"""
    metrics = {}
    # 匹配 F1 Score : 0.1234
    f1_match = re.search(r"F1 Score\s*:\s*([\d\.]+)", output_str)
    acc_match = re.search(r"Accuracy\s*:\s*([\d\.]+)", output_str)
    p_match = re.search(r"Precision\s*:\s*([\d\.]+)", output_str)
    r_match = re.search(r"Recall\s*:\s*([\d\.]+)", output_str)

    if f1_match: metrics["F1"] = float(f1_match.group(1))
    if acc_match: metrics["Accuracy"] = float(acc_match.group(1))
    if p_match: metrics["Precision"] = float(p_match.group(1))
    if r_match: metrics["Recall"] = float(r_match.group(1))
    return metrics


def run_stage1_eval(ckpt_dir, ckpts, gpu_id):
    """Stage 1: 批量运行 valid_eval.py (Baseline)"""
    results = []
    print(f"\n🚀 [Auto-Eval] Starting Stage 1 Evaluation on {len(ckpts)} checkpoints...")

    for ckpt in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt)
        print(f"\n>> Evaluating: {ckpt}")

        cmd = [
            "python", "-m", "qwen_coder.valid_eval",
            "--stage", "stage1",
            "--ckpt_path", ckpt_path,
            "--eval_prompt", "baseline"
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        try:
            # 捕获输出以便解析指标
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(result.stdout)  # 实时打印日志
            if result.returncode != 0:
                print(f"❌ Error evaluating {ckpt}:\n{result.stderr}")
                continue

            metrics = parse_valid_eval_output(result.stdout)
            metrics["Checkpoint"] = ckpt
            metrics["Step"] = int(ckpt.split("-")[1])
            results.append(metrics)

        except Exception as e:
            print(f"❌ Exception: {e}")

    return results


def run_stage2_1_eval(ckpt_dir, gpu_id):
    """Stage 2.1: 直接调用 eval_b_c.py (它自带循环逻辑)"""
    print(f"\n🚀 [Auto-Eval] Starting Stage 2.1 Evaluation (Task B/C)...")
    print(f"Note: eval_b_c.py already handles checkpoint iteration internally.")

    cmd = [
        "python", "-m", "qwen_coder.eval_b_c",
        "--output_dir", ckpt_dir
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    subprocess.run(cmd, env=env)
    return None  # eval_b_c 内部会保存 eval_bc_results.json


def run_stage2_2_eval(ckpt_dir, ckpts, expert_path, gpu_id):
    """Stage 2.2: 批量运行 valid_eval.py (CoT) + Expert Adapter"""
    results = []

    if not os.path.exists(expert_path):
        print(f"❌ Error: Expert Adapter not found at {expert_path}")
        print("Please check --expert_adapter_path or ensure Stage 2.1 produced a 'checkpoint-best' symlink/folder.")
        return []

    print(f"\n🚀 [Auto-Eval] Starting Stage 2.2 Evaluation (CoT + Expert)...")
    print(f"Configuration: Expert Path = {expert_path}")

    for ckpt in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt)
        print(f"\n>> Evaluating: {ckpt}")

        cmd = [
            "python", "-m", "qwen_coder.valid_eval",
            "--stage", "stage2_2",
            "--ckpt_path", ckpt_path,
            "--expert_adapter_path", expert_path,
            "--eval_prompt", "cot"
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f"❌ Error evaluating {ckpt}:\n{result.stderr}")
                continue

            metrics = parse_valid_eval_output(result.stdout)
            metrics["Checkpoint"] = ckpt
            metrics["Step"] = int(ckpt.split("-")[1])
            results.append(metrics)

        except Exception as e:
            print(f"❌ Exception: {e}")

    return results


def main():
    args = parse_args()

    # 1. 确定目标目录
    sub_dirs = {
        "stage1": "stage1_baseline",
        "stage2_1": "stage2_1_expert",
        "stage2_2": "stage2_2_fusion"
    }
    target_dir = os.path.join(args.output_root, sub_dirs[args.stage])

    if not os.path.exists(target_dir):
        print(f"❌ Error: Directory not found: {target_dir}")
        return

    # 2. 获取所有 checkpoints
    checkpoints = get_checkpoints(target_dir)
    if not checkpoints and args.stage != "stage2_1":  # Stage 2.1 脚本自己会找
        print(f"❌ No checkpoints found in {target_dir}")
        return

    # 3. 分发任务
    df = None
    if args.stage == "stage1":
        data = run_stage1_eval(target_dir, checkpoints, args.gpu)
        if data: df = pd.DataFrame(data)

    elif args.stage == "stage2_1":
        # Stage 2.1 特殊处理：直接调用 eval_b_c.py
        run_stage2_1_eval(target_dir, args.gpu)
        print(f"\n✅ Stage 2.1 Evaluation Finished. Check {target_dir}/eval_bc_results.json")
        return  # 提前结束，因为结果格式不同

    elif args.stage == "stage2_2":
        data = run_stage2_2_eval(target_dir, checkpoints, args.expert_adapter_path, args.gpu)
        if data: df = pd.DataFrame(data)

    # 4. 保存汇总报告
    if df is not None and not df.empty:
        # 按 step 排序
        df = df.sort_values(by="Step")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(target_dir, f"eval_summary_{timestamp}.csv")
        df.to_csv(report_path, index=False)

        print("\n" + "=" * 50)
        print(f"📊 Evaluation Summary Saved: {report_path}")
        print(df.to_string(index=False))
        print("=" * 50)

        # 找出最佳 Checkpoint
        if "F1" in df.columns:
            best_row = df.loc[df["F1"].idxmax()]
            print(f"🏆 Best Checkpoint based on F1: {best_row['Checkpoint']} (F1: {best_row['F1']:.4f})")
    else:
        print("\n⚠️ No results collected.")


if __name__ == "__main__":
    main()