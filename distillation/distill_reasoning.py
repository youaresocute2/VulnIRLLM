import json
import os
import asyncio
import time
import logging
import hashlib
from typing import List, Dict, Optional, Set
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区域 =================
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("distillation_multiturn.log"), logging.StreamHandler()]
)

# API 配置 (对应 start_server.sh 的 vLLM 设置)
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Llama-3.1"

# 文件路径
INPUT_FILE = "./dataset/train_aug_paired.jsonl"
OUTPUT_FILE = "./dataset/train-llama_paired.jsonl"

# --- 长度控制参数 ---
# 4090 (24GB) 建议安全上下文长度控制在 10k-12k tokens 以内，留足显存给 KV Cache
MAX_INPUT_LENGTH_TOKENS = 13000

# 核心参数
MAX_CONCURRENT_REQUESTS = 20
MAX_RETRIES = 2
MAX_NEW_TOKENS = 800  # 足够覆盖 100 单词的 JSON 输出
TEMPERATURE = 0.2
TOP_P = 0.95

# 强制 JSON 输出 Schema (深度思考：限制长度，强制代码引用)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {
            "type": "string",
            "description": "Concise technical analysis (30-100 words). MUST cite specific variables/functions from code. NO generic CVE summaries."
        },
        "safety_justification": {
            "type": "string",
            "description": "Concise fix explanation (30-100 words). Cite the specific check/logic added."
        }
    },
    "required": ["root_cause", "safety_justification"]
}

# ================= Few-Shot 知识库 (Domain Knowledge) =================

# [修改] 更新为“短小精悍”且“代码感知”的范例
FEW_SHOT_EXAMPLES = """
### Example 1 (Ideal Output):
**User Input**: Analyze `void snd_msnd_dsp_write(...)`
**Expert Analysis**:
{
  "root_cause": "Missing boundary check on the `channel` parameter allows it to be used as an array index even if negative or out-of-bounds. This leads to a heap buffer over-read when accessing `channel_map[channel]`, potentially leaking kernel memory.",
  "safety_justification": "The patch adds an explicit bounds check `if (channel >= size)`. This ensures the `channel` index is within the valid range of `channel_map` before access, preventing the out-of-bounds read."
}

### Example 2 (Logical Boundary Check):
**User Input**: Analyze `static int kvm_vcpu_ioctl_x86_setup_mce(...)`
**Expert Analysis**:
{
  "root_cause": "The variable `bank_num` is used as an index for `mce_banks` without verifying it stays below `KVM_MAX_MCE_BANKS`. An attacker can supply a large `bank_num` via IOCTL to trigger a buffer overflow.",
  "safety_justification": "A validation logic `if (bank_num >= KVM_MAX_MCE_BANKS)` is inserted. This strictly rejects invalid bank indices, ensuring subsequent array operations remain within allocated memory."
}
"""


# ================= 辅助工具函数 =================

def approximate_token_count(text: str) -> int:
    """快速估算 Token 数量 (按 1 token ~= 3.5 chars 计算，代码通常较密集)"""
    if not text:
        return 0
    return int(len(text) / 3.5)


def truncate_code(code: str, max_tokens: int) -> str:
    """
    智能截断代码：保留头尾，中间截断。
    策略：保留 40% 头部 (变量定义) + 60% 尾部 (逻辑判断通常在后面)
    """
    current_tokens = approximate_token_count(code)
    if current_tokens <= max_tokens:
        return code

    # 将 Token 预算转换为字符预算
    char_budget = int(max_tokens * 3.5)

    # 头部保留 40%，尾部保留 60%
    head_chars = int(char_budget * 0.4)
    tail_chars = int(char_budget * 0.6)

    truncated_text = (
            code[:head_chars]
            + "\n\n... [CODE TRUNCATED DUE TO LENGTH CONSTRAINT] ...\n\n"
            + code[-tail_chars:]
    )
    return truncated_text


def generate_content_hash(item: Dict) -> str:
    """
    生成唯一指纹，用于无 ID 数据的断点续传。
    使用 func_before 和 cve 字段的组合进行哈希。
    """
    # 提取关键字段，如果不存在则使用空字符串
    content = f"{item.get('cve', '')}|{item.get('func_before', '')}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


# ================= 核心逻辑类 =================

class DistillationAgent:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    def _build_system_prompt(self) -> str:
        # [修改] 强化系统提示词：强调 Concise, Code-Specific, No Fluff
        return (
            "You are an elite Vulnerability Research Expert. "
            "Your goal is to perform a deep technical analysis of code vulnerabilities.\n\n"
            "**STRICT OUTPUT RULES**:\n"
            "1. **Length**: Each field MUST be between **30 and 100 words**. Be extremely concise.\n"
            "2. **Code-Centric**: You MUST quote specific **variable names** (e.g., `len`, `buffer`), **function names**, or **logic conditions** from the provided code.\n"
            "3. **No Fluff**: Do NOT use phrases like 'The root cause is...', 'This vulnerability is related to...'. Start directly with the logic (e.g., 'Missing check on `len` leads to...').\n"
            "4. **Context Usage**: Use the CVE info to understand the bug, but your explanation must describe the **code**, not just summarize the CVE text.\n"
            f"**Reference Examples**:\n{FEW_SHOT_EXAMPLES}"
        )

    async def _call_llm(self, messages: List[Dict], use_json: bool = False) -> str:
        """封装单次 LLM 调用，支持重试"""
        for attempt in range(MAX_RETRIES + 1):
            try:
                extra_body = {
                    "repetition_penalty": 1.05,
                    "top_k": 50
                }
                if use_json:
                    extra_body["guided_json"] = JSON_SCHEMA

                response = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_NEW_TOKENS,
                    extra_body=extra_body
                )
                return response.choices[0].message.content
            except Exception as e:
                # 如果是 Context Length 错误，直接抛出，不要重试（重试也没用）
                if "maximum context length" in str(e):
                    logging.error(f"Context Overflow Error: {str(e)[:200]}...")
                    raise ValueError("CONTEXT_OVERFLOW")

                if attempt == MAX_RETRIES:
                    logging.warning(f"LLM Call Failed: {e}")
                    raise e
                await asyncio.sleep(1 * (attempt + 1))
        return ""

    async def distill(self, item: Dict) -> Dict:
        """
        执行 5 轮多步蒸馏流程 (Multi-Turn Chain)
        包含主动截断逻辑
        """
        # 1. 准备上下文
        func_before = item.get('func_before', '')
        func_after = item.get('func_after', '')

        # === 主动截断逻辑与标签 ===
        len_before = approximate_token_count(func_before)
        len_after = approximate_token_count(func_after)
        total_tokens = len_before + len_after

        if total_tokens > MAX_INPUT_LENGTH_TOKENS:
            # 给每个函数分配一半预算 (减去一些 buffer 给 prompt)
            limit_per_func = int((MAX_INPUT_LENGTH_TOKENS - 1000) / 2)
            func_before = truncate_code(func_before, limit_per_func)
            func_after = truncate_code(func_after, limit_per_func)

            # [统计] 添加 truncation_warning 标签
            item['truncation_warning'] = True

        cwe_list = item.get('cwe', [])
        cwe_str = ", ".join(cwe_list) if cwe_list else "Unknown CWE"

        # 获取 CVE 信息
        cve_id = item.get('cve', 'Unknown CVE')
        cve_desc = item.get('cve_desc', '')
        if not cve_desc:
            cve_desc = f"A vulnerability classified as {cwe_str}."

        # 初始化对话历史
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        try:
            # === Turn 1: Code Diff Perception ===
            user_msg_1 = (
                f"### Task: Analyze Code Diff\n"
                f"**Vulnerable Code**:\n```c\n{func_before}\n```\n"
                f"**Patched Code**:\n```c\n{func_after}\n```\n\n"
                f"Step 1: Identify the logical difference. What specific check or operation was added/modified? Mention specific variable names."
            )
            messages.append({"role": "user", "content": user_msg_1})
            resp_1 = await self._call_llm(messages)
            messages.append({"role": "assistant", "content": resp_1})

            # === Turn 2: CVE Grounding & Verification (Detailed Root Cause) ===
            # [修改] 强化指令：要求 30-100 词，基于代码事实
            user_msg_2 = (
                f"### Task: Root Cause Analysis\n"
                f"**Context (Reference Only)**: {cve_id} - {cve_desc}\n\n"
                f"Step 2: Explain the **Root Cause** in **30-100 words**.\n"
                f"Focus on the CODE LOGIC: Which variable is unchecked? What operation fails? \n"
                f"Do NOT just summarize the CVE description. Explain how the code enables the exploit."
            )
            messages.append({"role": "user", "content": user_msg_2})
            resp_2 = await self._call_llm(messages)
            messages.append({"role": "assistant", "content": resp_2})

            # === Turn 3: Safety Justification (Detailed Fix) ===
            # [修改] 强化指令：要求 30-100 词，基于具体修复
            user_msg_3 = (
                f"### Task: Safety Justification\n"
                f"Step 3: Explain the **Safety Justification** for the patch in **30-100 words**.\n"
                f"Quote the specific check added (e.g., `if (x > max)`). Explain logically why this prevents the vulnerability."
            )
            messages.append({"role": "user", "content": user_msg_3})
            resp_3 = await self._call_llm(messages)
            messages.append({"role": "assistant", "content": resp_3})

            # === Turn 4: Synthesis & Formatting ===
            user_msg_4 = (
                f"### Task: Final Output\n"
                f"Synthesize your analysis into a strictly formatted JSON object.\n"
                f"Ensure both 'root_cause' and 'safety_justification' are **30-100 words** and contain **specific code references**."
            )
            messages.append({"role": "user", "content": user_msg_4})

            # 使用 guided_json 保证格式
            final_json_str = await self._call_llm(messages, use_json=True)
            final_data = json.loads(final_json_str)

            # 注入结果
            item['root_cause'] = final_data.get('root_cause', resp_2)
            item['safety_justification'] = final_data.get('safety_justification', resp_3)
            item['quality_tag'] = "HIGH_MULTITURN"

            return item

        except Exception as e:
            if str(e) == "CONTEXT_OVERFLOW":
                logging.error(f"Skipping item due to massive context overflow.")
                item['root_cause'] = "[Skipped: Context Overflow]"
            else:
                logging.error(f"Error processing item: {str(e)}")
                item['root_cause'] = "[Analysis Failed]"

            item['safety_justification'] = "[Analysis Failed]"
            item['quality_tag'] = "FAIL"
            return item


# ================= 主程序 =================

async def main():
    # 1. 检查输入
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ================= [保留] 2. 断点续传：加载已处理的指纹 =================
    processed_hashes = set()
    if os.path.exists(OUTPUT_FILE):
        logging.info(f"Scanning existing output file: {OUTPUT_FILE} for resume...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    # 使用相同的哈希逻辑来识别已处理的记录
                    sig = generate_content_hash(record)
                    processed_hashes.add(sig)
                except json.JSONDecodeError:
                    pass
        logging.info(f"Found {len(processed_hashes)} already processed items. Resuming...")

    # 3. 加载并过滤数据
    logging.info(f"Loading data from {INPUT_FILE}...")
    data_items = []
    skipped_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)

                    # [核心逻辑] 检查是否已处理
                    item_hash = generate_content_hash(item)
                    if item_hash in processed_hashes:
                        skipped_count += 1
                        continue

                    data_items.append(item)
                except:
                    pass

    logging.info(f"Total items in input: {len(data_items) + skipped_count}")
    logging.info(f"Skipped (Already Processed): {skipped_count}")
    logging.info(f"Items remaining to process: {len(data_items)}")

    if not data_items:
        logging.info("All items have been processed. Exiting.")
        return

    # 4. 初始化并发控制
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    agent = DistillationAgent(client)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # [保留] 文件写入锁，确保多协程写入时不冲突
    file_lock = asyncio.Lock()

    # 5. 封装并发任务：处理完立即写入
    async def worker(item, pbar):
        async with semaphore:
            # 执行分析
            result_item = await agent.distill(item)

        # [核心逻辑] 实时写入文件 (Real-time Save)
        async with file_lock:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_item) + "\n")

        # 更新进度条
        pbar.update(1)
        return result_item

    logging.info(f"Starting Multi-Turn Distillation...")
    start_time = time.time()

    # 使用 tqdm 显示进度
    pbar = tqdm(total=len(data_items), desc="Distilling")
    tasks = [worker(item, pbar) for item in data_items]

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    pbar.close()

    # 6. 统计本次运行结果
    success_cnt = 0
    truncated_cnt = 0
    for item in results:
        if item.get('quality_tag') == "HIGH_MULTITURN":
            success_cnt += 1
        # 统计截断数据
        if item.get('truncation_warning') is True:
            truncated_cnt += 1

    duration = time.time() - start_time
    total_processed = len(data_items)

    logging.info("=" * 40)
    logging.info(f"Batch Complete.")
    logging.info(f"Time: {duration:.2f}s")
    if duration > 0:
        logging.info(f"Throughput: {total_processed / duration:.2f} items/s")

    if total_processed > 0:
        logging.info(
            f"Success Rate (Current Batch): {success_cnt}/{total_processed} ({success_cnt / total_processed * 100:.1f}%)")
        logging.info(
            f"Truncated Items (Current Batch): {truncated_cnt}/{total_processed} ({truncated_cnt / total_processed * 100:.1f}%)")
    logging.info("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())