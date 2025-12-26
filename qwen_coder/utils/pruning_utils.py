import re
from typing import List, Tuple, Optional

# ------------------------------------------------------------
# VulnIR-guided pruning utilities
# ------------------------------------------------------------
# Design goals:
# 1) Prefer keeping vulnerability-relevant spans (from VulnIR anchors).
# 2) Always keep some function "skeleton" (signature/head + tail/cleanup).
# 3) Be budget-aware: if spans exceed token budget, shrink window first
#    (instead of immediately falling back to generic truncation).
# ------------------------------------------------------------


_VULNIR_LINE_PATTERNS = [
    # Common: "Line 123", "line: 123", "Line=123"
    r'\bline\b\s*[:=]?\s*(\d{1,6})\b',
    # Sometimes: "L123"
    r'\bL(\d{1,6})\b',
]


def extract_critical_lines(vulnir_str: str) -> List[int]:
    """
    Extract 1-based line numbers from VulnIR-like strings.

    Returns:
        List[int] of line numbers (1-based).
    """
    if not vulnir_str:
        return []

    hits: List[int] = []
    low = vulnir_str.lower()

    for pat in _VULNIR_LINE_PATTERNS:
        for m in re.finditer(pat, low):
            try:
                n = int(m.group(1))
                if 1 <= n <= 10**6:
                    hits.append(n)
            except Exception:
                continue

    # Deduplicate while preserving order (stable)
    seen = set()
    out = []
    for n in hits:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_interest_set(
    critical_lines_1based: List[int],
    total_lines: int,
    window_size: int,
    keep_head: int = 5,
    keep_tail: int = 3,
) -> List[int]:
    """
    Build interest indices (0-based) = skeleton + critical windows.
    """
    if total_lines <= 0:
        return []

    window_size = max(0, int(window_size))
    keep_head = max(0, int(keep_head))
    keep_tail = max(0, int(keep_tail))

    interest = set()

    # Skeleton
    for i in range(min(keep_head, total_lines)):
        interest.add(i)
    for i in range(max(0, total_lines - keep_tail), total_lines):
        interest.add(i)

    # Anchors
    for ln in critical_lines_1based:
        idx = ln - 1
        if idx < 0 or idx >= total_lines:
            continue
        s = max(0, idx - window_size)
        e = min(total_lines, idx + window_size + 1)
        for j in range(s, e):
            interest.add(j)

    return sorted(interest)


def reconstruct_code(source_lines: List[str], interest_indices: List[int]) -> str:
    """
    Reconstruct code with concise gap markers.
    """
    if not interest_indices:
        return ""

    parts: List[str] = []
    last = None
    for idx in interest_indices:
        if last is not None and idx > last + 1:
            parts.append("// ...")
        if 0 <= idx < len(source_lines):
            parts.append(source_lines[idx].rstrip())
        last = idx
    return "\n".join(parts)


def fallback_pruning(text: str, max_token_limit: int, tokenizer) -> str:
    """
    Fallback Strategy (no VulnIR / cannot fit):
      Head 20% + Middle 60% + Tail 20% (token-space).

    Note: uses token budget, not char budget.
    """
    max_token_limit = int(max_token_limit)
    if max_token_limit <= 0:
        return ""
    if not text:
        return ""

    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    if len(tokens) <= max_token_limit:
        return text

    budget = max(0, max_token_limit - 10)
    if budget <= 0:
        return ""

    head_len = max(1, int(budget * 0.20))
    tail_len = max(1, int(budget * 0.20))
    mid_len = max(1, budget - head_len - tail_len)

    mid_start = max(0, (len(tokens) - mid_len) // 2)

    head_ids = tokens[:head_len]
    mid_ids = tokens[mid_start: mid_start + mid_len]
    tail_ids = tokens[-tail_len:]

    return (
        tokenizer.decode(head_ids) +
        "\n// ...\n" +
        tokenizer.decode(mid_ids) +
        "\n// ...\n" +
        tokenizer.decode(tail_ids)
    )


def structured_ir_pruning(vulnir_str: str, max_token_limit: int, tokenizer) -> str:
    """
    Prune structured VulnIR / IR-like strings by keeping:
      Head 30% + Tail 30% + frequent keywords lines.
    """
    max_token_limit = int(max_token_limit)
    if max_token_limit <= 0:
        return ""
    if not vulnir_str:
        return ""

    tokens = tokenizer.encode(vulnir_str, truncation=False, add_special_tokens=False)
    if len(tokens) <= max_token_limit:
        return vulnir_str

    budget = max(0, max_token_limit - 10)
    if budget <= 0:
        return ""

    head_len = max(1, int(budget * 0.30))
    tail_len = max(1, int(budget * 0.30))
    mid_len = max(1, budget - head_len - tail_len)

    head_ids = tokens[:head_len]
    tail_ids = tokens[-tail_len:]

    # Mid: keep keywords (best-effort in text space)
    text = tokenizer.decode(tokens)
    lines = text.splitlines()
    keywords = ("call", "sink", "source", "taint", "line", "memcpy", "strcpy", "sprintf", "malloc", "free", "bounds")
    mid_lines = [ln for ln in lines if any(k in ln.lower() for k in keywords)]
    mid_text = "\n".join(mid_lines)
    mid_ids = tokenizer.encode(mid_text, truncation=False, add_special_tokens=False)
    if len(mid_ids) > mid_len:
        mid_ids = mid_ids[:mid_len]

    return (
        tokenizer.decode(head_ids) +
        "\n# ...\n" +
        tokenizer.decode(mid_ids) +
        "\n# ...\n" +
        tokenizer.decode(tail_ids)
    )


def _fit_with_shrinking_window(
    source_lines: List[str],
    critical_lines_1based: List[int],
    max_token_limit: int,
    tokenizer,
    window_size: int,
    keep_head: int,
    keep_tail: int,
    density_floor_ratio: float = 0.60,
) -> Optional[str]:
    """
    Try to fit VulnIR-guided pruning within token budget by shrinking window_size.
    Returns pruned code if fit, else None.
    """
    total_lines = len(source_lines)
    if total_lines == 0:
        return ""

    window = max(0, int(window_size))
    max_token_limit = int(max_token_limit)
    keep_head = int(keep_head)
    keep_tail = int(keep_tail)

    # Start from requested window, shrink if needed
    while window >= 0:
        interest = get_interest_set(
            critical_lines_1based=critical_lines_1based,
            total_lines=total_lines,
            window_size=window,
            keep_head=keep_head,
            keep_tail=keep_tail,
        )

        # Density guarantee: if too sparse, add some evenly-spaced lines
        # (helps long-range context, avoids "only islands" effect)
        approx_min_tokens = int(max_token_limit * density_floor_ratio)
        candidate_text = reconstruct_code(source_lines, interest)
        cand_tokens = tokenizer.encode(candidate_text, truncation=False, add_special_tokens=False)

        if len(cand_tokens) < approx_min_tokens and total_lines > 0:
            # add up to ~15% extra lines uniformly from the rest
            need = approx_min_tokens - len(cand_tokens)
            # heuristic: assume ~8 tokens/line average
            add_lines = max(0, min(total_lines, need // 8))
            if add_lines > 0:
                step = max(1, total_lines // add_lines)
                for i in range(0, total_lines, step):
                    interest.append(i)
                interest = sorted(set(interest))
                candidate_text = reconstruct_code(source_lines, interest)
                cand_tokens = tokenizer.encode(candidate_text, truncation=False, add_special_tokens=False)

        if len(cand_tokens) <= max_token_limit:
            return candidate_text

        window -= 1

    return None


def smart_code_pruning(
    source_code: str,
    vulnir_str: str,
    max_token_limit: int,
    tokenizer,
    window_size: int = 3,
    keep_head: int = 5,
    keep_tail: int = 3,
) -> str:
    """
    Budget-Aware Adaptive Pruning.

    - If VulnIR anchors exist: keep skeleton + (critical +/- window) spans.
      If it does not fit, shrink window first before falling back.
    - If no VulnIR: use token-space fallback pruning.

    Args:
        window_size: expansion radius around each critical line (in lines).
        keep_head/keep_tail: always keep these lines for context.
    """
    if not source_code:
        return ""

    max_token_limit = int(max_token_limit)
    if max_token_limit <= 0:
        return ""

    # Fast budget check (token-accurate)
    tokens = tokenizer.encode(source_code, truncation=False, add_special_tokens=False)
    if len(tokens) <= max_token_limit:
        return source_code

    # If VulnIR is empty / no anchors => fallback
    critical = extract_critical_lines(vulnir_str)
    if not critical:
        return fallback_pruning(source_code, max_token_limit, tokenizer)

    # Guided pruning with adaptive shrinking window
    source_lines = source_code.splitlines()
    fit = _fit_with_shrinking_window(
        source_lines=source_lines,
        critical_lines_1based=critical,
        max_token_limit=max_token_limit,
        tokenizer=tokenizer,
        window_size=window_size,
        keep_head=keep_head,
        keep_tail=keep_tail,
    )
    if fit is not None and fit.strip():
        return fit

    # Last resort
    return fallback_pruning(source_code, max_token_limit, tokenizer)
