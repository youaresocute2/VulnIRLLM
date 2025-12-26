#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VulnIR Generator (v3.4)

目标：
- 面向 LLM 的“语义保真 + token 友好”的漏洞中间表示（VulnIR）提取。
- 强调：guard→sink 链、关键数据流（def-use）与控制流锚点，而非逐行“翻译代码”。

输入：
- JSONL，每行一个样本，至少包含字段：{"idx":..., "func":...}
- 若样本已有 "vulnir" 字段，将在原文件中就地覆盖更新（in-place）。

输出：
- 在原 JSONL 文件中写回更新后的 "vulnir" 字段（不生成 .bak 备份）。
- 解析失败：写入 "VULNIR_PARSE_FAILED"；可解析但无事件：写入 "VULNIR_EMPTY"。

兼容性：
- tree-sitter Python bindings 的 API 在不同版本间存在差异：
  * 旧版：Parser(lang) 或 parser.set_language(lang)
  * 新版：parser.language = lang
本脚本在运行时自动探测并兼容。

用法：
  python vulnIR/generator.py -i ./dataset/primevul_train.jsonl
  python vulnIR/generator.py -i ./dataset/primevul_train.jsonl --max-events 220

依赖：
  pip install tree_sitter tree_sitter_c tree_sitter_cpp tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from tree_sitter import Language, Parser

try:
    import tree_sitter_c as tsc
except Exception:
    tsc = None

try:
    import tree_sitter_cpp as tscpp
except Exception:
    tscpp = None

# 关键词/敏感 API 由独立文件维护，便于复用与迭代
try:
    from keywords import API_CATEGORIES, SINK_RULES, NORMALIZE_CALLEE
except Exception:
    API_CATEGORIES = {}
    SINK_RULES = {}

    def NORMALIZE_CALLEE(x: str) -> str:
        return x


# ----------------------------
# 基础工具
# ----------------------------

_RE_PREPROC = re.compile(r"^\s*#.*?$", re.MULTILINE)


def _strip_preprocessor_preserve_lines(code: str) -> str:
    """删除/屏蔽预处理指令，但保持行号一致（用空行替换）。"""
    return _RE_PREPROC.sub("", code)


def _clean_syntax_light(code: str) -> str:
    """
    轻量清洗：尽量不改变行号/结构，仅修复显著影响解析的噪声。
    - 统一换行/制表符
    - 去除行尾不可见字符
    """
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.replace("\t", "    ")
    code = "\n".join([ln.rstrip("\x00").rstrip() for ln in code.split("\n")])
    return code


def _count_error_nodes(root) -> int:
    """统计 tree-sitter 中 ERROR/MISSING 节点数量，用于挑选更“干净”的语言解析结果。"""
    cnt = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            cnt += 1
        stack.extend(reversed(n.children))
    return cnt


def _make_parser(lang: Language) -> Parser:
    """兼容不同 tree-sitter Python bindings 的 Parser API。"""
    # 旧版：Parser(lang)
    try:
        p = Parser(lang)  # type: ignore[arg-type]
        return p
    except TypeError:
        pass

    p = Parser()
    # 旧版：parser.set_language(lang)
    if hasattr(p, "set_language"):
        try:
            p.set_language(lang)  # type: ignore[attr-defined]
            return p
        except Exception:
            pass
    # 新版：parser.language = lang
    if hasattr(p, "language"):
        p.language = lang  # type: ignore[attr-defined]
        return p

    raise RuntimeError("Unsupported tree_sitter.Parser API: cannot set language")


def _load_language(mod) -> Language:
    """
    tree_sitter_c / tree_sitter_cpp 的 language() 在不同版本可能返回：
    - tree_sitter.Language
    - PyCapsule / int (deprecated)
    """
    raw = mod.language()
    if isinstance(raw, Language):
        return raw
    return Language(raw)  # type: ignore[arg-type]


# ----------------------------
# AST → 变量/符号提取（避免 regex 误命中 "%d"、"\n" 等）
# ----------------------------

_SKIP_NODE_TYPES = {
    "string_literal",
    "char_literal",
    "raw_string_literal",
    "comment",
    "preproc_include",
    "preproc_def",
    "preproc_function_def",
    "preproc_if",
    "preproc_ifdef",
    "preproc_else",
    "preproc_elif",
    "preproc_endif",
    "number_literal",
}


def _node_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _shorten(s: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _first_child_of_type(node, types: Tuple[str, ...]):
    for c in node.children:
        if c.type in types:
            return c
    return None


def _extract_field_symbol(code_bytes: bytes, node) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    field_expression: 形如  base . field  或 base -> field
    返回： (base_sym, field_name, full_sym)
    """
    base = node.child_by_field_name("argument") or node.child_by_field_name("value") or (
        node.children[0] if node.children else None
    )
    field = node.child_by_field_name("field") or _first_child_of_type(node, ("field_identifier", "identifier"))
    if base is None or field is None:
        return None, None, None
    base_sym = _expr_to_symbol(code_bytes, base)
    field_name = _node_text(code_bytes, field)
    raw = _node_text(code_bytes, node)
    op = "->" if "->" in raw else "."
    full = f"{base_sym}{op}{field_name}" if base_sym and field_name else None
    return base_sym, field_name, full


def _expr_to_symbol(code_bytes: bytes, node) -> str:
    """
    将表达式节点压缩为“符号级”表示，用于 U{} 关键变量。
    - identifier => x
    - field_expression => obj->field / obj.field
    - declarator（声明 lhs） => 提取内部 identifier
    """
    t = node.type

    # declarator（用于声明语句的 lhs）：提取内部 identifier，避免 "*pw" 之类噪声
    if t in (
        "pointer_declarator",
        "array_declarator",
        "function_declarator",
        "parenthesized_declarator",
        "init_declarator",
        "type_declarator",
    ):
        for c in node.children:
            sym = _expr_to_symbol(code_bytes, c)
            if sym:
                return sym
        for c in node.children:
            if c.type == "identifier":
                return _node_text(code_bytes, c)
        return ""

    if t in _SKIP_NODE_TYPES:
        return ""
    if t == "identifier":
        return _node_text(code_bytes, node)
    if t == "field_expression":
        base, _, full = _extract_field_symbol(code_bytes, node)
        return full or base or _shorten(_node_text(code_bytes, node), 60)
    if t in ("parenthesized_expression", "pointer_expression", "unary_expression"):
        for c in node.children:
            sym = _expr_to_symbol(code_bytes, c)
            if sym:
                return sym
        return ""
    if t == "subscript_expression":
        base = node.child_by_field_name("argument") or (node.children[0] if node.children else None)
        idx = node.child_by_field_name("index") or (node.children[-1] if node.children else None)
        b = _expr_to_symbol(code_bytes, base) if base else ""
        i = _expr_to_symbol(code_bytes, idx) if idx else ""
        if b and i:
            return f"{b}[{i}]"
        return b or i
    return _shorten(_node_text(code_bytes, node), 60)


def collect_symbols(code_bytes: bytes, node) -> Set[str]:
    """
    从表达式节点中提取“变量/字段符号集合”：
    - 只依赖 AST 的 identifier / field_expression 结构
    - 严格跳过 string/char literal，避免 "%d" 等被当作变量
    """
    syms: Set[str] = set()

    def walk(n):
        if n.type in _SKIP_NODE_TYPES:
            return
        if n.type == "identifier":
            syms.add(_node_text(code_bytes, n))
            return
        if n.type == "field_expression":
            base_sym, _, full = _extract_field_symbol(code_bytes, n)
            if full:
                syms.add(full)
            if base_sym:
                syms.add(base_sym)
            base = n.child_by_field_name("argument") or n.child_by_field_name("value")
            if base is not None:
                walk(base)
            return
        for c in n.children:
            walk(c)

    if node is not None:
        walk(node)

    bad = {"NULL", "nullptr", "true", "false"}
    return {s for s in syms if s and s not in bad}


# ----------------------------
# Guard → checked var 提取（更“硬核”）
# ----------------------------

_CMP_OPS = {"<", "<=", ">", ">=", "==", "!="}


def _extract_operator_text(code_bytes: bytes, node) -> str:
    raw = _node_text(code_bytes, node)
    for op in ("<=", ">=", "==", "!=", "&&", "||", "<", ">", "!"):
        if op in raw:
            return op
    return ""


def extract_checked_and_bounds(code_bytes: bytes, cond_node) -> Tuple[Set[str], Set[str]]:
    """
    从 guard 条件里提取：
    - checked vars：被校验/被约束的变量（如 idx < len => idx；ptr == NULL => ptr）
    - bounds vars：边界/参照变量（如 idx < len => len）
    """
    checked: Set[str] = set()
    bounds: Set[str] = set()

    def is_null_like(n) -> bool:
        if n is None:
            return False
        if n.type in ("null",):
            return True
        txt = _node_text(code_bytes, n).strip()
        return txt in {"NULL", "nullptr", "0", "0u", "0U"}

    def walk(n):
        if n is None or n.type in _SKIP_NODE_TYPES:
            return

        if n.type == "unary_expression":
            raw = _node_text(code_bytes, n).strip()
            if raw.startswith("!"):
                for c in n.children:
                    if c.type in ("identifier", "field_expression", "parenthesized_expression", "subscript_expression"):
                        sym = _expr_to_symbol(code_bytes, c)
                        if sym:
                            checked.add(sym)
                        break
            for c in n.children:
                walk(c)
            return

        if n.type in ("binary_expression", "logical_expression"):
            op = _extract_operator_text(code_bytes, n)
            left = n.child_by_field_name("left") or (n.children[0] if n.children else None)
            right = n.child_by_field_name("right") or (n.children[-1] if n.children else None)

            if op in _CMP_OPS:
                left_syms = collect_symbols(code_bytes, left) if left else set()
                right_syms = collect_symbols(code_bytes, right) if right else set()

                if is_null_like(left) and right_syms:
                    checked.update(right_syms)
                    return
                if is_null_like(right) and left_syms:
                    checked.update(left_syms)
                    return

                if left and left.type == "number_literal" and right_syms:
                    checked.update(right_syms)
                    return
                if right and right.type == "number_literal" and left_syms:
                    checked.update(left_syms)
                    return

                if left_syms and right_syms:
                    if op in {"<", "<="}:
                        checked.update(left_syms)
                        bounds.update(right_syms)
                    elif op in {">", ">="}:
                        checked.update(right_syms)
                        bounds.update(left_syms)
                    else:
                        checked.update(left_syms)
                        checked.update(right_syms)
                    return

            for c in n.children:
                walk(c)
            return

        for c in n.children:
            walk(c)

    walk(cond_node)
    return checked, bounds


# ----------------------------
# VulnIR 事件抽象
# ----------------------------

@dataclass
class Event:
    line: int
    kind: str
    text: str
    defs: Set[str]
    uses: Set[str]
    key_uses: Set[str]
    tag: str = ""


def _line_of(node) -> int:
    return int(node.start_point[0]) + 1 if hasattr(node, "start_point") else 1


# ----------------------------
# 主提取器
# ----------------------------

class VulnIRExtractor:
    def __init__(self, max_events: int = 220, max_expr_chars: int = 120):
        if tsc is None or tscpp is None:
            raise RuntimeError("tree_sitter_c / tree_sitter_cpp not available. Please pip install tree_sitter_c tree_sitter_cpp")

        self.lang_c = _load_language(tsc)
        self.lang_cpp = _load_language(tscpp)

        self.parser_c = _make_parser(self.lang_c)
        self.parser_cpp = _make_parser(self.lang_cpp)

        self.max_events = max_events
        self.max_expr_chars = max_expr_chars

    def analyze(self, source_code: str) -> str:
        if not source_code or not source_code.strip():
            return "VULNIR_EMPTY"

        code = _clean_syntax_light(source_code)
        code = _strip_preprocessor_preserve_lines(code)
        code_bytes = code.encode("utf-8", errors="replace")

        tree_c = self.parser_c.parse(code_bytes)
        tree_cpp = self.parser_cpp.parse(code_bytes)

        err_c = _count_error_nodes(tree_c.root_node)
        err_cpp = _count_error_nodes(tree_cpp.root_node)
        tree = tree_c if err_c <= err_cpp else tree_cpp
        root = tree.root_node

        if _count_error_nodes(root) > 120:
            return "VULNIR_PARSE_FAILED"

        func_body = self._find_function_body(root)
        visit = func_body or root

        events: List[Event] = []
        dep: Dict[str, Set[str]] = {}
        last_def_line: Dict[str, int] = {}

        self._walk_extract(code_bytes, visit, events, dep, last_def_line)

        if not events:
            return "VULNIR_EMPTY"

        events = self._summarize_repeated_calls(events)

        rendered = self._render(events, dep, last_def_line)
        return rendered if rendered.strip() else "VULNIR_EMPTY"

    def _find_function_body(self, root):
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type == "function_definition":
                body = n.child_by_field_name("body")
                return body or n
            stack.extend(reversed(n.children))
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type == "compound_statement":
                return n
            stack.extend(reversed(n.children))
        return None

    def _walk_extract(
        self,
        code_bytes: bytes,
        node,
        events: List[Event],
        dep: Dict[str, Set[str]],
        last_def_line: Dict[str, int],
    ):
        if node is None:
            return
        if node.type == "ERROR":
            return

        if node.type == "if_statement":
            cond = node.child_by_field_name("condition")
            if cond is not None:
                checked, bounds = extract_checked_and_bounds(code_bytes, cond)
                uses = collect_symbols(code_bytes, cond)
                key = set(checked) | set(bounds)
                txt = _shorten(_node_text(code_bytes, cond), self.max_expr_chars)
                events.append(
                    Event(
                        line=_line_of(node),
                        kind="BRANCH",
                        text=f"if ({txt})",
                        defs=set(),
                        uses=uses,
                        key_uses=key,
                        tag="GUARD",
                    )
                )
            for c in node.children:
                self._walk_extract(code_bytes, c, events, dep, last_def_line)
            return

        if node.type in ("for_statement", "while_statement", "do_statement"):
            cond = node.child_by_field_name("condition")
            if cond is not None:
                checked, bounds = extract_checked_and_bounds(code_bytes, cond)
                uses = collect_symbols(code_bytes, cond)
                key = set(checked) | set(bounds)
                txt = _shorten(_node_text(code_bytes, cond), self.max_expr_chars)
                events.append(
                    Event(
                        line=_line_of(node),
                        kind="BRANCH",
                        text=f"{node.type.replace('_statement','')} ({txt})",
                        defs=set(),
                        uses=uses,
                        key_uses=key,
                        tag="GUARD",
                    )
                )
            for c in node.children:
                self._walk_extract(code_bytes, c, events, dep, last_def_line)
            return

        if node.type == "switch_statement":
            val = node.child_by_field_name("value") or node.child_by_field_name("condition")
            if val is not None:
                uses = collect_symbols(code_bytes, val)
                txt = _shorten(_node_text(code_bytes, val), self.max_expr_chars)
                events.append(
                    Event(
                        line=_line_of(node),
                        kind="SWITCH",
                        text=f"switch ({txt})",
                        defs=set(),
                        uses=uses,
                        key_uses=set(uses),
                        tag="CTRL",
                    )
                )
            for c in node.children:
                self._walk_extract(code_bytes, c, events, dep, last_def_line)
            return

        if node.type == "goto_statement":
            label = node.child_by_field_name("label")
            txt = _shorten(_node_text(code_bytes, label), 60) if label else "?"
            events.append(Event(line=_line_of(node), kind="GOTO", text=f"goto {txt}", defs=set(), uses=set(), key_uses=set(), tag="CTRL"))
            return

        if node.type == "return_statement":
            val = node.child_by_field_name("argument")
            uses = collect_symbols(code_bytes, val) if val else set()
            txt = _shorten(_node_text(code_bytes, val), self.max_expr_chars) if val else ""
            events.append(Event(line=_line_of(node), kind="RET", text=f"return {txt}".strip(), defs=set(), uses=uses, key_uses=set(uses), tag="CTRL"))
            return

        if node.type == "declaration":
            for c in node.children:
                if c.type == "init_declarator":
                    self._handle_init_declarator(code_bytes, c, events, dep, last_def_line)
            for c in node.children:
                self._walk_extract(code_bytes, c, events, dep, last_def_line)
            return

        if node.type == "expression_statement":
            expr = node.children[0] if node.children else None
            if expr is not None:
                handled = self._handle_expression(code_bytes, expr, events, dep, last_def_line)
                if handled:
                    for c in expr.children:
                        self._walk_extract(code_bytes, c, events, dep, last_def_line)
                    return

        if node.type == "call_expression":
            self._handle_call_expression(code_bytes, node, events, dep, last_def_line, lhs_defs=set())
            for c in node.children:
                self._walk_extract(code_bytes, c, events, dep, last_def_line)
            return

        if node.type == "subscript_expression":
            base = node.child_by_field_name("argument") or (node.children[0] if node.children else None)
            idx = node.child_by_field_name("index") or (node.children[-1] if node.children else None)
            b = _expr_to_symbol(code_bytes, base) if base else ""
            i = _expr_to_symbol(code_bytes, idx) if idx else ""

            idx_syms = collect_symbols(code_bytes, idx) if idx else set()
            if not idx_syms:
                # 常量下标通常噪声（如 a[0]），跳过事件但继续深入遍历
                for c in node.children:
                    self._walk_extract(code_bytes, c, events, dep, last_def_line)
                return

            uses = collect_symbols(code_bytes, node)
            key = set()
            if b:
                key.add(b)
            if i:
                key.add(i)
            txt = _shorten(_node_text(code_bytes, node), 80)
            events.append(Event(line=_line_of(node), kind="INDEX", text=txt, defs=set(), uses=uses, key_uses=key, tag="SINK"))

            for c in node.children:
                self._walk_extract(code_bytes, c, events, dep, last_def_line)
            return

        for c in node.children:
            self._walk_extract(code_bytes, c, events, dep, last_def_line)

        if len(events) > self.max_events * 3:
            return

    def _handle_init_declarator(self, code_bytes, node, events, dep, last_def_line):
        name_node = node.child_by_field_name("declarator")
        value_node = node.child_by_field_name("value")
        lhs = _expr_to_symbol(code_bytes, name_node) if name_node else ""
        if not lhs:
            return
        if value_node is None:
            return

        if value_node.type == "call_expression":
            self._handle_call_expression(code_bytes, value_node, events, dep, last_def_line, lhs_defs={lhs})
            uses = collect_symbols(code_bytes, value_node)
            dep[lhs] = set(uses)
            last_def_line[lhs] = _line_of(node)
            return

        rhs_txt = _shorten(_node_text(code_bytes, value_node), self.max_expr_chars)
        uses = collect_symbols(code_bytes, value_node)
        events.append(Event(line=_line_of(node), kind="SET", text=f"{lhs} = {rhs_txt}", defs={lhs}, uses=uses, key_uses={lhs} | set(list(uses)[:2]), tag="DF"))
        dep[lhs] = set(uses)
        last_def_line[lhs] = _line_of(node)

    def _handle_expression(self, code_bytes, expr, events, dep, last_def_line) -> bool:
        if expr.type == "assignment_expression":
            left = expr.child_by_field_name("left")
            right = expr.child_by_field_name("right")
            lhs = _expr_to_symbol(code_bytes, left) if left else ""
            if not lhs or right is None:
                return False

            op = _extract_operator_text(code_bytes, expr)
            if op not in {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="}:
                op = "="

            rhs_txt_raw = _node_text(code_bytes, right).strip()
            if self._is_trivial_update(lhs, op, rhs_txt_raw):
                return True

            if right.type == "call_expression":
                self._handle_call_expression(code_bytes, right, events, dep, last_def_line, lhs_defs={lhs})
                uses = collect_symbols(code_bytes, right)
                dep[lhs] = set(uses)
                last_def_line[lhs] = _line_of(expr)
                return True

            rhs_txt = _shorten(_node_text(code_bytes, right), self.max_expr_chars)
            uses = collect_symbols(code_bytes, right)
            text = f"{lhs} {op} {rhs_txt}" if op != "=" else f"{lhs} = {rhs_txt}"
            events.append(Event(line=_line_of(expr), kind="SET", text=text, defs={lhs}, uses=uses, key_uses={lhs} | set(list(uses)[:2]), tag="DF"))
            dep[lhs] = set(uses)
            last_def_line[lhs] = _line_of(expr)
            return True

        if expr.type == "update_expression":
            return True

        if expr.type == "call_expression":
            self._handle_call_expression(code_bytes, expr, events, dep, last_def_line, lhs_defs=set())
            return True

        return False

    def _is_trivial_update(self, lhs: str, op: str, rhs_txt: str) -> bool:
        lhs_esc = re.escape(lhs)
        rhs = re.sub(r"\s+", "", rhs_txt)
        if op in {"+=", "-="} and rhs in {"1", "1u", "1U"}:
            return True
        if op == "=":
            if re.fullmatch(rf"{lhs_esc}\+1(u|U)?", rhs) or re.fullmatch(rf"{lhs_esc}\-1(u|U)?", rhs):
                return True
        return False

    def _callee_name(self, code_bytes: bytes, call_node) -> str:
        fn = call_node.child_by_field_name("function")
        if fn is None and call_node.children:
            fn = call_node.children[0]
        if fn is None:
            return ""

        if fn.type == "identifier":
            return _node_text(code_bytes, fn)
        if fn.type == "field_expression":
            _, _, full = _extract_field_symbol(code_bytes, fn)
            if full:
                return full.split("->")[-1].split(".")[-1]
            return _shorten(_node_text(code_bytes, fn), 60).split(".")[-1]
        raw = _node_text(code_bytes, fn)
        if "::" in raw:
            return raw.split("::")[-1].strip()
        return _shorten(raw, 60)

    def _call_args(self, call_node) -> List:
        args = call_node.child_by_field_name("arguments")
        if args is None:
            for c in call_node.children:
                if c.type in ("argument_list", "arguments"):
                    args = c
                    break
        if args is None:
            return []
        real = [c for c in args.children if c.type not in {"(", ")", ","}]
        return real

    def _handle_call_expression(self, code_bytes, node, events, dep, last_def_line, lhs_defs: Set[str]):
        callee_raw = self._callee_name(code_bytes, node)
        callee = NORMALIZE_CALLEE(callee_raw)
        args = self._call_args(node)

        cat = ""
        for k, s in API_CATEGORIES.items():
            if callee in s:
                cat = k
                break

        key_vars: Set[str] = set()
        uses = set()
        for a in args:
            uses |= collect_symbols(code_bytes, a)

        if cat and cat in SINK_RULES:
            roles = SINK_RULES[cat]
            for pos in roles:
                if 0 <= pos < len(args):
                    sym = _expr_to_symbol(code_bytes, args[pos])
                    if sym:
                        key_vars.add(sym)
        else:
            for a in args[:2]:
                sym = _expr_to_symbol(code_bytes, a)
                if sym:
                    key_vars.add(sym)

        key_vars |= set(lhs_defs)
        defs = set(lhs_defs)

        if defs:
            txt = f"{callee}(...) -> {', '.join(sorted(defs))}"
        else:
            txt = f"{callee}(...)"

        tag = cat  # 非分类 call 不输出 [CALL]，省 token

        events.append(Event(line=_line_of(node), kind="CALL", text=txt, defs=defs, uses=uses, key_uses=key_vars if key_vars else set(uses), tag=tag))

        for d in defs:
            dep[d] = set(uses)
            last_def_line[d] = _line_of(node)

    def _summarize_repeated_calls(self, events: List[Event]) -> List[Event]:
        out: List[Event] = []
        i = 0
        while i < len(events):
            e = events[i]
            if e.kind != "CALL":
                out.append(e)
                i += 1
                continue

            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", e.text)
            callee = m.group(1) if m else e.text.split("(")[0]

            j = i + 1
            cnt = 1
            union_uses = set(e.uses)
            union_key = set(e.key_uses)
            union_defs = set(e.defs)
            tag = e.tag

            while j < len(events):
                ej = events[j]
                if ej.kind != "CALL":
                    break
                mj = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", ej.text)
                callee_j = mj.group(1) if mj else ej.text.split("(")[0]
                if callee_j != callee:
                    break
                cnt += 1
                union_uses |= ej.uses
                union_key |= ej.key_uses
                union_defs |= ej.defs
                j += 1

            if cnt >= 3:
                out.append(
                    Event(
                        line=e.line,
                        kind="CALL",
                        text=f"{callee}(...) ×{cnt}" + (f" -> {', '.join(sorted(union_defs))}" if union_defs else ""),
                        defs=union_defs,
                        uses=union_uses,
                        key_uses=union_key,
                        tag=tag,
                    )
                )
                i = j
            else:
                out.append(e)
                i += 1
        return out

    def _render(self, events: List[Event], dep: Dict[str, Set[str]], last_def_line: Dict[str, int]) -> str:
        lines: List[str] = []
        max_events = self.max_events

        truncated = len(events) > max_events
        events = events[:max_events]

        for e in events:
            seeds = set([s for s in e.key_uses if s])
            closure = self._symbol_closure(seeds, dep, depth=2)
            closure |= set(e.defs)

            key_vars = self._select_key_vars(e, seeds, closure)

            u_render = []
            for v in key_vars:
                if v in last_def_line and len(u_render) < 3:
                    u_render.append(f"{v}@{last_def_line[v]}")
                else:
                    u_render.append(v)

            d_render = sorted([d for d in e.defs if d])
            d_part = f" D{{{','.join(d_render)}}}" if d_render else ""
            u_part = f" U{{{','.join(u_render)}}}" if u_render else ""

            tag = f"[{e.tag}]" if e.tag and e.tag not in {"DF", "CTRL", "GUARD"} else ""
            lines.append(f"[Line {e.line}] {e.kind}{tag}: {e.text}{d_part}{u_part}")

        if truncated:
            lines.append("[Line ?] ...: VULNIR_TRUNCATED")

        return " | ".join(lines)

    def _symbol_closure(self, seeds: Set[str], dep: Dict[str, Set[str]], depth: int = 2) -> Set[str]:
        out = set([s for s in seeds if s])
        frontier = set(out)
        for _ in range(depth):
            nxt: Set[str] = set()
            for v in frontier:
                for u in dep.get(v, set()):
                    if u and u not in out:
                        nxt.add(u)
            out |= nxt
            frontier = nxt
            if not frontier:
                break
        return out

    def _select_key_vars(self, e: Event, seeds: Set[str], closure: Set[str]) -> List[str]:
        """
        关键变量数目（3~5）的选择依据：
        - 少于 3：容易丢失 guard/sink 的最小语义单元（dst/src/len 或 checked/bound）
        - 多于 5：token 开销上升且噪声显著增多
        - 因此：SINK/INDEX/GUARD 事件保留 5；普通 DF 事件保留 3~4
        """
        if e.tag in {"SINK"} or e.kind in {"INDEX"} or (e.tag and e.tag in API_CATEGORIES):
            k = 5
        elif e.kind == "BRANCH":
            k = 5
        else:
            k = 4 if len(seeds) >= 2 else 3

        def score(v: str) -> Tuple[int, int, str]:
            s = 0
            if v in seeds:
                s += 100
            if v in e.defs:
                s += 60
            if v in e.uses:
                s += 20
            s -= min(len(v), 40) // 10
            return (-s, len(v), v)

        cand = [v for v in closure if v and v not in {"NULL", "nullptr"}]
        cand_sorted = sorted(cand, key=score)

        picked: List[str] = []
        for v in cand_sorted:
            if v not in picked:
                picked.append(v)
            if len(picked) >= k:
                break

        if len(picked) < 3:
            for v in sorted([x for x in e.uses if x]):
                if v not in picked:
                    picked.append(v)
                if len(picked) >= 3:
                    break

        return picked[:k]


# ----------------------------
# JSONL I/O（in-place 更新，不生成 .bak）
# ----------------------------

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            data.append(json.loads(ln))
    return data


def write_jsonl_inplace(path: str, rows: List[Dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input jsonl path")
    ap.add_argument("--max-events", type=int, default=220, help="max events per sample")
    ap.add_argument("--max-expr-chars", type=int, default=120, help="max chars for expression snippet")
    args = ap.parse_args()

    extractor = VulnIRExtractor(max_events=args.max_events, max_expr_chars=args.max_expr_chars)

    rows = read_jsonl(args.input)

    updated = 0
    failed = 0
    for r in tqdm(rows, desc="VulnIR v3.4"):
        code = r.get("func", "") or ""
        old = r.get("vulnir", None)
        try:
            new = extractor.analyze(code)
        except Exception:
            new = "VULNIR_PARSE_FAILED"

        # 解析失败/为空：不要保留旧值，避免新旧版本“错位混杂”
        r["vulnir"] = new

        if new in {"VULNIR_PARSE_FAILED"}:
            failed += 1
        if old != new:
            updated += 1

    write_jsonl_inplace(args.input, rows)

    print("=== VulnIR v3.4 Done ===")
    print(f"Total: {len(rows)}")
    print(f"Updated(vulnir changed): {updated}")
    print(f"Failed(parse_failed): {failed}")


if __name__ == "__main__":
    main()
