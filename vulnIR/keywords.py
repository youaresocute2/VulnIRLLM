# -*- coding: utf-8 -*-
"""
keywords.py (VulnIR v3.4)

设计目标：
- 被 generator.py 复用的“敏感 API / sink 分类 / 参数角色规则”
- 保持精简但覆盖主流 CWE 与 C/C++ 常见危险点：
  * 内存拷贝/格式化输出（缓冲区溢出、格式化漏洞）
  * 分配/释放（泄漏、UAF、double free、整数溢出导致 size 错误）
  * I/O 与网络（长度/边界）
  * 环境变量（LD_PRELOAD 等）
  * 进程/命令（命令注入、PATH/argv）
  * 解析/反序列化类（strtol/atoi 等不严格解析导致 CWE-20）
  * crypto/认证（只做少量标记，避免膨胀）

注意：
- 这里的分类不是漏洞检测规则，而是 VulnIR “结构提示”的轻量先验。
- generator 会在 CALL 事件上输出 [TAG]，并按规则抽取关键变量（dst/src/len...）。
"""

from __future__ import annotations

from typing import Dict, Set, List


def NORMALIZE_CALLEE(name: str) -> str:
    """
    将 callee 名称做轻量归一：
    - 去掉前后空白
    - 将常见前缀去除（如 __builtin_memcpy）
    """
    n = (name or "").strip()
    if n.startswith("__builtin_"):
        n = n[len("__builtin_") :]
    return n


# ----------------------------
# API 分类（尽量精简）
# ----------------------------

API_CATEGORIES: Dict[str, Set[str]] = {
    # 典型“写入外部缓冲区”的 sink
    "MEMCPY": {
        "memcpy",
        "memmove",
        "bcopy",
        "mempcpy",
    },
    "STRCPY": {
        "strcpy",
        "stpcpy",
        "strncpy",
        "strcat",
        "strncat",
        "strlcpy",
        "strlcat",
    },
    "FORMAT": {
        "sprintf",
        "snprintf",
        "vsprintf",
        "vsnprintf",
        "asprintf",
        "vasprintf",
        "printf",
        "fprintf",
        "vprintf",
        "vfprintf",
    },
    "READ": {
        "read",
        "pread",
        "recv",
        "recvfrom",
        "fread",
        "gets",
        "fgets",
        "getline",
    },
    # 分配/释放相关（size 溢出、泄漏、UAF）
    "ALLOC": {
        "malloc",
        "calloc",
        "realloc",
        "aligned_alloc",
        "new",
        "operator new",
        "operator new[]",
        "kmalloc",
        "kzalloc",
        "vmalloc",
        "strdup",
        "strndup",
    },
    "FREE": {
        "free",
        "kfree",
        "vfree",
        "delete",
        "operator delete",
        "operator delete[]",
    },
    # I/O / 路径 / 命令
    "FILE": {
        "open",
        "fopen",
        "freopen",
        "close",
        "fclose",
        "write",
        "pwrite",
        "send",
        "sendto",
        "fwrite",
    },
    "EXEC": {
        "system",
        "popen",
        "execve",
        "execl",
        "execlp",
        "execv",
        "execvp",
    },
    # 环境变量
    "ENV": {
        "getenv",
        "setenv",
        "putenv",
        "unsetenv",
    },
    # 解析/转换（CWE-20 输入校验）
    "PARSE": {
        "atoi",
        "atol",
        "atoll",
        "strtol",
        "strtoll",
        "strtoul",
        "strtoull",
        "sscanf",
        "scanf",
        "fscanf",
    },
    # 少量认证/crypto 关键词（不扩张）
    "CRYPTO": {
        "memcmp",
        "EVP_DigestVerifyInit",
        "EVP_DigestVerifyFinal",
    },
}

# ----------------------------
# Sink 参数角色规则：抽取关键变量（dst/src/len...）
#   规则：类别 -> 需要抽取的参数位置列表
#   * 位置从 0 开始
# ----------------------------

SINK_RULES: Dict[str, List[int]] = {
    # memcpy(dst, src, n)
    "MEMCPY": [0, 1, 2],
    # strcpy(dst, src) / strncpy(dst, src, n) / strcat(dst, src) / strncat(dst, src, n)
    "STRCPY": [0, 1, 2],
    # snprintf(dst, size, fmt, ...) / sprintf(dst, fmt, ...)
    "FORMAT": [0, 1],
    # read(fd, buf, n) / recv(sock, buf, n, flags) / fgets(buf, n, fp) / getline(&buf, &n, fp)
    "READ": [0, 1, 2],
    # malloc(size) / calloc(n, size) / realloc(p, size)
    "ALLOC": [0, 1],
    # free(ptr)
    "FREE": [0],
    # open(path, ...) / fopen(path, mode) / write(fd, buf, n) / send(sock, buf, n, flags)
    "FILE": [0, 1, 2],
    # system(cmd) / execve(path, argv, envp)
    "EXEC": [0],
    # getenv(name) / setenv(name, val, overwrite)
    "ENV": [0, 1],
    # strtol(str, endp, base) / sscanf(buf, fmt, ...)
    "PARSE": [0, 1],
}
