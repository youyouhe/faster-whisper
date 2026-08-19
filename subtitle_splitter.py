#!/usr/bin/env python3
"""共享字幕切段策略 —— faster_whisper_api.py 与 qwen_whisper_api.py 共用

按"显示宽度"切字幕段，替代旧的按词数/字数切段：
  - 显示宽度：CJK 全宽字符计 1，ASCII/空格/半角标点计 0.5
    （同一上限对中文约 15 字、对英文约 30 字符，一个环境变量同时管两种语言）
  - 断点优先级：句末标点 > 从句标点/较大停顿 > 硬切兜底；绝不切开一个 token 内部
  - 微块抑制：宽度 < MIN_BLOCK_WIDTH 或时长 < MIN_BLOCK_DURATION 的块并入前一块，
    消除"星期/五"式孤块
  - 输出块时间严格单调（next.start >= prev.end），保证 srt_merger 不会把
    相邻块重新拼接成长行（srt_merger._remove_overlaps 对任何 1ms 重叠都会拼接）

对两类输入通用：
  - whisper 后端的词级 token（中文 1-4 字一个词、英文带前导空格）
  - qwen 后端的字级 item（标点已由 merge_punctuation 附着到前字）

环境变量：
  MAX_CHARS_PER_SEGMENT  字幕段显示宽度上限，默认 15（两个后端同名同默认，保持对齐）
"""

import os
import re
import unicodedata
from typing import List, Optional, Tuple

MAX_CHARS_PER_SEGMENT = float(os.getenv("MAX_CHARS_PER_SEGMENT", "15"))
HARD_FLUSH_MARGIN = 5.0      # 无可用断点时的硬切阈值（宽度单位）
MIN_BLOCK_WIDTH = 3.0        # 宽度低于此值的块并入前一块
MIN_BLOCK_DURATION = 0.5     # 时长低于此值（秒）的块并入前一块
MICRO_MERGE_MAX_GAP = 1.0    # 与前块间隔超过此值（秒）不并，避免字幕跨长静音挂屏

SENT_END_PUNCT = "。！？；…!?."   # 句末标点：出现即断（含英文句点）
CLAUSE_PUNCT = "，、：,.—;"       # 从句标点：超限时优先在此处断

Block = Tuple[float, float, str]          # (start, end, text)
Token = Tuple[float, float, str]          # (start, end, token_text)


def clean_text(text):
    """Clean text by removing extra spaces and normalizing punctuation
    （自 faster_whisper_api.py 迁移，两后端共用同一实现）"""
    if not text:
        return text
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([,.!?;:，。！？；：])', r'\1', text)
    # Remove extra spaces between Chinese characters and punctuation
    text = re.sub(r'([^\s])\s+([,.!?;:，。！？；：])', r'\1\2', text)
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def display_width(text: str) -> float:
    """显示宽度：CJK 全宽字符计 1，其余（ASCII/空格/半角标点）计 0.5"""
    return sum(1.0 if unicodedata.east_asian_width(ch) in "WF" else 0.5
               for ch in text)


def _last_char(token_text: str) -> str:
    t = token_text.rstrip()
    return t[-1] if t else ""


def _starts_with_punct(token_text: str) -> bool:
    t = token_text.lstrip()
    return bool(t) and t[0] in SENT_END_PUNCT + CLAUSE_PUNCT


def _best_cut(cur: List[int], cur_w: float, tokens: List[Token],
              last_token: bool) -> Optional[int]:
    """在当前缓冲 cur（token 下标列表）内找最佳断点。
    返回 cut：保留 cur[:cut] 作为一块，cur[cut:] 留给下一块；无可用断点返回 None。
    候选必须 head_w <= MAX；打分 (从句标点结尾, 断点处停顿时长, head_w) 取最大。"""
    best, best_score = None, None
    head_w = 0.0
    for k in range(len(cur) - 1):                 # 尾部至少保留 1 个 token
        head_w += display_width(tokens[cur[k]][2])
        if head_w > MAX_CHARS_PER_SEGMENT:
            break
        if _starts_with_punct(tokens[cur[k + 1]][2]):
            continue                              # 不把标点切到下一块开头
        if last_token and cur_w - head_w < MIN_BLOCK_WIDTH:
            continue                              # 段末不留孤儿尾巴
        gap = tokens[cur[k + 1]][0] - tokens[cur[k]][1]
        punct = 1.0 if _last_char(tokens[cur[k]][2]) in CLAUSE_PUNCT else 0.0
        score = (punct, gap, head_w)
        if best_score is None or score > best_score:
            best, best_score = k + 1, score
    return best


def split_tokens(tokens: List[Token]) -> List[Block]:
    """把词级/字级时间戳流切成 <= MAX_CHARS_PER_SEGMENT 宽度的字幕块。

    tokens: [(start, end, text), ...] 按时间升序。返回 [(start, end, text)]，
    块间时间严格单调。超宽且无断点的缓冲在 MAX + HARD_FLUSH_MARGIN 处硬切兜底；
    单个超长 token（宽度已超上限）整体输出、绝不内切。"""
    blocks: List[Block] = []
    cur: List[int] = []
    cur_w = 0.0
    prev_end: Optional[float] = None
    n = len(tokens)

    def emit(idxs: List[int]) -> None:
        nonlocal prev_end
        start = tokens[idxs[0]][0]
        end = tokens[idxs[-1]][1]
        if prev_end is not None:
            start = max(start, prev_end)          # 单调钳位
        end = max(end, start + 0.001)
        text = clean_text("".join(tokens[i][2] for i in idxs))
        if text:
            blocks.append((start, end, text))
            prev_end = end

    for i in range(n):
        cur.append(i)
        cur_w += display_width(tokens[i][2])

        if _last_char(tokens[i][2]) in SENT_END_PUNCT:
            emit(cur)                              # 句末标点：直接断
            cur, cur_w = [], 0.0
            continue

        if cur_w < MAX_CHARS_PER_SEGMENT:
            continue

        cut = _best_cut(cur, cur_w, tokens, last_token=(i == n - 1))
        if cut is not None:                        # 最佳断点处断
            emit(cur[:cut])
            cur = cur[cut:]
            cur_w = sum(display_width(tokens[j][2]) for j in cur)
        elif cur_w >= MAX_CHARS_PER_SEGMENT + HARD_FLUSH_MARGIN:
            emit(cur)                              # 硬切兜底（单 token 过长时允许超宽）
            cur, cur_w = [], 0.0

    if cur:
        emit(cur)
    return blocks


def suppress_micro_blocks(blocks: List[Block]) -> List[Block]:
    """微块抑制：宽度/时长过小的块并入前一块（时间取并集，文本相接）。
    仅当前块确实过小、与前块间隔 <= MICRO_MERGE_MAX_GAP、且合并后宽度不超过
    MAX + HARD_FLUSH_MARGIN 时才合并（防止连续短句被无上限地叠进同一块）。"""
    out: List[List] = []
    for start, end, text in blocks:
        tiny = display_width(text) < MIN_BLOCK_WIDTH or (end - start) < MIN_BLOCK_DURATION
        if (tiny and out
                and start - out[-1][1] <= MICRO_MERGE_MAX_GAP
                and display_width(out[-1][2]) + display_width(text)
                    <= MAX_CHARS_PER_SEGMENT + HARD_FLUSH_MARGIN):
            prev = out[-1]
            sep = " " if prev[2][-1:].isascii() and text[:1].isascii() else ""
            prev[2] = clean_text(prev[2] + sep + text)
            prev[1] = end
        else:
            out.append([start, end, text])
    return [(s, e, t) for s, e, t in out]
