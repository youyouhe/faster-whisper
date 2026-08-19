#!/usr/bin/env python3
"""subtitle_splitter.py 的纯逻辑测试（无 GPU，python3 直接跑）。

用法:
  python3 test_subtitle_splitting.py                      # 跑全部合成用例
  python3 test_subtitle_splitting.py --replay words.json  # 重放真实词级时间戳
      --baseline pub_round_1.json                         # 与基准 SRT 对比宽度分布

合成用例覆盖：中文无标点长句 / 中文标点断句 / 英文长句 / 微块孤块 / 连续短句 /
超长单 token / 空与纯标点 token / 词时间戳轻微重叠 / srt_merger 往返（证明
块时间严格单调、merger 不会把切好的块重新拼成长行）。
"""

import argparse
import json
import re
import statistics
import sys

from subtitle_splitter import (
    MAX_CHARS_PER_SEGMENT, HARD_FLUSH_MARGIN, display_width,
    split_tokens, suppress_micro_blocks, clean_text,
)
from srt_merger import SRTMerger

FAILURES = []


def run_case(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
    except AssertionError as e:
        FAILURES.append(name)
        print(f"  ❌ {name}: {e}")
    except Exception as e:
        FAILURES.append(name)
        print(f"  💥 {name}: {type(e).__name__}: {e}")


def format_ts(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def blocks_to_srt(blocks):
    lines = []
    for i, (start, end, text) in enumerate(blocks, 1):
        lines.append(str(i))
        lines.append(f"{format_ts(start)} --> {format_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


NORM_RE = re.compile(r'[\s"\'“”‘’…—\-·,，。！？；：、.!?;:]+')


def norm(s):
    return NORM_RE.sub("", s)


def assert_monotonic(blocks, ctx=""):
    for a, b in zip(blocks, blocks[1:]):
        assert b[0] >= a[1] - 1e-9, \
            f"{ctx} 时间重叠: {a!r} -> {b!r} (start {b[0]} < prev end {a[1]})"


def assert_widths(blocks, ctx=""):
    for s, e, t in blocks:
        w = display_width(t)
        assert w <= MAX_CHARS_PER_SEGMENT + HARD_FLUSH_MARGIN, \
            f"{ctx} 块超宽 {w}: {t!r}"
        assert t.strip(), f"{ctx} 空文本块: {(s, e, t)!r}"


def assert_text_preserved(tokens, blocks, ctx=""):
    src = norm("".join(t for _, _, t in tokens))
    dst = norm("".join(t for _, _, t in blocks))
    assert src == dst, f"{ctx} 文本丢失/多出:\n  源: {src!r}\n  出: {dst!r}"


def merger_round_trip(blocks, ctx=""):
    """格式化 → parse → _remove_overlaps，块数必须不变（否则 merger 会回拼）。"""
    merger = SRTMerger(overlap_seconds=0.0)
    parsed = merger.parse_srt_content(blocks_to_srt(blocks), 0, 0.0)
    merged = merger._remove_overlaps(parsed)
    assert len(merged) == len(blocks), \
        f"{ctx} merger 回拼: {len(blocks)} 块 -> {len(merged)} 块"


# ---------------------------------------------------------------- 合成用例

def make_tokens(texts, step=0.6, dur=0.5):
    """把 token 文本列表变成均匀间隔的 (start, end, text)，词间停顿 step 秒。"""
    tokens = []
    t = 0.0
    for txt in texts:
        tokens.append((t, t + dur, txt))
        t += dur + step
    return tokens


def case_chinese_no_punct():
    text = "今天天气很好我们一起去公园散步然后吃饭聊天很开心大家都很高兴的样子"
    toks = [text[i:i + 2] for i in range(0, len(text), 2)]  # 16 个双字 token
    tokens = make_tokens(toks)
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert len(blocks) > 1, "40 字长句应被切成多块"
    assert_widths(blocks, "中文长句")
    assert_text_preserved(tokens, blocks, "中文长句")
    assert_monotonic(blocks, "中文长句")
    merger_round_trip(blocks, "中文长句")


def case_chinese_punctuated():
    texts = ["我该", "从何", "说起，", "或许", "就从", "我以为", "人生，",
             "最得意", "的那天", "开始吧。", "那天", "我终于", "签下了",
             "长安城", "的房契，", "我叫", "李善德，", "一个", "以名算",
             "科技地", "在九品", "关节上", "熬了", "半辈子。"]
    tokens = make_tokens(texts)
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert_widths(blocks, "中文标点")
    assert_text_preserved(tokens, blocks, "中文标点")
    assert_monotonic(blocks, "中文标点")
    punct = "，。！？；、,.!?:;…"
    for _, _, t in blocks:
        assert not (t.strip() and t.strip()[0] in punct), f"标点落到块开头: {t!r}"
    # 句末标点 token 必须位于所在块的结尾
    for s, e, t in blocks:
        assert "。" not in t[:-1], f"句号不在块尾: {t!r}"
    merger_round_trip(blocks, "中文标点")


def case_english():
    words = ("The quick brown fox jumps over the lazy dog and runs away "
             "into the dark forest where nobody can find it any more today").split()
    toks = [(" " if i else "") + w for i, w in enumerate(words)]
    toks[8] = "dog."  # 句点应立即触发断句
    tokens = make_tokens(toks, step=0.05, dur=0.2)
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert len(blocks) > 1, "英文长句应被切成多块"
    assert_widths(blocks, "英文")
    assert_text_preserved(tokens, blocks, "英文")
    assert_monotonic(blocks, "英文")
    for _, _, t in blocks:
        assert t == t.strip(), f"行首/尾空格: {t!r}"
        assert "  " not in t, f"双空格: {t!r}"
    # 不切词：每块必须由整词组成（去掉空格后是原文连续子串，已在文本保全中断言）
    merger_round_trip(blocks, "英文")


def case_micro_block_orphan():
    # 复刻 result_189.srt 的 "星期/五" 孤块：句号断句后尾随一个 0.16s 的单字
    texts = ["各位", "網友", "大家", "好，", "今天", "是十", "月十", "六號", "星期"]
    tokens = make_tokens(texts, step=0.3, dur=0.4)
    t_end = tokens[-1][1]
    tokens.append((t_end + 0.04, t_end + 0.20, "五。"))  # 0.16s 孤块
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert_text_preserved(tokens, blocks, "微块")
    assert_monotonic(blocks, "微块")
    for s, e, t in blocks:
        w = display_width(t)
        assert w >= 3.0, f"残留孤块(宽 {w}): {t!r}"
    merger_round_trip(blocks, "微块")


def case_rapid_short_sentences():
    tokens = []
    t = 0.0
    for _ in range(6):
        tokens.append((t, t + 0.3, "好。"))
        t += 0.35
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert_text_preserved(tokens, blocks, "连续短句")
    assert_widths(blocks, "连续短句")  # 合并不得超 MAX+HARD_FLUSH_MARGIN
    assert_monotonic(blocks, "连续短句")
    merger_round_trip(blocks, "连续短句")


def case_single_long_token():
    long_word = "这是一个非常非常非常长的没有任何标点的超长词组"
    tokens = [(0.0, 4.0, long_word)]
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert len(blocks) == 1, f"超长单 token 应整体输出: {blocks!r}"
    assert blocks[0][2] == long_word, "超长 token 内容被改动"
    merger_round_trip(blocks, "超长token")


def case_empty_and_punct_tokens():
    tokens = [(0.0, 0.3, ""), (0.3, 0.6, " "), (0.6, 0.9, "，"),
              (0.9, 1.5, "你好。"), (1.5, 1.8, "  ")]
    blocks = suppress_micro_blocks(split_tokens(tokens))
    for s, e, t in blocks:
        assert t.strip(), f"空文本块: {(s, e, t)!r}"
    assert_monotonic(blocks, "空token")


def case_overlapping_word_stamps():
    # 词时间戳轻微重叠（whisper 偶发）：切分后块仍必须严格单调
    texts = ["今天", "天气", "很好", "我们", "一起", "去公园", "散步", "然后",
             "吃饭", "聊天", "开心", "大家", "都很", "高兴", "的样子", "继续",
             "说话", "聊天"]
    tokens = []
    t = 0.0
    for txt in texts:
        tokens.append((t, t + 0.55, txt))  # dur 0.55 > step 0.5 → 相邻词重叠 0.05s
        t += 0.5
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert_text_preserved(tokens, blocks, "重叠时间戳")
    assert_monotonic(blocks, "重叠时间戳")
    merger_round_trip(blocks, "重叠时间戳")


def case_qwen_char_items():
    # 模拟 qwen 后端 merge_punctuation 后的字级 item：标点附着前字、每字一个时间戳
    text = "我该从何说起，或许就从我以为人生，最得意的那天开始吧。那天我终于签下了长安城的房契。"
    tokens = []
    t = 0.0
    for ch in text:
        dur = 0.12 if ch not in "，。" else 0.02   # 标点几乎不占时
        tokens.append((t, t + dur, ch))
        t += dur + 0.18
    blocks = suppress_micro_blocks(split_tokens(tokens))
    assert_widths(blocks, "字级item")
    assert_text_preserved(tokens, blocks, "字级item")
    assert_monotonic(blocks, "字级item")
    punct = "，。"
    for _, _, txt in blocks:
        assert not (txt.strip() and txt.strip()[0] in punct), f"标点落到块开头: {txt!r}"
        assert "。" not in txt[:-1], f"句号不在块尾: {txt!r}"
    merger_round_trip(blocks, "字级item")


def case_width_metric():
    assert display_width("中") == 1.0
    assert display_width("a") == 0.5
    assert display_width("，") == 1.0   # U+FF0C 全宽
    assert display_width("。") == 1.0   # U+3002
    assert display_width("...") == 1.5
    assert display_width("abc中def") == 4.0
    assert display_width("") == 0.0
    # clean_text（生产版原样迁移）只去标点前的空格，标点后空格保留
    assert clean_text("你好 ， 世界 。") == "你好， 世界。"
    assert clean_text("今天 天气  很好") == "今天 天气 很好"


# ---------------------------------------------------------------- 重放模式

def pipeline_like_whisper(segments):
    """与 faster_whisper_api.py 新块生成循环一致的处理流水线。"""
    all_blocks = []
    prev_end = None
    for seg in segments:
        if seg.get("words"):
            tokens = [(w["start"], w["end"], w["word"]) for w in seg["words"]]
            blocks = split_tokens(tokens)
        else:
            text = clean_text(seg.get("text", ""))
            blocks = [(seg["start"], seg["end"], text)] if text else []
        for start, end, text in blocks:
            if prev_end is not None and start < prev_end:
                start = prev_end
            end = max(end, start + 0.001)
            prev_end = end
            all_blocks.append((start, end, text))
    return suppress_micro_blocks(all_blocks)


def pipeline_old_whisper(segments, max_words=15):
    """复刻 faster_whisper_api.py 改造前的旧算法（固定 15 词切分 + 整段直出），
    用于同一份词流上的新旧对比。"""
    blocks = []
    for seg in segments:
        words = seg.get("words") or []
        if words and len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = words[i:i + max_words]
                text = clean_text("".join(w["word"] for w in chunk))
                if text.strip():
                    blocks.append((chunk[0]["start"], chunk[-1]["end"], text))
        else:
            text = clean_text(seg.get("text", ""))
            if text.strip():
                blocks.append((seg["start"], seg["end"], text))
    return blocks


def width_stats(block_texts):
    widths = sorted(display_width(t) for t in block_texts)
    n = len(widths)
    return {
        "count": n,
        "median": widths[n // 2] if n else 0,
        "p90": widths[int(n * 0.9)] if n else 0,
        "max": widths[-1] if n else 0,
        "gt16": sum(1 for w in widths if w > 16),
        "gt16_pct": 100.0 * sum(1 for w in widths if w > 16) / n if n else 0,
    }


def parse_srt_texts(srt_text):
    pattern = r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n([^\n]*)'
    return re.findall(pattern, srt_text.strip())


def do_replay(words_json, baseline_json):
    with open(words_json) as f:
        capture = json.load(f)
    blocks = pipeline_like_whisper(capture["segments"])

    src = norm("".join(w["word"] for seg in capture["segments"] for w in seg["words"]))
    dst = norm("".join(t for _, _, t in blocks))
    assert src == dst, f"重放文本不一致:\n  源: {src[:80]!r}\n  出: {dst[:80]!r}"
    assert_monotonic(blocks, "重放")
    assert_widths(blocks, "重放")
    merger_round_trip(blocks, "重放")

    after = width_stats([t for _, _, t in blocks])
    old = pipeline_old_whisper(capture["segments"])
    before = width_stats([t for _, _, t in old])
    print(f"\n重放 {words_json} ({capture.get('audio', '?')}, "
          f"{capture.get('compute_type', '?')}):")
    print(f"\n  同词流新旧算法对比:")
    print(f"  {'指标':<10}{'旧算法':>8}{'新算法':>8}")
    for k, label in [("count", "块数"), ("median", "中位宽"), ("p90", "p90"),
                     ("max", "最大宽"), ("gt16", ">16块数"), ("gt16_pct", ">16占比%")]:
        print(f"  {label:<10}{before[k]:>8}{after[k]:>8}")

    if baseline_json:
        with open(baseline_json) as f:
            base = json.load(f)
        base_stats = width_stats(parse_srt_texts(base["data"]))
        print(f"\n  参考基准 {baseline_json} (生产 float32 实测):")
        print(f"  {'指标':<10}{'基准':>8}{'新算法':>8}")
        for k, label in [("count", "块数"), ("median", "中位宽"), ("p90", "p90"),
                         ("max", "最大宽"), ("gt16", ">16块数"), ("gt16_pct", ">16占比%")]:
            print(f"  {label:<10}{base_stats[k]:>8}{after[k]:>8}")

    micro = sum(1 for s, e, t in blocks
                if display_width(t) < 3.0 or (e - s) < 0.5)
    print(f"\n  微块(宽<3 或 时长<0.5s)残留: {micro}")
    print("\n  ✅ 重放校验通过（文本无丢失、时间单调、merger 不回拼）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="重放 test_capture_changan_words.py 捕获的词级时间戳 JSON")
    ap.add_argument("--baseline", help="基准 SRT JSON（pub_round_*.json，含 data 字段）")
    args = ap.parse_args()

    if args.replay:
        return do_replay(args.replay, args.baseline)

    print(f"subtitle_splitter 纯逻辑测试 (MAX_CHARS_PER_SEGMENT={MAX_CHARS_PER_SEGMENT})\n")
    run_case("中文无标点长句", case_chinese_no_punct)
    run_case("中文标点断句", case_chinese_punctuated)
    run_case("英文长句", case_english)
    run_case("微块孤块(星期/五)", case_micro_block_orphan)
    run_case("连续短句合并上限", case_rapid_short_sentences)
    run_case("超长单 token 不内切", case_single_long_token)
    run_case("空/纯标点 token", case_empty_and_punct_tokens)
    run_case("词时间戳轻微重叠", case_overlapping_word_stamps)
    run_case("字级item(qwen对齐器)", case_qwen_char_items)
    run_case("宽度计量", case_width_metric)

    if FAILURES:
        print(f"\n❌ {len(FAILURES)} 个用例失败: {FAILURES}")
        return 1
    print("\n✅ 全部用例通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
