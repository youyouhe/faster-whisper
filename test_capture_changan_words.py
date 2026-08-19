#!/usr/bin/env python3
"""捕获词级时间戳 JSON，供 test_subtitle_splitting.py --replay 离线重放。

需 GPU，每个音频只需跑一次；之后调切段逻辑无限次离线验证，不再占 GPU。

用法:
  CUDA_VISIBLE_DEVICES=0 ./faster-whisper-env/bin/python3 test_capture_changan_words.py changan_2min.wav
  追加 --float32 可用与生产完全一致的 float32 参数（需 ~4GB 空闲显存，建议空闲时段）

显存说明: 默认 int8 峰值约 1.3GB，可与生产实例同卡共存。int8 与 float32 的
词流几乎一致，宽度分布统计上等价；replay 的同词流新旧算法对比不受影响。

输出: benchmark_results/words_<basename>.json
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="音频文件路径（一个或多个）")
    ap.add_argument("--float32", action="store_true",
                    help="使用与生产完全一致的 float32 参数（显存 ~4GB）")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    compute_type = "float32" if args.float32 else "int8"
    print(f"加载 large-v3-turbo ({compute_type}) ...")
    t0 = time.time()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type=compute_type)
    print(f"模型加载完成 {time.time() - t0:.1f}s")

    out_dir = "benchmark_results"
    os.makedirs(out_dir, exist_ok=True)

    for path in args.audio:
        print(f"\n转录 {path} (language=zh, 参数与生产一致: beam5 + word_timestamps + VAD) ...")
        t1 = time.time()
        # 参数与 faster_whisper_api.py 生产调用逐字一致（除 language 显式 zh，
        # 省去 tiny 检测；中文音频检测结果即 zh）
        segments, info = model.transcribe(
            path,
            beam_size=5,
            word_timestamps=True,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        seg_list = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (s.words or [])
                ],
            }
            for s in segments
        ]
        elapsed = time.time() - t1

        out = {
            "audio": os.path.basename(path),
            "language": "zh",
            "duration": info.duration,
            "compute_type": compute_type,
            "segments": seg_list,
        }
        out_path = os.path.join(out_dir, f"words_{os.path.splitext(os.path.basename(path))[0]}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False)
        n_words = sum(len(s["words"]) for s in seg_list)
        print(f"✅ {len(seg_list)} segments / {n_words} words, "
              f"{info.duration:.0f}s 音频耗时 {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
