#!/usr/bin/env python3
"""
Qwen3-ASR FastAPI 后端 —— faster_whisper_api.py 的 drop-in 替代品

接口契约与 faster_whisper_api.py 完全一致：
  POST /inference  (multipart: file, form: response_format=srt, language=auto)
                   → {"code":0,"msg":"ok","data":<SRT文本>}
  GET  /health     → {"status":"healthy"}
  GET  /stats      → 实例统计

与 whisper 后端的差异（GTX 1080 / 8GB 适配，详见 benchmark 结论）：
  - ASR: Qwen3-ASR-0.6B，fp32 + sdpa（Pascal 不支持 bf16，fp16 计算 1:64 残废）
  - 时间戳: Qwen3-ForcedAligner-0.6B（fp16，非自回归单趟，可接受），字级时间戳
  - 显存: 音频预切 120 秒块；对齐器块长补丁到 60 秒（官方默认 1200s/180s 必 OOM）
  - 两模型同卡峰值约 5.3-6.3GB，需独占一张 8GB 卡（本机用闲置的 GPU 2）

环境变量：
  API_PORT          默认 5004（pilot 端口，5002/5003 是 whisper 生产实例）
  INSTANCE_ID       默认 qwen-gpu2
  QWEN_ASR_DIR      ASR 模型目录，默认 models/Qwen3-ASR-0.6B
  QWEN_ALIGNER_DIR  对齐器模型目录，默认 models/Qwen3-ForcedAligner-0.6B
  QWEN_CHUNK_SEC    ASR 切块秒数，默认 120（8GB 卡 fp32 安全上限）
  MAX_QUEUE_SIZE    默认 100
  MAX_CHARS_PER_SEGMENT  字幕段显示宽度上限，默认 15（CJK=1/ASCII=0.5，
                        与 whisper 后端共用，见 subtitle_splitter.py）
  API_KEY           与现有服务共用；未设置则开放访问
"""

import asyncio
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from auth_middleware import require_auth

# 共享字幕切段策略（与 whisper 后端对齐）
from subtitle_splitter import split_tokens, suppress_micro_blocks

# ---------------------------------------------------------------- 配置

API_PORT = int(os.getenv("API_PORT", "5004"))
INSTANCE_ID = os.getenv("INSTANCE_ID", "qwen-gpu2")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ASR_DIR = os.getenv("QWEN_ASR_DIR", os.path.join(REPO_DIR, "models", "Qwen3-ASR-0.6B"))
ALIGNER_DIR = os.getenv("QWEN_ALIGNER_DIR", os.path.join(REPO_DIR, "models", "Qwen3-ForcedAligner-0.6B"))
CHUNK_SEC = float(os.getenv("QWEN_CHUNK_SEC", "120"))
ALIGN_CHUNK_SEC = float(os.getenv("QWEN_ALIGN_CHUNK_SEC", "60"))

# faster-whisper ISO 语言码 → Qwen 语言名；auto → None（自动检测）
LANGUAGE_MAP = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English",
    "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German",
    "es": "Spanish", "ru": "Russian",
}

# 切段策略（MAX_CHARS_PER_SEGMENT / 标点集 / 微块抑制阈值）统一在 subtitle_splitter.py，
# 与 whisper 后端共用同一环境变量与默认值（默认 15 显示宽度：CJK=1, ASCII=0.5）

# ---------------------------------------------------------------- 统计

@dataclass
class ProcessingStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_files_processed: int = 0
    total_file_size_mb: float = 0.0
    total_audio_seconds: float = 0.0
    total_processing_time: float = 0.0
    instance_id: str = ""
    gpu_device: str = ""
    port: int = 0
    start_time: float = field(default_factory=time.time)

stats = ProcessingStats()
stats_lock = threading.Lock()

# ---------------------------------------------------------------- 模型

model = None  # Qwen3ASRModel 全局实例


def load_model():
    """加载 ASR + ForcedAligner，并打显存安全补丁"""
    global model
    import torch
    from qwen_asr import Qwen3ASRModel
    import qwen_asr.inference.qwen3_asr as qwen3_asr_mod

    # 显存安全补丁：官方默认单块 1200s（ASR）/180s（对齐）在 8GB fp32 下必 OOM。
    # 实测 120s ASR 块峰值 6.28GB、60s 对齐块双模型峰值 5.33GB。
    qwen3_asr_mod.MAX_ASR_INPUT_SECONDS = CHUNK_SEC
    qwen3_asr_mod.MAX_FORCE_ALIGN_INPUT_SECONDS = ALIGN_CHUNK_SEC

    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        ASR_DIR,
        dtype=torch.float32,
        device_map="cuda:0",
        attn_implementation="sdpa",
        forced_aligner=ALIGNER_DIR,
        forced_aligner_kwargs=dict(
            dtype=torch.float16,  # 对齐器是单趟非自回归，fp16 慢速可接受，省一半显存
            device_map="cuda:0",
            attn_implementation="sdpa",
        ),
        max_inference_batch_size=1,  # 逐块串行，显存优先
        max_new_tokens=448,
    )
    print(f"✅ Qwen3-ASR 模型加载完成，耗时 {time.time() - t0:.1f}s "
          f"(显存 {torch.cuda.memory_allocated() / 2**30:.2f} GiB)")


# ---------------------------------------------------------------- 音频

def decode_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """解码任意格式音频为 16kHz 单声道 float32（走 ffmpeg，与生产 whisper 后端一致）"""
    import soundfile as sf

    suffix = os.path.splitext(file_path)[1].lower()
    if suffix in (".wav", ".flac", ".ogg"):
        try:
            wav, sr = sf.read(file_path, dtype="float32")
            if sr == 16000:
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                return wav, sr
        except Exception:
            pass  # 落到 pydub 重采样

    from pydub import AudioSegment
    audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= float(1 << (8 * audio.sample_width - 1))
    return samples, 16000


# ---------------------------------------------------------------- SRT 生成

def format_timestamp_srt(seconds: float) -> str:
    """秒 → SRT 时间戳 (HH:MM:SS,mmm)"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def merge_punctuation(text: str, items: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """把 ASR 文本中的标点合并回字级时间戳流。

    ForcedAligner 对齐时会丢弃标点（标点无对应声学），导致输出字幕无标点、
    也无法在句末断段。这里按顺序把非标点字符与对齐项一一对应，
    标点附着到前一个字的文本上（不占独立时间戳）。
    """
    import re
    is_word_char = re.compile(r"[一-鿿぀-ヿ가-힯a-zA-Z0-9]").match

    merged: List[List] = []
    idx = 0
    for ch in text:
        if is_word_char(ch):
            if idx < len(items):
                merged.append([items[idx][0], items[idx][1], items[idx][2]])
                idx += 1
            # 多于对齐项的字直接丢弃（正常不应发生）
        else:
            if merged:
                merged[-1][2] += ch
            # 开头标点无前字可附着，丢弃
    if idx != len(items):
        print(f"⚠️  标点合并: 文本消耗 {idx} 字，对齐项 {len(items)} 个，截断尾部")
    return [(s, e, t) for s, e, t in merged]


def items_to_srt(items: List[Tuple[float, float, str]], offset: float, segment_index: int,
                 srt_lines: List[str]) -> int:
    """把字级时间戳按显示宽度聚合成字幕段（共用 subtitle_splitter 策略）。

    items: [(start, end, text), ...] 块内相对时间；offset: 块的全局偏移秒数。
    """
    blocks = split_tokens([(s, e, t) for s, e, t in items])
    blocks = suppress_micro_blocks(blocks)
    for start, end, text in blocks:
        srt_lines.append(str(segment_index))
        srt_lines.append(f"{format_timestamp_srt(start + offset)} --> "
                         f"{format_timestamp_srt(end + offset)}")
        srt_lines.append(text)
        srt_lines.append("")
        segment_index += 1
    return segment_index


# ---------------------------------------------------------------- 转录

def transcribe_to_srt(file_path: str, language: str = "auto") -> str:
    """整块串行转录：120s 切块 → ASR + 字级对齐 → 聚合 SRT（全局时间轴）"""
    wav, sr = decode_audio(file_path)
    duration = len(wav) / sr
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"🎵 开始转录: {os.path.basename(file_path)} ({file_size_mb:.1f}MB, {duration:.0f}s)")

    qwen_lang = LANGUAGE_MAP.get(language) if language and language != "auto" else None

    n_chunks = max(1, int(np.ceil(duration / CHUNK_SEC)))
    srt_lines: List[str] = []
    segment_index = 1
    total_start = time.time()

    for i in range(n_chunks):
        piece = wav[int(i * CHUNK_SEC * sr): int((i + 1) * CHUNK_SEC * sr)]
        if len(piece) < sr // 2:  # 不足 0.5s 的尾巴跳过
            continue
        offset = i * CHUNK_SEC
        chunk_start = time.time()

        results = model.transcribe(
            audio=[(piece, sr)],
            language=qwen_lang,
            return_time_stamps=True,
        )
        r = results[0]
        items = [
            (it.start_time, it.end_time, it.text)
            for it in (r.time_stamps or [])
            if it.text.strip()
        ]
        if items and r.text.strip():
            items = merge_punctuation(r.text, items)
        if items:
            segment_index = items_to_srt(items, offset, segment_index, srt_lines)
        elif r.text.strip():
            # 对齐器没出结果但 ASR 有文本：退化为整段均布，保底不丢内容
            print(f"⚠️  块 {i+1} 无字级时间戳，退化为整段")
            srt_lines.append(str(segment_index))
            srt_lines.append(f"{format_timestamp_srt(offset)} --> "
                             f"{format_timestamp_srt(offset + len(piece) / sr)}")
            srt_lines.append(r.text.strip())
            srt_lines.append("")
            segment_index += 1

        print(f"  块 {i+1}/{n_chunks}: {len(piece)/sr:.0f}s, "
              f"{time.time() - chunk_start:.1f}s, {len(items)} 个字")

    elapsed = time.time() - total_start
    speed = duration / elapsed if elapsed > 0 else 0
    print(f"✅ 转录完成: {duration:.0f}s 音频耗时 {elapsed:.1f}s ({speed:.1f}x 实时), "
          f"{segment_index - 1} 条字幕")

    with stats_lock:
        stats.total_audio_seconds += duration
        stats.total_processing_time += elapsed

    srt_content = "\n".join(srt_lines).strip()
    # 与 whisper 后端的 "Final SRT content preview" 对齐，便于日志里逐条对比两模型输出
    print(f"📝 SRT 预览: {srt_content[:150]!r}")
    return srt_content


# ---------------------------------------------------------------- HTTP 服务

app = FastAPI(title="Qwen3-ASR API", description="Drop-in replacement for faster_whisper_api")

task_queue = deque()
processing_lock = asyncio.Lock()
current_processing = 0


@app.on_event("startup")
async def startup_event():
    load_model()
    stats.instance_id = INSTANCE_ID
    stats.gpu_device = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    stats.port = API_PORT
    print(f"实例 {stats.instance_id} 启动于 GPU {stats.gpu_device}, 端口 {stats.port}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/stats")
async def get_instance_stats():
    with stats_lock:
        uptime = time.time() - stats.start_time
        return {
            "status": "healthy",
            "stats": {
                "instance_id": stats.instance_id,
                "gpu_device": stats.gpu_device,
                "port": stats.port,
                "model": "Qwen3-ASR-0.6B + ForcedAligner-0.6B",
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "total_files_processed": stats.total_files_processed,
                "total_audio_seconds": round(stats.total_audio_seconds, 1),
                "total_processing_time": round(stats.total_processing_time, 1),
                "avg_realtime_factor": round(
                    stats.total_audio_seconds / stats.total_processing_time, 2
                ) if stats.total_processing_time > 0 else 0,
                "queue_size": len(task_queue),
                "uptime_seconds": round(uptime, 1),
            },
        }


@app.post("/inference")
async def inference(
    request: Request,
    file: UploadFile = File(...),
    response_format: str = Form("srt"),
    language: str = Form("auto"),
):
    """与 faster_whisper_api 完全一致的推理接口"""
    request_start_time = time.time()
    with stats_lock:
        stats.total_requests += 1

    require_auth(request)

    if response_format != "srt":
        raise HTTPException(status_code=400, detail="Only SRT format is supported")

    if len(task_queue) >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail=f"Service busy, queue is full (max {MAX_QUEUE_SIZE} tasks)")

    temp_file_path = None
    try:
        # 流式接收上传文件
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            total_size = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
                total_size += len(chunk)
        file_size_mb = total_size / (1024 * 1024)
        print(f"✅ 文件接收完成: {file.filename} ({file_size_mb:.2f}MB)")

        task_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        task_queue.append((task_id, temp_file_path, language, future, request_start_time, file_size_mb))
        print(f"任务 {task_id} 入队，队列长度: {len(task_queue)}")

        asyncio.create_task(process_queue())
        return await future

    except Exception as e:
        print(f"❌ 请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


async def process_queue():
    """串行处理队列（GPU 单实例，一次一个任务）"""
    global current_processing

    if current_processing >= 1:
        return
    async with processing_lock:
        if current_processing >= 1 or len(task_queue) == 0:
            return
        task_id, temp_file_path, language, future, request_start_time, file_size_mb = task_queue.popleft()
        current_processing += 1
        print(f"开始处理任务 {task_id}，剩余队列: {len(task_queue)}")

    try:
        loop = asyncio.get_event_loop()
        # 转录是 CPU/GPU 密集同步调用，丢到线程池避免阻塞事件循环（/health 保持可响应）
        srt_content = await loop.run_in_executor(None, transcribe_to_srt, temp_file_path, language)

        response = JSONResponse(
            content={"code": 0, "msg": "ok", "data": srt_content},
            status_code=200,
            media_type="application/json",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Debug-Process": "qwen_whisper_api",
            },
        )

        with stats_lock:
            stats.successful_requests += 1
            stats.total_files_processed += 1
            stats.total_file_size_mb += file_size_mb
        future.set_result(response)

    except Exception as e:
        import traceback
        print(f"ERROR: 推理异常: {e}\n{traceback.format_exc()}")
        with stats_lock:
            stats.failed_requests += 1
        future.set_result(JSONResponse(
            content={"code": 500, "msg": f"Processing error: {str(e)}", "data": ""},
            headers={"Content-Type": "application/json; charset=utf-8"},
        ))

    finally:
        async with processing_lock:
            current_processing -= 1
        if len(task_queue) > 0:
            asyncio.create_task(process_queue())


if __name__ == "__main__":
    print(f"启动 Qwen3-ASR 后端服务，端口 {API_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
