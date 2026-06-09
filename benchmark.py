# Benchmarking script: measures TTFC, RTF, tok/s, and end-to-end latency.
# Run on the RTX 5090 after all components are wired up (Phase 6).
#
# Usage:
#   python benchmark.py --model_path ./models/qwen3-tts-0.6b

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BenchResult:
    tokens_per_sec: float = 0.0
    ttfc_ms: float = 0.0          # time to first audio chunk
    rtf: float = 0.0              # real-time factor
    total_audio_sec: float = 0.0
    synthesis_wall_sec: float = 0.0
    raw_step_times_ms: List[float] = field(default_factory=list)

    def report(self):
        print(f"{'─'*40}")
        print(f"  tokens/sec        : {self.tokens_per_sec:.1f}")
        print(f"  TTFC              : {self.ttfc_ms:.1f} ms  (target <60ms)")
        print(f"  RTF               : {self.rtf:.4f}  (target <0.15)")
        print(f"  audio duration    : {self.total_audio_sec:.2f} s")
        print(f"  synthesis wall    : {self.synthesis_wall_sec:.2f} s")
        print(f"{'─'*40}")


async def bench_tts(model_path: str, text: str = "Hello, this is a benchmark of the voice agent.") -> BenchResult:
    from tts_server import MegakernelTTSServer

    server = MegakernelTTSServer(model_path)
    result = BenchResult()

    first_chunk = True
    chunks: List[bytes] = []
    t_start = time.perf_counter()

    async for pcm_chunk in server.synthesize(text):
        if first_chunk:
            result.ttfc_ms = (time.perf_counter() - t_start) * 1000
            first_chunk = False
        chunks.append(pcm_chunk)

    result.synthesis_wall_sec = time.perf_counter() - t_start

    total_samples = sum(len(c) for c in chunks) // 4  # float32 = 4 bytes
    result.total_audio_sec = total_samples / 24000
    result.rtf = result.synthesis_wall_sec / result.total_audio_sec if result.total_audio_sec > 0 else float("inf")

    return result


def bench_megakernel(model_path: str, steps: int = 200):
    from qwen_megakernel.bench import run_bench
    run_bench(model_path, steps=steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/qwen3-tts-0.6b")
    parser.add_argument("--text", default="Hello, this is a benchmark of the Qwen3-TTS megakernel voice agent pipeline.")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    print("\n=== Megakernel tok/s bench ===")
    bench_megakernel(args.model_path, args.steps)

    print("\n=== Full TTS pipeline bench ===")
    result = asyncio.run(bench_tts(args.model_path, args.text))
    result.report()
