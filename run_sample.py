"""Quick end-to-end TTS sample.

Usage (on RTX 5090):
    cd /workspace/voice-agent
    python run_sample.py --model /workspace/models/qwen3-tts-0.6b \
                         --text "Hello, this is a test of the Qwen3 TTS system." \
                         --out /tmp/sample.wav
"""

import argparse
import struct
import time

import torch


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int = 24000):
    """Write raw float32 PCM bytes → 16-bit PCM WAV."""
    import array, wave
    import numpy as np

    samples = np.frombuffer(pcm_bytes, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    duration = len(pcm16) / sample_rate
    print(f"  Saved {path}  ({duration:.2f}s audio, {len(pcm16)} samples)")
    return duration


def run(model_path: str, text: str, out_path: str, temperature: float = 0.0):
    print(f"\n{'='*50}")
    print(f"  Model  : {model_path}")
    print(f"  Text   : {text!r}")
    print(f"{'='*50}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from qwen_megakernel import TalkerDecoder
    from vocoder import load_vocoder

    talker  = TalkerDecoder(model_path)
    vocoder = load_vocoder(model_path, device="cpu")
    load_s  = time.perf_counter() - t0
    print(f"  Loaded in {load_s:.1f}s\n")

    # ── Synthesize ────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    first_chunk_t = None
    all_pcm = bytearray()
    frame_times = []

    for frame_idx, codes in enumerate(talker.generate_frames(text, temperature=temperature)):
        t_frame = time.perf_counter()

        codes_t = torch.tensor([codes], dtype=torch.long)   # [1, 16] CPU
        pcm = vocoder.decode(codes_t)                        # [1920] float32
        chunk = pcm.cpu().float().numpy().tobytes()
        all_pcm.extend(chunk)

        elapsed = (time.perf_counter() - t_frame) * 1000
        frame_times.append(elapsed)

        if first_chunk_t is None:
            first_chunk_t = time.perf_counter()
            ttfc_ms = (first_chunk_t - t_start) * 1000
            print(f"  TTFC (first audio chunk): {ttfc_ms:.1f} ms")

        if frame_idx < 5 or frame_idx % 20 == 0:
            print(f"  frame {frame_idx:4d}  codes={codes[:4]}...  vocoder={elapsed:.1f}ms")

    wall_s = time.perf_counter() - t_start
    n_frames = len(frame_times)

    # ── Results ───────────────────────────────────────────────────────────────
    import numpy as np
    audio_s = n_frames * (1920 / 24000)   # 1920 samples @ 24kHz per frame
    rtf     = wall_s / audio_s if audio_s > 0 else float("inf")

    print(f"\n{'─'*50}")
    print(f"  Frames generated  : {n_frames}")
    print(f"  Audio duration    : {audio_s:.2f}s")
    print(f"  Wall time         : {wall_s:.2f}s")
    print(f"  RTF               : {rtf:.4f}  (target < 0.15)")
    print(f"  TTFC              : {ttfc_ms:.1f} ms  (target < 60 ms)")
    print(f"  Avg frame time    : {np.mean(frame_times):.1f} ms  (vocoder only)")
    print(f"{'─'*50}\n")

    audio_s = write_wav(out_path, bytes(all_pcm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/models/qwen3-tts-0.6b",
                        help="Path to Qwen3-TTS model directory")
    parser.add_argument("--text",  default="Hello, this is a test of the Qwen3 TTS megakernel.")
    parser.add_argument("--out",   default="/tmp/sample.wav")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run(args.model, args.text, args.out, args.temperature)
