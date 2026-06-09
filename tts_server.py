"""MegakernelTTSServer — in-process async TTS engine.

Pipeline: text → TalkerDecoder (megakernel) → CodePredictor (PyTorch)
          → DAC vocoder → raw PCM bytes streamed per codec frame.

Each codec frame = ~83ms at 24kHz (12 Hz codec rate).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch


class MegakernelTTSServer:
    """Thread-safe async TTS server. One instance per process."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self._lock = asyncio.Lock()

        print(f"[TTS] Loading TalkerDecoder from {model_path} …")
        t0 = time.perf_counter()
        from qwen_megakernel import TalkerDecoder
        self._talker = TalkerDecoder(str(model_path))
        print(f"[TTS] TalkerDecoder loaded in {time.perf_counter()-t0:.1f}s")

        print("[TTS] Loading vocoder …")
        t0 = time.perf_counter()
        self._vocoder = _load_vocoder(str(model_path))
        print(f"[TTS] Vocoder loaded in {time.perf_counter()-t0:.1f}s")

    async def synthesize(
        self,
        text: str,
        language: str = "english",
        temperature: float = 0.0,
        first_token_cb=None,
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM bytes (float32 LE, 24kHz, mono) one codec frame at a time.

        Args:
            text: Input text to synthesize.
            language: One of english/chinese/japanese/korean/…
            temperature: Sampling temperature (0 = greedy).
            first_token_cb: Optional callable invoked just before yielding the
                            first audio chunk — used to measure TTFC.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            # Run the CPU-bound decode loop in a thread so we don't block the
            # event loop (megakernel CUDA calls release the GIL).
            queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=4)

            async def _decode():
                def _worker():
                    first = True
                    for codes in self._talker.generate_frames(
                        text, language=language, temperature=temperature
                    ):
                        pcm = _codes_to_pcm(codes, self._vocoder)
                        if first and first_token_cb is not None:
                            first_token_cb()
                            first = False
                        asyncio.run_coroutine_threadsafe(
                            queue.put(pcm), loop
                        ).result()
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None), loop
                    ).result()

                await loop.run_in_executor(None, _worker)

            task = asyncio.create_task(_decode())

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

            await task


# ─── vocoder helpers ──────────────────────────────────────────────────────────

def _load_vocoder(model_path: str):
    """Load the Qwen3TTS speech tokenizer decoder (runs on CPU — fast for conv decode)."""
    import sys
    sys.path.insert(0, str(Path(model_path).parent.parent / "voice-agent"))
    from vocoder import load_vocoder as _lv
    return _lv(model_path, device="cpu")


def _codes_to_pcm(codes: list[int], vocoder) -> bytes:
    """Convert 16 codec codes for one frame → float32 PCM bytes at 24kHz."""
    codes_t = torch.tensor([codes], dtype=torch.long)  # [1, 16] on CPU
    pcm = vocoder.decode(codes_t)                       # [1920] float32
    return pcm.cpu().float().numpy().tobytes()
