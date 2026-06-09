# MegakernelTTSServer — in-process async TTS engine.
# Orchestrates: tokenizer → speaker encoder → megakernel talker
#               → code predictor → vocoder → raw PCM bytes
#
# Used directly by QwenMegakernelTTSService (no HTTP hop).
# TODO: implement after Phases 2–3

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np


class MegakernelTTSServer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        # TODO: load all sub-components
        # self.tokenizer       = Qwen3TTSTokenizer(model_path)
        # self.speaker_encoder = SpeakerEncoder(model_path, device)
        # self.talker          = MegakernelTalker(model_path, device)
        # self.code_predictor  = CodePredictor(model_path, device)
        # self.vocoder         = DACVocoder(model_path, device)
        raise NotImplementedError

    async def synthesize(
        self,
        text: str,
        speaker_ref: Optional[np.ndarray] = None,
        language: str = "english",
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM bytes (float32, 24kHz, mono) one codec frame at a time."""
        tokens = self.tokenizer.encode(text, language=language)
        spk_emb = self.speaker_encoder(speaker_ref) if speaker_ref is not None else None

        queue: asyncio.Queue = asyncio.Queue()

        async def decode_loop():
            from qwen_megakernel.generate import generate
            for codec_token in generate(self.talker, tokens, spk_emb):
                await queue.put(codec_token)
            await queue.put(None)  # sentinel

        asyncio.create_task(decode_loop())

        while True:
            codec_token = await queue.get()
            if codec_token is None:
                break
            codes = self.code_predictor(codec_token)  # [16] code groups
            pcm = self.vocoder.decode(codes)           # ~83ms float32 PCM at 24kHz
            yield pcm.tobytes()
