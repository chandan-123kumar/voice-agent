# Pipecat TTSService subclass backed by the megakernel inference server.
# Plugs into the Pipecat pipeline as a drop-in TTS service.
# TODO: wire up once tts_server.py is implemented (Phase 5)

from __future__ import annotations

from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from tts_server import MegakernelTTSServer


class QwenMegakernelTTSService(TTSService):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self._server = MegakernelTTSServer(model_path)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame()
        try:
            async for pcm_chunk in self._server.synthesize(text):
                yield TTSAudioRawFrame(
                    audio=pcm_chunk,
                    sample_rate=24000,
                    num_channels=1,
                )
        except Exception as e:
            yield ErrorFrame(str(e))
        finally:
            yield TTSStoppedFrame()
