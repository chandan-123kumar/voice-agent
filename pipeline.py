# Full Pipecat voice pipeline: STT → LLM → TTS (megakernel)
# Runs on the RTX 5090 machine. Clients connect via WebSocket.
#
# Usage:
#   python pipeline.py
#
# Env vars required:
#   DEEPGRAM_API_KEY
#   OPENAI_API_KEY
#   MODEL_PATH  (default: ./models/qwen3-tts-0.6b)

import asyncio
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from services.qwen_megakernel_tts import QwenMegakernelTTSService


async def main():
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8765)),
            audio_in_sample_rate=16000,   # mic input from client
            audio_out_sample_rate=24000,  # TTS output to client
        )
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(model="gpt-4o")
    tts = QwenMegakernelTTSService(
        model_path=os.getenv("MODEL_PATH", "./models/qwen3-tts-0.6b")
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
