# Local WebSocket client for the voice agent demo.
# Streams microphone audio to the RTX 5090 pipeline and plays back TTS audio.
# Runs on your laptop — no models, no GPU needed.
#
# Usage:
#   python client.py --host <vast-ai-ip> --port 8765
#
# Dependencies:
#   pip install websockets pyaudio

import argparse
import asyncio
import struct

import pyaudio
import websockets

SAMPLE_RATE_IN  = 16000   # mic → server
SAMPLE_RATE_OUT = 24000   # server → speaker
CHUNK           = 1024
FORMAT          = pyaudio.paInt16
CHANNELS        = 1


async def send_mic(ws, pa):
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE_IN,
        input=True,
        frames_per_buffer=CHUNK,
    )
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            await ws.send(data)
    finally:
        stream.stop_stream()
        stream.close()


async def play_audio(ws, pa):
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE_OUT,
        output=True,
        frames_per_buffer=CHUNK,
    )
    try:
        async for message in ws:
            if isinstance(message, bytes):
                stream.write(message)
    finally:
        stream.stop_stream()
        stream.close()


async def main(host: str, port: int):
    uri = f"ws://{host}:{port}"
    print(f"Connecting to {uri} ...")
    pa = pyaudio.PyAudio()
    try:
        async with websockets.connect(uri) as ws:
            print("Connected. Start speaking.")
            await asyncio.gather(
                send_mic(ws, pa),
                play_audio(ws, pa),
            )
    finally:
        pa.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
