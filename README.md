# RTX 5090 Megakernel → Qwen3-TTS → Pipecat

Real-time voice agent using AlpinDale's CUDA megakernel as the decode backend for Qwen3-TTS, streamed into a Pipecat WebSocket pipeline.

## Architecture

```
Client (laptop)                     Server (RTX 5090, Vast.ai)
──────────────                      ────────────────────────────
microphone                          Pipecat pipeline
   │  raw PCM 16kHz                   ├── Deepgram STT
   │  WebSocket ─────────────────────►│
   │                                  ├── OpenAI LLM
   │  raw PCM 24kHz                   ├── QwenMegakernelTTSService
   │◄─────────────────────────────────│     └── MegakernelTTSServer
speaker                                           ├── Tokenizer
                                                  ├── Megakernel (CUDA sm_120)
                                                  ├── Code predictor
                                                  └── DAC vocoder
```

## Performance Targets

| Metric | Target |
|--------|--------|
| tokens/sec | >900 |
| TTFC | <60 ms |
| RTF | <0.15 |

## Build Instructions

### 1. Provision GPU

Rent an RTX 5090 on Vast.ai. Requires `sm_120` / Blackwell architecture.

### 2. Clone and build the megakernel

```bash
git clone https://github.com/AlpinDale/qwen_megakernel
cd qwen_megakernel
uv pip install -r requirements.txt
python -m qwen_megakernel.bench   # confirm ~1000 tok/s baseline
```

### 3. Download model weights

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./models/qwen3-tts-0.6b
```

### 4. Extract talker weights
```bash
python extract_weights.py --hf_model ./models/qwen3-tts-0.6b --out_dir ./models/qwen3-tts-0.6b
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the pipeline

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
python pipeline.py
```

### 7. Connect from your laptop

```bash
pip install websockets pyaudio
python client.py --host <vast-ai-ip> --port 8765
```

## Benchmark

```bash
python benchmark.py --model_path ./models/qwen3-tts-0.6b
```

## Kernel Modifications

See `csrc/megakernel.cu` for full details. Summary of changes from the original AlpinDale kernel:

| Change | Original | Modified |
|--------|----------|----------|
| FFN intermediate size | 2816 | 3072 |
| Output vocab size | 151936 | 3072 (audio tokens) |
| RoPE | Standard | M-RoPE [24, 20, 20] |

## Repository Structure

```
├── csrc/megakernel.cu          # modified CUDA kernel
├── qwen_megakernel/            # Python wrapper for the kernel
│   ├── model.py                # weight loading
│   ├── generate.py             # autoregressive decode loop
│   └── bench.py                # tok/s benchmark
├── tts_server.py               # in-process async TTS engine
├── extract_weights.py          # HF → megakernel weight remapping
├── services/
│   └── qwen_megakernel_tts.py  # Pipecat TTSService subclass
├── pipeline.py                 # full pipeline (runs on RTX 5090)
├── client.py                   # local WebSocket client
├── benchmark.py                # TTFC / RTF / E2E metrics
└── requirements.txt
```
