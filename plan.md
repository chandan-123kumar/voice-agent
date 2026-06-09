# RTX 5090 Megakernel → Qwen3-TTS → Pipecat: Implementation Plan

## Architecture Overview

The Qwen3-TTS model (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, ~906M params total) has three stages:

1. **Talker decoder** — autoregressive Qwen3 transformer: text tokens → audio codec tokens. **This is the megakernel target.**
2. **Code predictor** — small 5-layer transformer: 1 codec token → 16 code groups (non-autoregressive, fast).
3. **Audio decoder** — DAC/EnCodec vocoder: 16-group codec → PCM at 24 kHz.

The "12Hz" codec rate means each generated token = **~83ms of audio**. At 1,000 tok/s from the megakernel, theoretical RTF ≈ 0.012 — well under the 0.15 target.

---

## Critical Kernel Differences (Qwen3-0.6B vs Talker)

From inspecting the actual kernel source (`qwen_megakernel/csrc/kernel.cu`) and the talker `config.json`:

| Parameter            | Kernel (`kernel.cu`)        | Talker config       |
|----------------------|-----------------------------|---------------------|
| `HIDDEN_SIZE`        | 1024                        | 1024 ✓              |
| `NUM_LAYERS`         | 28                          | 28 ✓                |
| `NUM_Q_HEADS`        | 16                          | 16 ✓                |
| `NUM_KV_HEADS`       | 8                           | 8 ✓                 |
| `HEAD_DIM`           | 128                         | 128 ✓               |
| `INTERMEDIATE_SIZE`  | **3072** ✓                  | **3072** ✓          |
| `LDG_VOCAB_SIZE`     | **151936** (text)           | **3072** (audio) ✗  |
| RoPE type            | standard                    | **M-RoPE** [24,20,20] ✗ |
| LM head              | tied to embed_tokens        | separate projection ✗ |

**Good news**: `INTERMEDIATE_SIZE` is already 3072 — no FFN changes needed. Only two kernel modifications are required: the LM head vocab size and the RoPE implementation.

---

## Inference Data Flow (Streaming)

```
text input
  │
  ├─→ [Qwen3-TTS tokenizer]    → text token IDs
  ├─→ [Speaker encoder]        → speaker embedding (optional, for voice cloning)
  │
  ▼
[Talker decode loop]           ← MEGAKERNEL (autoregressive, ~1ms/token)
  │  generates one audio codec token per step
  │
  ├─→ immediately enqueue each token (asyncio.Queue)
  │
  ▼
[Code predictor]               (PyTorch, 5-layer, non-autoregressive, ~2-5ms)
  │  expands 1 codec token → 16 code groups
  │
  ▼
[Audio decoder]                (DAC/EnCodec vocoder)
  │  16 codes → ~83ms of PCM at 24kHz
  │
  ▼
[Pipecat TTSAudioRawFrame]     pushed per chunk — no full-utterance buffering
```

---

## Phase 1 — Environment Setup (est. 1–2 hrs)

- [ ] Provision RTX 5090 on Vast.ai (requires `sm_120` / Blackwell arch)
- [ ] Clone `https://github.com/AlpinDale/qwen_megakernel`
- [ ] Download `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` weights from HuggingFace
- [ ] Install CUDA 12.8+, `pipecat-ai`, `qwen-tts`, `descript-audio-codec` (or equivalent vocoder)
- [ ] Run original megakernel bench: `python -m qwen_megakernel.bench` — confirm ~1,000 tok/s baseline
- [ ] Save baseline numbers for comparison

---

## Phase 2 — Weight Extraction & Inspection (est. 1–2 hrs)

- [ ] Load the HF model in Python, iterate all `model.talker.*` parameter names and shapes
- [ ] Confirm transformer core shapes match expected talker dimensions:
  - Q proj: `[1024, 2048]` (16 heads × 128 dim)
  - K/V proj: `[1024, 1024]` (8 kv_heads × 128 dim)
  - O proj: `[2048, 1024]`
  - Gate/Up proj: `[3072, 1024]`
  - Down proj: `[1024, 3072]`
  - Embedding: `[151936 + 3072, 1024]` (text + audio token embeddings)
  - LM head: `[3072, 1024]` (audio vocab output)
- [ ] Write `extract_weights.py`: loads HF model, remaps `model.talker.*` → megakernel flat binary format
- [ ] Identify speaker conditioning injection point in the forward pass

---

## Phase 3 — Kernel Modification (est. 2–3 hrs)

Two changes to `qwen_megakernel/csrc/kernel.cu` (INTERMEDIATE_SIZE is already correct at 3072):

### 3a. LM Head Vocab: 151936 → 3072
- Change `LDG_VOCAB_SIZE` from 151936 → 3072
- The LM head kernel (`lm_head_kernel`) launches with `LDG_LM_NUM_BLOCKS=1184` and `LDG_LM_BLOCK_SIZE=256` — re-tune these for the much smaller 3072-row projection
- In `model.py`: the original uses tied embeddings (`lm_head_weight = embed_weight`). The talker has a separate LM head; load `model.talker.lm_head.weight` instead
- The LM head being 50× smaller should make this stage essentially free

### 3b. M-RoPE (Multimodal RoPE)
- The talker uses `mrope_section: [24, 20, 20]` over the 128-dim head:
  - Dims 0–23: text sequence position
  - Dims 24–43: audio time position
  - Dims 44–63: zero (padding section)
- In the kernel: replace the `apply_rope` device function with an M-RoPE variant that applies different position frequencies per section
- In `model.py`: generate separate cos/sin tables per section instead of one unified table; pass as three pairs to the kernel
- Validate a single forward pass against HF reference output before running any perf benchmarks

### 3c. Weight Loading (host-side, `model.py`)
- Change `load_weights()` to load from `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` and pull the `model.talker.*` sub-tree
- Embedding table: load `model.talker.embed_tokens.weight` (covers both text and audio vocab)
- LM head: load `model.talker.lm_head.weight` (NOT tied; separate tensor)
- All 28 layer weight keys change prefix from `model.layers.{i}.` → `model.talker.layers.{i}.`
- All other tensor names within each layer are identical (q_proj, k_proj, mlp.gate_proj, etc.)

---

## Phase 4 — Inference Server (est. 2–3 hrs)

Build `tts_server.py` as an in-process async generator (no HTTP overhead for the Pipecat integration):

```python
class MegakernelTTSServer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.talker = MegakernelTalker(model_path)       # modified kernel
        self.code_predictor = CodePredictor(model_path)  # PyTorch, 5-layer
        self.vocoder = DACVocoder(model_path)            # audio decoder
        self.tokenizer = Qwen3TTSTokenizer(model_path)

    async def synthesize(
        self,
        text: str,
        speaker_ref: Optional[np.ndarray] = None,
        language: str = "english",
    ) -> AsyncIterator[bytes]:
        tokens = self.tokenizer.encode(text, language=language)
        spk_emb = self.speaker_encoder(speaker_ref) if speaker_ref else None

        queue: asyncio.Queue = asyncio.Queue()

        async def decode_loop():
            for codec_token in self.talker.generate(tokens, spk_emb):
                await queue.put(codec_token)
            await queue.put(None)  # sentinel

        asyncio.create_task(decode_loop())

        while True:
            codec_token = await queue.get()
            if codec_token is None:
                break
            codes = self.code_predictor(codec_token)   # [16] code groups
            pcm = self.vocoder.decode(codes)            # ~83ms of float32 PCM
            yield pcm.tobytes()
```

Key decisions:
- Decode loop and audio decode run as pipelined async tasks — no full-utterance buffering
- `synthesize()` yields raw PCM bytes per codec frame (~83ms chunks at 24kHz)
- Speaker encoder runs once before the decode loop starts

---

## Phase 5 — Pipecat Integration (est. 1–2 hrs)

### Transport: WebSocket (not WebRTC/Daily)

WebRTC adds 20–80ms of jitter-buffer and ICE negotiation overhead — enough to blow the TTFC budget before a single token is decoded. `WebsocketServerTransport` sends raw PCM both ways over a single persistent connection with no codec round-trip, no STUN/TURN servers, and no signaling overhead.

```
Your laptop (browser or Python client)
    │  WebSocket — raw 16-bit PCM, bidirectional
    ▼
RTX 5090 (Vast.ai, port 8765)
  └── Pipecat pipeline (WebsocketServerTransport)
        ├── Deepgram STT   (cloud API)
        ├── OpenAI LLM     (cloud API)
        └── QwenMegakernelTTSService (in-process, zero-copy to transport)
```

Open one port on the Vast.ai instance (e.g. 8765). SSH into the box to run the server. Connect from a local Python client or a minimal browser page.

### Custom TTS Service

```python
# services/qwen_megakernel_tts.py
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

class QwenMegakernelTTSService(TTSService):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.server = MegakernelTTSServer(model_path)

    async def run_tts(self, text: str):
        yield TTSStartedFrame()
        async for pcm_chunk in self.server.synthesize(text):
            yield TTSAudioRawFrame(
                audio=pcm_chunk,
                sample_rate=24000,
                num_channels=1,
            )
        yield TTSStoppedFrame()
```

### Full Pipeline

```python
# pipeline.py
from pipecat.transports.network.websocket_server import (
    WebsocketServerTransport,
    WebsocketServerParams,
)

transport = WebsocketServerTransport(
    params=WebsocketServerParams(
        host="0.0.0.0",
        port=8765,
        audio_in_sample_rate=16000,   # mic input from client
        audio_out_sample_rate=24000,  # TTS output to client
    )
)

pipeline = Pipeline([
    transport.input(),                      # raw PCM from WebSocket client
    DeepgramSTTService(api_key=...),        # speech → text
    OpenAILLMService(model="gpt-4o"),       # text → text
    QwenMegakernelTTSService(               # text → audio (megakernel)
        model_path="./models/qwen3-tts-0.6b"
    ),
    transport.output(),                     # raw PCM back over WebSocket
])
```

### Local Client (for demo / testing)

```python
# client.py  — runs on your laptop
import asyncio, websockets, pyaudio, sys

async def main():
    async with websockets.connect("ws://<vast-ai-ip>:8765") as ws:
        # send mic audio in one task, play received audio in another
        await asyncio.gather(send_mic(ws), play_audio(ws))
```

For the demo recording, `client.py` is all that runs locally — no Pipecat, no models, just mic-in and speaker-out over the WebSocket.

---

## Phase 6 — Validation & Benchmarking (est. 1–2 hrs)

### Metrics to Instrument

Instrument with `time.perf_counter_ns()` at each boundary:

| Metric | Measurement point |
|---|---|
| **tokens/sec** | megakernel step timer (already in bench script) |
| **TTFC** | `t(first TTSAudioRawFrame pushed) - t(text arrives to run_tts)` |
| **RTF** | `total_synthesis_wall_time / total_audio_duration` |
| **E2E latency** | `t(first speaker audio) - t(last mic sample in utterance)` |

### Targets

| Metric | Target | Stretch |
|---|---|---|
| tokens/sec | >900 | >1000 |
| TTFC | <60ms | <50ms |
| RTF | <0.15 | <0.10 |
| E2E latency | <500ms | <300ms |

### Validation Checklist

- [ ] Single forward pass matches HF reference output (logits within 1e-3 tolerance)
- [ ] Generated audio is intelligible with no glitches or dropped frames
- [ ] Audio arrives at Pipecat frame-by-frame (confirm queue is not buffering full utterance)
- [ ] Full round-trip test: speak → transcribe → LLM response → TTS → playback
- [ ] Record demo video of end-to-end voice agent

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| LM head block config after vocab shrink | Low | Re-tune `LDG_LM_NUM_BLOCKS` downward from 1184; smaller vocab = fewer blocks needed |
| M-RoPE correctness | High | Validate single step against HF reference before any perf work |
| Code predictor latency spikes RTF | Low | It's 5-layer non-autoregressive; if >10ms, run it on a CUDA stream parallel to the next talker decode step |
| Audio decoder not bundled in megakernel repo | Certain | Use `qwen-tts` library vocoder or `descript-audio-codec` |
| TTFC >60ms from Python pre-processing overhead | Medium | Pre-warm speaker encoder; minimize tokenizer calls; keep embedding lookup on GPU |
| `sm_120` kernel fails to compile on slightly different Blackwell variant | Low | Check `__CUDA_ARCH__` and test with `nvcc --generate-code arch=compute_120,code=sm_120` |

---

## File Structure

```
voice_agent/
├── plan.md                              # this file
├── qwen_megakernel/                     # cloned from AlpinDale/qwen_megakernel
│   ├── csrc/
│   │   ├── kernel.cu                    # MODIFIED: LDG_VOCAB_SIZE=3072, M-RoPE
│   │   └── torch_bindings.cpp
│   └── qwen_megakernel/
│       ├── model.py                     # MODIFIED: talker weight loading, separate LM head
│       ├── bench.py                     # benchmark script (reuse, point at TTS model)
│       └── build.py
├── tts_server.py                        # MegakernelTTSServer (async generator)
├── extract_weights.py                   # (optional) HF weight inspection helper
├── services/
│   └── qwen_megakernel_tts.py           # Pipecat TTSService subclass
├── pipeline.py                          # full STT → LLM → TTS pipeline (runs on RTX 5090)
├── client.py                            # local WebSocket client: mic input + speaker output
├── benchmark.py                         # TTFC / RTF / E2E measurement script
├── requirements.txt
└── README.md
```

---

## Deliverables

1. **Working repo** with `README.md` covering build instructions, architecture decisions, kernel modifications documented
2. **Performance report**: real tok/s, TTFC, RTF, E2E latency numbers with methodology
3. **Demo recording**: end-to-end voice agent in action
