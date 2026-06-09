# RTX 5090 Megakernel → Qwen3-TTS → Pipecat: Implementation Plan

## Architecture Overview

The Qwen3-TTS model (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, ~906M params total) has three stages:

1. **Talker decoder** — autoregressive Qwen3 transformer: text tokens → audio codec tokens. **This is the megakernel target.**
2. **Code predictor** — small 5-layer transformer: 1 codec token → 16 code groups (non-autoregressive, fast).
3. **Audio decoder** — DAC/EnCodec vocoder: 16-group codec → PCM at 24 kHz.

The "12Hz" codec rate means each generated token = **~83ms of audio**. At 1,000 tok/s from the megakernel, theoretical RTF ≈ 0.012 — well under the 0.15 target.

---

## Critical Kernel Differences (Qwen3-0.6B vs Talker)

From `config.json` of the HF model:

| Parameter            | Qwen3-0.6B (kernel default) | Talker config       |
|----------------------|-----------------------------|---------------------|
| `hidden_size`        | 1024                        | 1024 ✓              |
| `num_hidden_layers`  | 28                          | 28 ✓                |
| `num_attention_heads`| 16                          | 16 ✓                |
| `num_kv_heads`       | 8                           | 8 ✓                 |
| `head_dim`           | 128                         | 128 ✓               |
| `intermediate_size`  | **2816**                    | **3072** ✗          |
| output `vocab_size`  | 151936 (text)               | **3072** (audio) ✗  |
| RoPE type            | standard                    | **M-RoPE** [24,20,20] ✗ |

The FFN size change (2816 → 3072) affects every weight matrix in the up/gate/down projections across all 28 layers. The vocab change shrinks the LM head significantly. M-RoPE requires a sectioned position encoding implementation.

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

## Phase 3 — Kernel Modification (est. 3–5 hrs)

Four changes to `csrc/megakernel.cu`:

### 3a. FFN Intermediate Size: 2816 → 3072
- Find the `INTERMEDIATE` constant (or equivalent `#define`)
- Update to 3072
- Audit all tiled matmul calls that reference this constant
- Verify tile size divides 3072 evenly (3072 = 2^10 × 3 — use 256-element tiles; pad if needed)
- Re-run the bench to confirm no regression in kernel correctness

### 3b. LM Head Vocab: 151936 → 3072
- Update the output projection dimension constant
- The LM head is much smaller now — should be faster
- Adjust any shared-memory or register allocations that were sized for the larger vocab

### 3c. M-RoPE (Multimodal RoPE)
- The talker uses `mrope_section: [24, 20, 20]` over the 128-dim head
  - Dims 0–23: text position encoding
  - Dims 24–43: time position encoding
  - Dims 44–63: reserved/padding
- Replace the existing `apply_rope` device function with a sectioned variant
- Validate against a single HF reference forward pass before benchmarking

### 3d. Input Embeddings (host-side, no kernel change needed)
- Handle embedding lookup in Python: text tokens → `[seq_len, 1024]` tensor
- Inject speaker embedding at the designated position
- Pass the pre-embedded `[seq_len, 1024]` activation tensor directly into the kernel
- This keeps the kernel interface clean and avoids adding a second vocab table

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
| Tile size doesn't divide 3072 cleanly | Medium | 3072 = 1024×3; use 256-tile with padding to next power-of-2 boundary |
| M-RoPE correctness | High | Validate single step against HF reference before any perf work |
| Code predictor latency spikes RTF | Low | It's 5-layer non-autoregressive; if >10ms, run it on a CUDA stream parallel to the next talker decode step |
| Audio decoder not bundled in megakernel repo | Certain | Use `qwen-tts` library vocoder or `descript-audio-codec` |
| TTFC >60ms from Python pre-processing overhead | Medium | Pre-warm speaker encoder; minimize tokenizer calls; keep embedding lookup on GPU |
| `sm_120` kernel fails to compile on slightly different Blackwell variant | Low | Check `__CUDA_ARCH__` and test with `nvcc --generate-code arch=compute_120,code=sm_120` |

---

## File Structure

```
voice_agent/
├── plan.md                          # this file
├── csrc/
│   └── megakernel.cu                # modified kernel (3072 FFN, 3072 vocab, M-RoPE)
├── qwen_megakernel/
│   ├── model.py                     # weight loading for talker
│   ├── generate.py                  # autoregressive decode loop
│   └── bench.py                     # benchmark script
├── tts_server.py                    # MegakernelTTSServer (async generator)
├── extract_weights.py               # HF → megakernel weight remapping
├── services/
│   └── qwen_megakernel_tts.py       # Pipecat TTSService subclass
├── pipeline.py                      # full STT → LLM → TTS Pipecat pipeline (runs on RTX 5090)
├── client.py                        # local WebSocket client: mic input + speaker output
├── benchmark.py                     # TTFC / RTF / E2E measurement script
├── requirements.txt
└── README.md
```

---

## Deliverables

1. **Working repo** with `README.md` covering build instructions, architecture decisions, kernel modifications documented
2. **Performance report**: real tok/s, TTFC, RTF, E2E latency numbers with methodology
3. **Demo recording**: end-to-end voice agent in action
