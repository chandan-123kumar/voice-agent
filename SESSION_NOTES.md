# Session Notes — Qwen3-TTS Megakernel Voice Agent

## Project Goal
ssh-ed25519                                                        AAAAC3NzaC1lZDI1NTE5AAAAINBD2pNkSAfMiDwJU94bg48+hGn3fMJpH9TaXqzy+3VG       
  chandankuma
Build a low-latency voice agent on an RTX 5090 (Blackwell sm_120) using:
- **STT** → **LLM** → **TTS** (Pipecat pipeline)
- **TTS model**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (talker + vocoder)
- **Targets**: TTFC < 60 ms, RTF < 0.15, streaming audio output

---

## Infrastructure

- **Remote server**: `ssh -p 40330 root@180.189.55.38` (vast.ai RTX 5090)
- **CUDA**: 13.0, sm_120 Blackwell, PyTorch nightly 2.10.0a0
- **Model path on server**: `/workspace/models/qwen3-tts-0.6b/`
- **Project path on server**: `/workspace/voice-agent/`
- **Local path**: `/Users/chandankumar/Desktop/voice_agent/`
- **Git repo**: `github.com/chandan-123kumar/voice-agent`

---

## Key Discoveries

### Token IDs (from `config.json` → `talker_config`)
```
CODEC_BOS_ID       = 2149
CODEC_EOS_ID       = 2150
CODEC_PAD_ID       = 2148
CODEC_THINK_ID     = 2154
CODEC_NOTHINK_ID   = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157
TTS_TEXT_BOS_ID    = 151672   # <tts_text_bos>
TTS_TEXT_EOD_ID    = 151673   # <tts_text_eod>
TTS_TEXT_PAD_ID    = 151671   # <tts_pad>

Language IDs: english=2050, chinese=2055, ...
```

### Correct Chat Template
```
<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
```

### Official Codec Prefix (English, non-streaming)
```
[think(2154), think_bos(2156), lang(2050), think_eos(2157), pad(2148)]
```

### Embedding Construction (official model)
Text and codec embeddings are **summed** at each position, not concatenated.

---

## Bugs Fixed

| Bug | Fix |
|-----|-----|
| `torchaudio` ABI mismatch (`libtorchaudio.abi3.so: undefined symbol`) | Mock all `torchaudio` submodules with `types.ModuleType` before any import |
| Model loops forever (never hits EOS) | Stop on `token >= 2048` (catches EOS, PAD, and all special tokens) |
| Audio silent / very quiet (amplitude 610/32767) | Peak-normalise in `write_wav()` — `samples = samples / peak * 0.95` |
| `run_sample.py: unrecognized arguments: --max_frames` | Added `--max_frames` argparse argument |
| Wrong language / 1-second audio | Wrap text in chat template; pass `languages=[language]` to `generate()` |
| `generate_talker_codes` AttributeError | Correct method is `model.generate()` |
| `input_ids` IndexError (too many indices for 1D tensor) | Model expects a **list** of 2D `[1, seq]` tensors |
| `Qwen3TTSTokenizerV2DecoderOutput` has no `.dim()` | Audio is in `out.audio_values[0]` (list of tensors), not `.audio_values` directly |
| Vocoder `IndexError: index 2115 out of bounds` | Special token in audio stream — fixed by `token >= 2048: break` |
| Tokenizer loading wrong vocab | Load from `model_path` with `fix_mistral_regex=True`, not `"Qwen/Qwen3-0.6B"` |
| Custom PyTorch prefill → noise | Multiple attempts failed (wrong KV cache). Reverted to simple version. |

---

## Current State of Files

### `run_official.py` ✅ WORKING BASELINE
Uses `Qwen3TTSForConditionalGeneration` (official HuggingFace class) via `qwen-tts` pip package.  
Generates **correct English speech**. No kernel optimisation — RTF ~1.04.

```python
# torchaudio mock must come before all imports
sys.modules['torchaudio'] = _ta  # + submodules

model = AutoModel.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
vocoder = Qwen3TTSTokenizerV2Model.from_pretrained(model_path + "/speech_tokenizer").to("cpu")

full_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()

result = model.generate(input_ids=[input_ids], languages=[language],
                        non_streaming_mode=True, do_sample=False, temperature=1.0)
codes = result[0][0]          # [T, 16]
out = vocoder.decode(codes.cpu().unsqueeze(0))
wav = out.audio_values[0].squeeze().float()
```

### `run_sample.py` — megakernel path (generates noise)
Uses `TalkerDecoder` (custom CUDA megakernel) + `vocoder.py`.  
Generates noise because `TalkerDecoder.prefill()` produces wrong KV cache.

### `qwen_megakernel/qwen_megakernel/model.py` — reverted
- Good changes kept: correct constants, tokenizer from `model_path`, `token >= 2048` stop
- Broken complex prefill (summed text+codec embeddings) **reverted** to simple version

### `vocoder.py`
From-scratch decoder. `device` param fixed. ~77 ms/frame on CPU.

---

## Performance (Current)

| Metric | Current | Target |
|--------|---------|--------|
| RTF | ~1.04 | < 0.15 |
| TTFC | ~470 ms | < 60 ms |
| Vocoder (CPU) | ~77 ms/frame | ~5 ms (GPU) |

---

## Architecture: Two Paths

```
run_official.py (WORKING, slow)
  HF model.generate() → [T, 16] codes → Qwen3TTSTokenizerV2Model.decode() → WAV

run_sample.py (FAST target, broken prefill)
  TalkerDecoder.prefill() → [KV cache] → megakernel step() × T → [T, 16] codes → vocoder.decode() → WAV
```

---

## Next Steps (Planned)

1. **Fix prefill** — use official `Qwen3TTSForConditionalGeneration` for prefill and extract KV cache; hand off to megakernel for streaming AR decode
2. **Move vocoder to GPU** — reduce ~77 ms/frame → ~5 ms/frame (7× RTF improvement alone)
3. **Wire Pipecat pipeline** — `tts_server.py` → STT → LLM → TTS streaming
4. **Benchmark** — verify RTF < 0.15 and TTFC < 60 ms on RTX 5090

---

## HuggingFace Model
- Repo: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`  
- Note: `Qwen/Qwen3-TTS-0.6B` returns 404 — use the full name above
