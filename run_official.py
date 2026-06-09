"""Run TTS using official qwen-tts model classes (bypassing torchaudio import).

Usage:
    cd /workspace/voice-agent
    python3 run_official.py --model /workspace/models/qwen3-tts-0.6b \
                            --text "Hello, this is a test." \
                            --out /tmp/official.wav
"""
import sys, types, importlib.util, argparse, time, wave
import numpy as np

# ── mock torchaudio before any qwen_tts imports ──────────────────────────────
_spec = importlib.util.spec_from_loader('torchaudio', loader=None)
_ta = types.ModuleType('torchaudio')
_ta.__spec__ = _spec
_ta.__version__ = '2.11.0'
sys.modules['torchaudio'] = _ta
for _sub in ['torchaudio.compliance', 'torchaudio.compliance.kaldi',
             'torchaudio._extension', 'torchaudio.functional', 'torchaudio.transforms']:
    _m = types.ModuleType(_sub)
    _m.__spec__ = _spec
    sys.modules[_sub] = _m

# ── now safe to import ────────────────────────────────────────────────────────
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model


def write_wav(path, samples, sr=24000):
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    peak = np.max(np.abs(samples))
    if peak > 1e-6:
        samples = samples / peak * 0.95
    pcm16 = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    print(f"  Saved {path}  ({len(pcm16)/sr:.2f}s)")


def run(model_path, text, out_path, language="English"):
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    print(f"\nLoading model from {model_path} ...")
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\nLoading vocoder ...")
    t0 = time.perf_counter()
    vocoder = Qwen3TTSTokenizerV2Model.from_pretrained(
        model_path + "/speech_tokenizer",
        torch_dtype=torch.float32,
    ).to("cpu")
    vocoder.eval()
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\nSynthesizing: {text!r}")
    t_start = time.perf_counter()

    # Format text with the expected chat template:
    # <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    full_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    print(f"  Input ids shape: {input_ids.shape}, first 3: {input_ids[0,:3].tolist()}")

    with torch.no_grad():
        result = model.generate(
            input_ids=[input_ids],  # list of 2D [1, seq] tensors
            languages=[language],
            non_streaming_mode=True,
            do_sample=False,
            temperature=1.0,
        )

    talker_codes_list, _ = result
    codes = talker_codes_list[0]  # [T, 16]
    print(f"  Generated {codes.shape[0]} frames in {time.perf_counter()-t_start:.2f}s")
    print(f"  First frame codes: {codes[0].tolist()}")

    # Decode with vocoder — expects [batch, T, 16]
    t_voc = time.perf_counter()
    with torch.no_grad():
        out = vocoder.decode(codes.cpu().unsqueeze(0))  # [1, T, 16]
    print(f"  Vocoder decode in {time.perf_counter()-t_voc:.2f}s")
    print(f"  Vocoder output type: {type(out)}, attrs: {[a for a in dir(out) if not a.startswith('_')]}")
    # audio_values is a list of tensors, one per batch item
    wav = out.audio_values[0].squeeze().float()  # [samples]

    write_wav(out_path, wav.float().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/models/qwen3-tts-0.6b")
    parser.add_argument("--text",  default="Hello, this is a test of the Qwen3 TTS system.")
    parser.add_argument("--out",   default="/tmp/official.wav")
    parser.add_argument("--language", default="English")
    args = parser.parse_args()
    run(args.model, args.text, args.out, args.language)
