# Extract and remap Qwen3-TTS talker weights from HuggingFace safetensors
# to the flat binary format expected by the megakernel.
#
# Usage:
#   python extract_weights.py \
#     --hf_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
#     --out_dir ./models/qwen3-tts-0.6b

import argparse
from pathlib import Path


# Expected talker weight shapes (from config.json inspection):
#   Q proj:        [2048, 1024]   (16 heads × 128 dim)
#   K/V proj:      [1024, 1024]   (8 kv_heads × 128 dim)
#   O proj:        [1024, 2048]
#   Gate/Up proj:  [3072, 1024]   (intermediate_size=3072)
#   Down proj:     [1024, 3072]
#   Embedding:     [154[text_vocab + audio_vocab], 1024]
#   LM head:       [3072, 1024]   (audio vocab output)

HF_TO_KERNEL_MAP = {
    # TODO: populate after Phase 2 weight inspection
    # "model.talker.layers.{i}.self_attn.q_proj.weight": "layer_{i}_q",
}


def extract(hf_model: str, out_dir: Path):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise SystemExit("pip install transformers safetensors")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {hf_model} ...")
    # TODO: load only talker sub-module to avoid loading full model into CPU RAM
    raise NotImplementedError("Complete after Phase 2 weight shape inspection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--out_dir", default="./models/qwen3-tts-0.6b")
    args = parser.parse_args()
    extract(args.hf_model, Path(args.out_dir))
