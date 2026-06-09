"""Qwen Megakernel — Qwen3-TTS talker decode for RTX 5090."""

from qwen_megakernel.build import get_extension as _get_ext

_get_ext()

from qwen_megakernel.model import load_weights, TalkerDecoder  # noqa: E402
from qwen_megakernel.code_predictor import CodePredictor       # noqa: E402

__all__ = ["load_weights", "TalkerDecoder", "CodePredictor"]
