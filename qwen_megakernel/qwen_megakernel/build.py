"""JIT compilation of the megakernel CUDA extension."""

import os
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_DIR, "../csrc")


# RTX 5090 (sm_120) tuning flags.
def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


KERNEL_FLAGS = [
    f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
    f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
    f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}",
    f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
    f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
    f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
    f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
    f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
    f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
    f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
    f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
    f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
    f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
    "-DLDG_USE_UINT4",
    "-DLDG_ATTENTION_VEC4",
    "-DLDG_WEIGHT_LDCS",
    "-DLDG_MLP_SMEM",
]

CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-arch=sm_120a",
    f"-I{_CSRC}",
] + KERNEL_FLAGS


def _find_cached_so():
    """Locate any pre-built qwen_megakernel_C.so across all cache variants."""
    base = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    if not os.path.isdir(base):
        return None
    for variant in sorted(os.listdir(base), reverse=True):
        so = os.path.join(base, variant, "qwen_megakernel_C", "qwen_megakernel_C.so")
        if os.path.exists(so):
            return so
    return None


def get_extension():
    """Build (or return cached) the megakernel extension. Triggers torch.ops.qwen_megakernel_C.*"""
    import torch
    global _module
    if _module is not None:
        return _module

    # Fast path: load pre-built .so directly if it exists
    cached_so = _find_cached_so()
    if cached_so is not None:
        try:
            torch.ops.load_library(cached_so)
            _module = True
            return _module
        except OSError:
            pass  # stale .so (wrong ABI); fall through to rebuild

    # Slow path: JIT build — patch _get_cuda_arch_flags so sm_120a isn't rejected
    import torch.utils.cpp_extension as _cext
    _orig = _cext._get_cuda_arch_flags

    def _patched_arch_flags(cflags=None):
        return []  # we provide -arch=sm_120a explicitly

    _cext._get_cuda_arch_flags = _patched_arch_flags
    try:
        _module = load(
            name="qwen_megakernel_C",
            sources=[
                os.path.join(_CSRC, "torch_bindings.cpp"),
                os.path.join(_CSRC, "kernel.cu"),
            ],
            extra_cuda_cflags=CUDA_FLAGS,
            extra_cflags=[f"-I{_CSRC}"],
            verbose=True,
        )
    finally:
        _cext._get_cuda_arch_flags = _orig
    return _module
