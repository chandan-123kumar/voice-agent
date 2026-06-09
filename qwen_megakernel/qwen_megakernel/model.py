"""Weight loading and decode API for Qwen3-TTS talker decoder.

Adapted from the Qwen3-0.6B megakernel to target the Qwen3-TTS talker:
  - Weights loaded from talker.* prefix in Qwen3-TTS-12Hz-0.6B-CustomVoice
  - Prefill (text tokens) runs in PyTorch: text_embedding(2048-dim) → text_projection → 1024-dim
  - Autoregressive decode (audio tokens) runs via megakernel: codec_embedding(1024-dim)
  - LM head: talker.codec_head.weight [3072, 1024] (not tied, audio vocab)
"""

import math
import os
import struct

import torch
import torch.nn.functional as F

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM   # 2048
KV_SIZE = 8 * HEAD_DIM   # 1024
MAX_SEQ_LEN = 4096
AUDIO_VOCAB_SIZE = 3072  # codec token vocab (talker output)
TEXT_EMBED_DIM = 2048    # text_embedding output dim before projection

# Special token ids (from config.json)
CODEC_BOS_ID   = 2149
CODEC_EOS_ID   = 2150
CODEC_PAD_ID   = 2148
LANGUAGE_IDS = {
    "english":    2050,
    "chinese":    2055,
    "japanese":   2058,
    "korean":     2064,
    "french":     2061,
    "german":     2053,
    "spanish":    2054,
    "russian":    2069,
    "portuguese": 2071,
    "italian":    2070,
}

_decode = torch.ops.qwen_megakernel_C.decode


def load_weights(model_path: str, verbose: bool = True):
    """Load Qwen3-TTS talker weights from a local model directory."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if verbose:
        print(f"Loading talker weights from {model_path} ...")

    # Collect all tensors from all safetensors shards
    state = {}
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            fpath = os.path.join(model_path, fname)
            with safe_open(fpath, framework="pt") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k).cuda().to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",  # reuse Qwen3 text tokenizer
        trust_remote_code=True,
    )

    # RoPE tables for audio token positions (standard RoPE for decode loop)
    # M-RoPE is applied during prefill in PyTorch; decode uses positional offset
    rope_theta = 1_000_000.0
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Per-layer weights (11 tensors per layer) — same key structure as Qwen3-0.6B
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"talker.model.layers.{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])

    weights = dict(
        # Megakernel decode weights
        embed_weight=state["talker.model.codec_embedding.weight"].contiguous(),  # [3072, 1024]
        layer_weights=layer_weights,
        final_norm_weight=state["talker.model.norm.weight"].contiguous(),
        lm_head_weight=state["talker.codec_head.weight"].contiguous(),           # [3072, 1024]
        cos_table=cos_table,
        sin_table=sin_table,
        # Prefill weights (PyTorch)
        text_embedding=state["talker.model.text_embedding.weight"].contiguous(), # [151936, 2048]
        text_proj_fc1_w=state["talker.text_projection.linear_fc1.weight"].contiguous(),
        text_proj_fc1_b=state["talker.text_projection.linear_fc1.bias"].contiguous(),
        text_proj_fc2_w=state["talker.text_projection.linear_fc2.weight"].contiguous(),
        text_proj_fc2_b=state["talker.text_projection.linear_fc2.bias"].contiguous(),
    )

    # Code predictor weights (keep as raw dict for CodePredictor to consume)
    cp_weights = {}
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            fpath = os.path.join(model_path, fname)
            with safe_open(fpath, framework="pt") as f:
                for k in f.keys():
                    if "code_predictor" in k:
                        cp_weights[k] = f.get_tensor(k).cuda().to(torch.bfloat16)

    weights["cp_weights"] = cp_weights

    del state
    torch.cuda.empty_cache()
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TalkerDecoder:
    """Qwen3-TTS talker decoder: text prompt → audio codec token stream.

    Prefill runs in PyTorch (text_embedding + text_projection).
    Autoregressive decode loop runs via the CUDA megakernel (codec_embedding).
    """

    def __init__(self, model_path: str, verbose: bool = True):
        from qwen_megakernel.code_predictor import CodePredictor
        weights, tokenizer = load_weights(model_path, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights
        self._code_predictor = CodePredictor(weights["cp_weights"])

        # Megakernel weights
        self._embed_weight       = weights["embed_weight"]
        self._final_norm_weight  = weights["final_norm_weight"]
        self._lm_head_weight     = weights["lm_head_weight"]
        self._cos_table          = weights["cos_table"]
        self._sin_table          = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # Prefill weights (kept on GPU as bf16 for F.linear calls)
        self._text_embedding   = weights["text_embedding"]
        self._text_proj_fc1_w  = weights["text_proj_fc1_w"]
        self._text_proj_fc1_b  = weights["text_proj_fc1_b"]
        self._text_proj_fc2_w  = weights["text_proj_fc2_w"]
        self._text_proj_fc2_b  = weights["text_proj_fc2_b"]

        # KV cache — sized for combined prefill + decode
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers for single-token megakernel decode
        f32  = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden    = torch.empty(HIDDEN_SIZE, **bf16)
        self._act       = torch.empty(HIDDEN_SIZE, **f32)
        self._res       = torch.empty(HIDDEN_SIZE, **f32)
        self._q         = torch.empty(Q_SIZE, **f32)
        self._k         = torch.empty(KV_SIZE, **f32)
        self._v         = torch.empty(KV_SIZE, **f32)
        self._attn_out  = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def _text_embed_and_project(self, token_ids: list[int]) -> torch.Tensor:
        """Embed text tokens [seq] → project to [seq, 1024] (talker hidden dim)."""
        ids = torch.tensor(token_ids, dtype=torch.long, device="cuda")
        x = self._text_embedding[ids].to(torch.float32)          # [seq, 2048]
        x = F.linear(x, self._text_proj_fc1_w.float(), self._text_proj_fc1_b.float())
        x = F.silu(x)
        x = F.linear(x, self._text_proj_fc2_w.float(), self._text_proj_fc2_b.float())
        return x.to(torch.bfloat16)                               # [seq, 1024]

    def prefill(self, text: str, language: str = "english") -> int:
        """Run PyTorch prefill over text tokens. Returns position after prefill.

        Populates KV cache for positions 0..prefill_len-1 so the megakernel
        decode loop can start immediately after.
        """
        self.reset()

        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        lang_id  = LANGUAGE_IDS.get(language, LANGUAGE_IDS["english"])

        # Sequence: [text tokens] + [lang_id] + [CODEC_BOS_ID]
        full_ids = text_ids + [lang_id, CODEC_BOS_ID]

        # Embed text tokens via text_embedding + projection
        hidden = self._text_embed_and_project(full_ids)  # [seq, 1024]

        # Run prefill through the 28 transformer layers in PyTorch
        # (uses the same weights as the megakernel — KV cache is shared)
        with torch.no_grad():
            self._pytorch_prefill(hidden)

        return self._position

    def _pytorch_prefill(self, hidden: torch.Tensor):
        """Simple PyTorch forward pass to fill the KV cache. Not optimised."""
        seq_len = hidden.shape[0]
        layer_weights = self._weights["layer_weights"]
        n = 11  # tensors per layer

        for i in range(NUM_LAYERS):
            w = layer_weights[i * n: (i + 1) * n]
            (rms_w, q_w, k_w, v_w, q_norm_w, k_norm_w,
             o_w, post_rms_w, gate_w, up_w, down_w) = w

            # RMSNorm input
            x = _rms_norm(hidden, rms_w)

            # QKV projections
            x_f = x.float()
            q = F.linear(x_f, q_w.float())             # [seq, 2048]
            k = F.linear(x_f, k_w.float())             # [seq, 1024]
            v = F.linear(x_f, v_w.float())             # [seq, 1024]

            # QK norms
            q = _rms_norm(q.reshape(seq_len, 16, HEAD_DIM), q_norm_w).reshape(seq_len, -1)
            k = _rms_norm(k.reshape(seq_len, 8,  HEAD_DIM), k_norm_w).reshape(seq_len, -1)

            # Reshape for heads before RoPE
            q = q.reshape(seq_len, 16, HEAD_DIM)
            k = k.reshape(seq_len,  8, HEAD_DIM)
            v = v.reshape(seq_len,  8, HEAD_DIM)

            # Apply RoPE
            positions = torch.arange(seq_len, device="cuda")
            q = _apply_rope(q, self._cos_table, self._sin_table, positions)
            k = _apply_rope(k, self._cos_table, self._sin_table, positions)

            # Store in KV cache
            self._k_cache[i, :, :seq_len, :] = k.permute(1, 0, 2).to(torch.bfloat16)
            self._v_cache[i, :, :seq_len, :] = v.permute(1, 0, 2).to(torch.bfloat16)

            # Attention (causal)
            attn_out = _causal_attention(q, k, v, self._attn_scale)  # [seq, 2048]
            attn_out = F.linear(attn_out.float(), o_w.float())        # [seq, 1024]
            hidden = hidden + attn_out.to(torch.bfloat16)

            # FFN
            x = _rms_norm(hidden, post_rms_w)
            gate = F.silu(F.linear(x.float(), gate_w.float()))
            up   = F.linear(x.float(), up_w.float())
            ffn  = F.linear(gate * up, down_w.float())
            hidden = hidden + ffn.to(torch.bfloat16)

        self._position = seq_len

    def step(self, token_id: int) -> int:
        """Decode one audio codec token via the megakernel. Returns next token id."""
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def generate_audio_tokens(
        self,
        text: str,
        language: str = "english",
        max_tokens: int = 2048,
    ):
        """Prefill on text, then yield audio codec tokens until EOS."""
        self.prefill(text, language=language)
        token = CODEC_BOS_ID
        for _ in range(max_tokens):
            token = self.step(token)
            if token == CODEC_EOS_ID:
                break
            yield token

    def generate_frames(
        self,
        text: str,
        language: str = "english",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        """Yield [16]-code frames until EOS. Each frame = one codec time step."""
        self.prefill(text, language=language)
        self._code_predictor.reset()
        token = CODEC_BOS_ID
        for _ in range(max_tokens):
            token = self.step(token)
            if token == CODEC_EOS_ID:
                break
            codes = self._code_predictor.step(
                g0_token=token,
                talker_embed_weight=self._embed_weight,
                temperature=temperature,
            )
            yield codes

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._code_predictor.reset()

    @property
    def position(self) -> int:
        return self._position


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight.float()).to(torch.bfloat16)


def _apply_rope(
    x: torch.Tensor,          # [seq, heads, head_dim]
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    positions: torch.Tensor,  # [seq]
) -> torch.Tensor:
    cos = cos_table[positions].unsqueeze(1)   # [seq, 1, head_dim]
    sin = sin_table[positions].unsqueeze(1)
    x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(torch.bfloat16)


def _causal_attention(
    q: torch.Tensor,  # [seq, 16, head_dim]
    k: torch.Tensor,  # [seq, 8,  head_dim]
    v: torch.Tensor,  # [seq, 8,  head_dim]
    scale: float,
) -> torch.Tensor:
    seq = q.shape[0]
    # Expand KV heads to match Q heads (GQA: 16 Q heads, 8 KV heads)
    k = k.repeat_interleave(2, dim=1)  # [seq, 16, head_dim]
    v = v.repeat_interleave(2, dim=1)

    q = q.permute(1, 0, 2).float()  # [16, seq, head_dim]
    k = k.permute(1, 0, 2).float()
    v = v.permute(1, 0, 2).float()

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [16, seq, seq]
    mask = torch.triu(torch.full((seq, seq), float("-inf"), device=q.device), diagonal=1)
    scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)          # [16, seq, head_dim]
    return out.permute(1, 0, 2).reshape(seq, -1)  # [seq, 2048]
