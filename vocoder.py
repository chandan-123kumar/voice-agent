"""Qwen3-TTS speech tokenizer decoder (Qwen3TTSTokenizerV2Model).

Decodes [T, 16] code indices → float32 PCM waveform at 24kHz.

Architecture:
  RVQ decode → pre_conv → pre_transformer (8-layer, 512-dim)
  → upsample (×2 × 2 = ×4) → convolutional decoder (×8×5×4×3 = ×480)
  Total upsample: 1920× (12.5Hz → 24000Hz)
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


def load_vocoder(model_path: str, device: str = "cpu") -> "VocoderDecoder":
    """Load vocoder. Defaults to CPU since conv decode is fast there."""
    from safetensors import safe_open
    speech_tok = os.path.join(model_path, "speech_tokenizer", "model.safetensors")
    state = {}
    with safe_open(speech_tok, framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return VocoderDecoder(state, device=device)


class VocoderDecoder:
    """Decode 16-group RVQ codes → PCM waveform."""

    def __init__(self, state: dict, device: str = "cpu"):
        dev = device

        # ── RVQ codebooks ───────────────────────────────────────────────────
        def _get_embed(pfx):
            es = state[f"{pfx}._codebook.embedding_sum"].to(dev, torch.float32)
            cu = state[f"{pfx}._codebook.cluster_usage"].to(dev, torch.float32).clamp(min=1)
            return es / cu.unsqueeze(-1)  # [2048, 256]

        self._cb_first = _get_embed("decoder.quantizer.rvq_first.vq.layers.0")
        self._cb_rest = [
            _get_embed(f"decoder.quantizer.rvq_rest.vq.layers.{i}")
            for i in range(15)
        ]
        self._rvq_first_out = state["decoder.quantizer.rvq_first.output_proj.weight"].to(dev)
        self._rvq_rest_out  = state["decoder.quantizer.rvq_rest.output_proj.weight"].to(dev)

        # ── pre_conv ────────────────────────────────────────────────────────
        self._pc_w = state["decoder.pre_conv.conv.weight"].to(dev)  # [1024, 512, 3]
        self._pc_b = state["decoder.pre_conv.conv.bias"].to(dev)

        # ── pre_transformer ─────────────────────────────────────────────────
        self._pt_in_w  = state["decoder.pre_transformer.input_proj.weight"].to(dev)   # [512, 1024]
        self._pt_in_b  = state["decoder.pre_transformer.input_proj.bias"].to(dev)
        self._pt_out_w = state["decoder.pre_transformer.output_proj.weight"].to(dev)  # [1024, 512]
        self._pt_out_b = state["decoder.pre_transformer.output_proj.bias"].to(dev)
        self._pt_norm_w = state["decoder.pre_transformer.norm.weight"].to(dev)        # [512]
        self._pt_layers = []
        for i in range(8):
            p = f"decoder.pre_transformer.layers.{i}."
            self._pt_layers.append(dict(
                in_norm_w=state[p + "input_layernorm.weight"].to(dev),
                post_norm_w=state[p + "post_attention_layernorm.weight"].to(dev),
                q_w=state[p + "self_attn.q_proj.weight"].to(dev),
                k_w=state[p + "self_attn.k_proj.weight"].to(dev),
                v_w=state[p + "self_attn.v_proj.weight"].to(dev),
                o_w=state[p + "self_attn.o_proj.weight"].to(dev),
                gate_w=state[p + "mlp.gate_proj.weight"].to(dev),
                up_w=state[p + "mlp.up_proj.weight"].to(dev),
                down_w=state[p + "mlp.down_proj.weight"].to(dev),
                attn_scale=state[p + "self_attn_layer_scale.scale"].to(dev),
                mlp_scale=state[p + "mlp_layer_scale.scale"].to(dev),
            ))

        # RoPE for pre_transformer (theta=10000, head_dim=64, num_heads=16)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        positions = torch.arange(8000, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._pt_cos = torch.cos(freqs).repeat(1, 2).to(dev)  # [8000, 64]
        self._pt_sin = torch.sin(freqs).repeat(1, 2).to(dev)
        self._pt_attn_scale = 1.0 / math.sqrt(64)
        self._pt_sw = 72  # sliding window size

        # ── upsample blocks (2× each, 2 total = 4×) ─────────────────────────
        self._up = []
        for j in range(2):
            p = f"decoder.upsample.{j}."
            self._up.append(dict(
                # ConvTranspose1d: weight [in, out, kernel] → stride 2 upsample
                ct_w=state[p + "0.conv.weight"].to(dev),
                ct_b=state[p + "0.conv.bias"].to(dev),
                # ConvNeXt block
                dw_w=state[p + "1.dwconv.conv.weight"].to(dev),
                dw_b=state[p + "1.dwconv.conv.bias"].to(dev),
                norm_w=state[p + "1.norm.weight"].to(dev),
                norm_b=state[p + "1.norm.bias"].to(dev),
                pw1_w=state[p + "1.pwconv1.weight"].to(dev),
                pw1_b=state[p + "1.pwconv1.bias"].to(dev),
                pw2_w=state[p + "1.pwconv2.weight"].to(dev),
                pw2_b=state[p + "1.pwconv2.bias"].to(dev),
                gamma=state[p + "1.gamma"].to(dev),
            ))

        # ── convolutional decoder ────────────────────────────────────────────
        self._d0_w = state["decoder.decoder.0.conv.weight"].to(dev)  # [1536, 1024, 7]
        self._d0_b = state["decoder.decoder.0.conv.bias"].to(dev)

        # Upsampling ConvTranspose blocks: strides 8,5,4,3
        strides = [8, 5, 4, 3]
        self._d_blocks = []
        channels = [(1536, 768), (768, 384), (384, 192), (192, 96)]
        kernels_ct = [16, 10, 8, 6]

        for idx, ((in_ch, out_ch), stride, k_ct) in enumerate(zip(channels, strides, kernels_ct)):
            blk_idx = idx + 1
            p = f"decoder.decoder.{blk_idx}.block."
            # block.0: layer scale (alpha/beta activation)
            # block.1: ConvTranspose1d
            # block.2-4: residual conv blocks
            block_data = dict(
                alpha=state[p + "0.alpha"].to(dev),
                beta=state[p + "0.beta"].to(dev),
                ct_w=state[p + "1.conv.weight"].to(dev),
                ct_b=state[p + "1.conv.bias"].to(dev),
                stride=stride,
                res_blocks=[],
            )
            for rb in range(2, 5):
                bp = f"{p}{rb}."
                block_data["res_blocks"].append(dict(
                    a1=state[bp + "act1.alpha"].to(dev),
                    b1=state[bp + "act1.beta"].to(dev),
                    a2=state[bp + "act2.alpha"].to(dev),
                    b2=state[bp + "act2.beta"].to(dev),
                    c1_w=state[bp + "conv1.conv.weight"].to(dev),
                    c1_b=state[bp + "conv1.conv.bias"].to(dev),
                    c2_w=state[bp + "conv2.conv.weight"].to(dev),
                    c2_b=state[bp + "conv2.conv.bias"].to(dev),
                ))
            self._d_blocks.append(block_data)

        # Final norm + output conv
        self._d5_alpha = state["decoder.decoder.5.alpha"].to(dev)
        self._d5_beta  = state["decoder.decoder.5.beta"].to(dev)
        self._d6_w = state["decoder.decoder.6.conv.weight"].to(dev)  # [1, 96, 7]
        self._d6_b = state["decoder.decoder.6.conv.bias"].to(dev)

    # ── public API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes [T, 16] → PCM waveform [T*1920] float32."""
        T = codes.shape[0]
        dev = self._cb_first.device
        codes = codes.to(dev, torch.long)

        # ── 1. RVQ decode ────────────────────────────────────────────────────
        # group 0: rvq_first
        z0 = self._cb_first[codes[:, 0]]  # [T, 256]
        z0 = F.conv1d(z0.unsqueeze(-1), self._rvq_first_out).squeeze(-1)  # [T, 512]

        z = z0.clone()
        for i in range(1, 16):
            zi = self._cb_rest[i - 1][codes[:, i]]     # [T, 256]
            zi = F.conv1d(zi.unsqueeze(-1), self._rvq_rest_out).squeeze(-1)  # [T, 512]
            z = z + zi

        # ── 2. pre_conv ──────────────────────────────────────────────────────
        # z: [T, 512] → [1, 512, T] → conv → [1, 1024, T] → [T, 1024]
        z_conv = F.conv1d(
            z.t().unsqueeze(0),       # [1, 512, T]
            self._pc_w,
            self._pc_b,
            padding=1,                # same-length output
        )                             # [1, 1024, T]
        z_conv = z_conv.squeeze(0).t()  # [T, 1024]

        # ── 3. pre_transformer ───────────────────────────────────────────────
        h = F.linear(z_conv, self._pt_in_w, self._pt_in_b)  # [T, 512]
        positions = torch.arange(T, device=dev)

        for lw in self._pt_layers:
            # Pre-norm (RMSNorm approximation via LayerNorm weights — model uses RMSNorm)
            x = _rms_norm(h, lw["in_norm_w"])

            # QKV (16 heads, head_dim=64, total=1024)
            q = F.linear(x, lw["q_w"]).reshape(T, 16, 64)  # [T, 16, 64]
            k = F.linear(x, lw["k_w"]).reshape(T, 16, 64)
            v = F.linear(x, lw["v_w"]).reshape(T, 16, 64)

            # RoPE
            cos = self._pt_cos[positions]
            sin = self._pt_sin[positions]
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            # Attention with sliding window
            attn = _sliding_window_attention(q, k, v, self._pt_sw, self._pt_attn_scale)
            attn = F.linear(attn.float(), lw["o_w"].float())  # [T, 512]
            h = h + (attn * lw["attn_scale"].float()).to(h.dtype)

            # FFN
            x = _rms_norm(h, lw["post_norm_w"])
            gate = F.silu(F.linear(x, lw["gate_w"]))
            up   = F.linear(x, lw["up_w"])
            ffn  = F.linear(gate * up, lw["down_w"])
            h = h + (ffn * lw["mlp_scale"]).to(h.dtype)

        h = _rms_norm(h, self._pt_norm_w)
        h = F.linear(h, self._pt_out_w, self._pt_out_b)  # [T, 1024]

        # ── 4. upsample (×4 total) ───────────────────────────────────────────
        # [T, 1024] → [1, 1024, T]
        x = h.t().unsqueeze(0)
        for ub in self._up:
            # ConvTranspose1d stride=2 (kernel=2 → stride inferred from weight)
            ct_w = ub["ct_w"]                   # [in_ch, out_ch, 2]
            stride = ct_w.shape[-1]             # = 2
            x = F.conv_transpose1d(x, ct_w, ub["ct_b"], stride=stride)

            # ConvNeXt: depthwise + 2-layer pointwise
            res = x
            x_dw = F.conv1d(x, ub["dw_w"], ub["dw_b"], padding=3, groups=x.shape[1])
            ch = x_dw.shape[1]
            x_ln = F.layer_norm(x_dw.permute(0, 2, 1), [ch], ub["norm_w"], ub["norm_b"])
            x_ln = x_ln.permute(0, 2, 1)
            x_pw = F.silu(F.linear(x_ln.permute(0, 2, 1), ub["pw1_w"], ub["pw1_b"]))
            x_pw = F.linear(x_pw, ub["pw2_w"], ub["pw2_b"])
            x_pw = x_pw.permute(0, 2, 1)
            x = res + x_pw * ub["gamma"].unsqueeze(0).unsqueeze(-1)

        # ── 5. convolutional decoder ─────────────────────────────────────────
        x = F.conv1d(x, self._d0_w, self._d0_b, padding=3)  # [1, 1536, T*4]

        for blk in self._d_blocks:
            # Snake activation via alpha/beta scale
            alpha = blk["alpha"].unsqueeze(0).unsqueeze(-1)
            beta  = blk["beta"].unsqueeze(0).unsqueeze(-1)
            x = _snake_activation(x, alpha, beta)

            # ConvTranspose1d upsampling
            ct_w   = blk["ct_w"]
            stride = blk["stride"]
            pad    = ct_w.shape[-1] // 2
            x = F.conv_transpose1d(x, ct_w, blk["ct_b"], stride=stride,
                                   padding=pad, output_padding=stride - 1)

            # Residual blocks
            for rb in blk["res_blocks"]:
                res = x
                a1 = rb["a1"].unsqueeze(0).unsqueeze(-1)
                b1 = rb["b1"].unsqueeze(0).unsqueeze(-1)
                x = _snake_activation(x, a1, b1)
                x = F.conv1d(x, rb["c1_w"], rb["c1_b"], padding=3)
                a2 = rb["a2"].unsqueeze(0).unsqueeze(-1)
                b2 = rb["b2"].unsqueeze(0).unsqueeze(-1)
                x = _snake_activation(x, a2, b2)
                x = F.conv1d(x, rb["c2_w"], rb["c2_b"], padding=0)
                x = x + res

        # Final activation + output conv
        alpha5 = self._d5_alpha.unsqueeze(0).unsqueeze(-1)
        beta5  = self._d5_beta.unsqueeze(0).unsqueeze(-1)
        x = _snake_activation(x, alpha5, beta5)
        x = F.conv1d(x, self._d6_w, self._d6_b, padding=3)  # [1, 1, T*1920]

        return x.squeeze().float()  # [T*1920]


# ── helpers ───────────────────────────────────────────────────────────────────

def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [T, heads, head_dim], cos/sin: [T, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1 = x[..., :32]
    x2 = x[..., 32:]
    return (x.float() * cos.float() + torch.cat([-x2, x1], dim=-1).float() * sin.float()).to(x.dtype)


def _sliding_window_attention(
    q: torch.Tensor,   # [T, H, D]
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Full causal attention clipped to sliding window (simplified: use full for short seqs)."""
    T, H, D = q.shape
    q = q.permute(1, 0, 2).float()   # [H, T, D]
    k = k.permute(1, 0, 2).float()
    v = v.permute(1, 0, 2).float()

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [H, T, T]

    # Causal mask
    mask = torch.triu(torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    # Sliding window: block attention beyond window
    if T > window:
        sw_mask = torch.tril(
            torch.full((T, T), float("-inf"), device=q.device), diagonal=-(window + 1)
        )
        mask = mask + sw_mask

    scores = scores + mask
    attn_w = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_w, v)                         # [H, T, D]
    return out.permute(1, 0, 2).reshape(T, -1).to(q.dtype)  # [T, H*D]


def _snake_activation(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Snake activation: x + sin²(alpha * x) / alpha."""
    return x + torch.sin(alpha * x) ** 2 / (alpha + 1e-8) * beta
