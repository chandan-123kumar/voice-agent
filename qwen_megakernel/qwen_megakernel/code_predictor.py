"""Code predictor: expands a single talker group-0 token → 16 code groups.

Architecture:
  5-layer Qwen3 (16 Q-heads / 8 KV-heads GQA, head_dim=128, ffn_dim=3072)

Optimisations applied:
  1. Non-autoregressive within-frame: all 15 groups predicted from the same
     group-0 hidden state (one forward pass vs. 15).
  2. Persistent frame KV cache: O(t) work per frame instead of O(t²).
  3. SDPA (scaled_dot_product_attention) for efficient attention.
  4. bfloat16 throughout — no fp32 upcast in the hot path.
  5. Frame commit reuses group-0 K/V (approximation — avoids extra pass).
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

CP_NUM_LAYERS    = 5
CP_NUM_Q_HEADS   = 16
CP_NUM_KV_HEADS  = 8
CP_HEAD_DIM      = 128
CP_HIDDEN_SIZE   = 1024
CP_NUM_GROUPS    = 16
CP_MAX_FRAMES    = 2048
CP_ROPE_THETA    = 1_000_000.0
CP_GQA           = CP_NUM_Q_HEADS // CP_NUM_KV_HEADS   # = 2


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [H, 1, D]; cos/sin: [1, D]
    x1, x2 = x[..., :CP_HEAD_DIM//2], x[..., CP_HEAD_DIM//2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CodePredictor:
    """Per-utterance streaming code predictor with persistent frame KV cache."""

    def __init__(self, weights: dict):
        dev  = "cuda"
        bf16 = torch.bfloat16

        self._embed_tables: list[torch.Tensor] = [
            weights[f"talker.code_predictor.model.codec_embedding.{i}.weight"]
            .to(dev, bf16).contiguous()
            for i in range(CP_NUM_GROUPS - 1)
        ]
        self._lm_heads: list[torch.Tensor] = [
            weights[f"talker.code_predictor.lm_head.{i}.weight"]
            .to(dev, bf16).contiguous()
            for i in range(CP_NUM_GROUPS - 1)
        ]
        self._final_norm = weights[
            "talker.code_predictor.model.norm.weight"
        ].to(dev, bf16).contiguous()

        # Stack layer weights for efficient access
        self._lw: list[tuple] = []
        for i in range(CP_NUM_LAYERS):
            p = f"talker.code_predictor.model.layers.{i}."
            self._lw.append((
                weights[p + "input_layernorm.weight"].to(dev, bf16),
                weights[p + "self_attn.q_proj.weight"].to(dev, bf16),
                weights[p + "self_attn.k_proj.weight"].to(dev, bf16),
                weights[p + "self_attn.v_proj.weight"].to(dev, bf16),
                weights[p + "self_attn.q_norm.weight"].to(dev, bf16),
                weights[p + "self_attn.k_norm.weight"].to(dev, bf16),
                weights[p + "self_attn.o_proj.weight"].to(dev, bf16),
                weights[p + "post_attention_layernorm.weight"].to(dev, bf16),
                weights[p + "mlp.gate_proj.weight"].to(dev, bf16),
                weights[p + "mlp.up_proj.weight"].to(dev, bf16),
                weights[p + "mlp.down_proj.weight"].to(dev, bf16),
            ))

        # RoPE (bfloat16)
        inv_freq = 1.0 / (
            CP_ROPE_THETA ** (torch.arange(0, CP_HEAD_DIM, 2, dtype=torch.float32) / CP_HEAD_DIM)
        )
        pos   = torch.arange(CP_MAX_FRAMES, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        self._cos = torch.cos(freqs).repeat(1, 2).to(bf16).to(dev)   # [F, 128]
        self._sin = torch.sin(freqs).repeat(1, 2).to(bf16).to(dev)

        # Persistent frame KV cache — ONE slot per committed frame
        self._fk = torch.zeros(CP_NUM_LAYERS, CP_NUM_KV_HEADS, CP_MAX_FRAMES, CP_HEAD_DIM,
                               dtype=bf16, device=dev)
        self._fv = torch.zeros_like(self._fk)
        self._fp = 0   # how many frames are committed

    def reset(self):
        self._fk.zero_()
        self._fv.zero_()
        self._fp = 0

    def step(
        self,
        g0_token: int,
        talker_embed_weight: torch.Tensor,  # [3072, 1024] bfloat16
        temperature: float = 0.0,
    ) -> list[int]:
        fp  = self._fp
        g0e = talker_embed_weight[g0_token]    # [1024] bfloat16

        # Single forward pass through 5 layers for g0 (with past frame context)
        hidden = self._forward_one(g0e, fp)    # [1024]

        # Predict all 15 remaining groups from the same hidden state (non-AR)
        h_n = _rms_norm(hidden.unsqueeze(0), self._final_norm).squeeze(0)   # [1024]
        codes  = [g0_token]
        embeds = [g0e]

        for s in range(CP_NUM_GROUPS - 1):
            logits = F.linear(h_n, self._lm_heads[s])    # [2048] bf16
            if temperature <= 0.0:
                token = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                token = int(torch.multinomial(probs, 1).item())
            codes.append(token)
            embeds.append(self._embed_tables[s][token])

        # Commit: reuse group-0 K/V as this frame's cache entry (fast approximation)
        # The K/V were written during _forward_one; frame cache already updated.
        self._fp += 1
        return codes

    def empty_past_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 0, CP_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # ── internals ──────────────────────────────────────────────────────────────

    def _forward_one(self, emb: torch.Tensor, fp: int) -> torch.Tensor:
        """Run one token through 5 layers.  Writes K/V to frame cache at slot fp."""
        hidden  = emb                          # [1024] bf16
        cos_fp  = self._cos[fp]                # [128]
        sin_fp  = self._sin[fp]

        for i, (rms_w, q_w, k_w, v_w, q_norm_w, k_norm_w,
                o_w, post_w, gate_w, up_w, down_w) in enumerate(self._lw):

            x  = _rms_norm(hidden.unsqueeze(0), rms_w).squeeze(0)   # [1024]
            q  = F.linear(x, q_w).view(CP_NUM_Q_HEADS,  1, CP_HEAD_DIM)  # [16,1,128]
            k  = F.linear(x, k_w).view(CP_NUM_KV_HEADS, 1, CP_HEAD_DIM)  # [8,1,128]
            v  = F.linear(x, v_w).view(CP_NUM_KV_HEADS, 1, CP_HEAD_DIM)

            # QK norm + RoPE (keep bf16)
            q = _rms_norm(q, q_norm_w.view(1, 1, CP_HEAD_DIM))
            k = _rms_norm(k, k_norm_w.view(1, 1, CP_HEAD_DIM))
            q = _rope(q, cos_fp.view(1, 1, CP_HEAD_DIM), sin_fp.view(1, 1, CP_HEAD_DIM))
            k = _rope(k, cos_fp.view(1, 1, CP_HEAD_DIM), sin_fp.view(1, 1, CP_HEAD_DIM))

            # Write K/V to frame cache at slot fp
            self._fk[i, :, fp, :] = k.squeeze(1)
            self._fv[i, :, fp, :] = v.squeeze(1)

            # Full context: frame cache 0..fp (inclusive)
            ctx_k = self._fk[i, :, :fp + 1, :]   # [8, fp+1, 128]
            ctx_v = self._fv[i, :, :fp + 1, :]

            # GQA expansion: replicate KV heads to match Q heads
            ctx_k = ctx_k.repeat_interleave(CP_GQA, dim=0)   # [16, ctx, 128]
            ctx_v = ctx_v.repeat_interleave(CP_GQA, dim=0)

            # SDPA — [batch=1, heads=16, q_len=1, kv_len=ctx]
            q_sdpa = q.unsqueeze(0)          # [1, 16, 1, 128]
            k_sdpa = ctx_k.unsqueeze(0)      # [1, 16, ctx, 128]
            v_sdpa = ctx_v.unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=False
            )                                                  # [1, 16, 1, 128]
            attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(1, -1)  # [1, 2048]
            attn_out = F.linear(attn_out, o_w).squeeze(0)     # [1024]
            hidden   = hidden + attn_out

            # FFN (SiLU gated)
            x    = _rms_norm(hidden.unsqueeze(0), post_w).squeeze(0)
            gate = F.silu(F.linear(x, gate_w))
            up   = F.linear(x, up_w)
            ffn  = F.linear(gate * up, down_w)
            hidden = hidden + ffn

        return hidden   # [1024]
