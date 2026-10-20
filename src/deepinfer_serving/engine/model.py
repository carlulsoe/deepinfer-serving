"""Tiny decoder-only transformer (mini-only, not a mirror). See contract section 8.

Two properties make the oracles exact:

- Token-serial: positions are computed one at a time, so the ops for position ``i`` have identical shapes and inputs
  regardless of how ``i`` was reached (prefill, chunk, decode, cache hit, handoff).
- SWA-layer K/V are projected from the normalized token-embedding stream, so an SWA entry is a pure function of
  (token, position, layer) and a one-block replay rebuilds the window exactly. Full-attention K/V come from the
  residual stream as usual, so a wrong SWA window propagates into block KV that later positions read. Block KV
  loaded from a cache hit (``Sequence.kv_reuse_tokens``) is reused as-is and never overwritten by the replay.
"""

import math

import torch

from deepinfer_serving.engine.cache_engine import CacheEngine
from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.sequence import SequenceMetadata

_EPS = 1e-6


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean() + _EPS) * weight


class MiniModel:
    def __init__(self, config: MiniConfig, device: torch.device) -> None:
        c = config
        self.config = c
        self.device = device
        gen = torch.Generator().manual_seed(c.seed)

        def rand(*shape: int) -> torch.Tensor:
            # Generated on CPU so every device gets identical weights.
            return (torch.randn(*shape, generator=gen) / math.sqrt(shape[0])).to(device)

        hd = c.n_heads * c.head_dim
        self.embed = torch.randn(c.vocab_size, c.d_model, generator=gen).to(device)
        self.kv_norm = torch.ones(c.d_model, device=device)
        self.layers = []
        for kind in c.layer_pattern:
            self.layers.append(
                {
                    "kind": kind,
                    "attn_norm": torch.ones(c.d_model, device=device),
                    "wq": rand(c.d_model, hd),
                    "wk": rand(c.d_model, hd),
                    "wv": rand(c.d_model, hd),
                    "wo": rand(hd, c.d_model),
                    "mlp_norm": torch.ones(c.d_model, device=device),
                    "w1": rand(c.d_model, c.d_ff),
                    "w2": rand(c.d_ff, c.d_model),
                }
            )
        self.final_norm = torch.ones(c.d_model, device=device)
        self.lm_head = rand(c.d_model, c.vocab_size)
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim)).to(device)
        self.scale = 1.0 / math.sqrt(c.head_dim)

    def _rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        angles = self.inv_freq * position
        cos, sin = torch.cos(angles), torch.sin(angles)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    @torch.inference_mode()
    def forward(self, meta: SequenceMetadata, cache: CacheEngine) -> torch.Tensor | None:
        """Compute positions ``[seqlen_k - seqlen_q, seqlen_k)``, writing K/V into the cache.

        Returns logits at the last position if this step samples (decode or last prefill chunk), else ``None``.
        """
        c = self.config
        seq = meta.seq
        start, end = meta.seqlen_k - meta.seqlen_q, meta.seqlen_k
        tokens = seq.get_token_ids()
        block_table = torch.tensor(seq.block_table, dtype=torch.long, device=self.device)
        positions = torch.arange(end, device=self.device)
        pos_block = block_table[positions // c.block_size]
        pos_offset = positions % c.block_size
        state_id = seq.state_id
        logits = None

        for j in range(start, end):
            e = self.embed[tokens[j]]
            kv_src = _rms_norm(e, self.kv_norm)
            x = e
            full_idx = swa_idx = 0
            for layer in self.layers:
                h = _rms_norm(x, layer["attn_norm"])
                q = self._rope((h @ layer["wq"]).view(c.n_heads, c.head_dim), j)
                src = h if layer["kind"] == "full" else kv_src
                k = self._rope((src @ layer["wk"]).view(c.n_heads, c.head_dim), j)
                v = (src @ layer["wv"]).view(c.n_heads, c.head_dim)
                if layer["kind"] == "full":
                    if j >= seq.kv_reuse_tokens:  # loaded block KV is reused as-is, never overwritten
                        blk, off = seq.block_table[j // c.block_size], j % c.block_size
                        cache.blocks[blk, full_idx, 0, off] = k
                        cache.blocks[blk, full_idx, 1, off] = v
                    keys = cache.blocks[pos_block[: j + 1], full_idx, 0, pos_offset[: j + 1]]
                    values = cache.blocks[pos_block[: j + 1], full_idx, 1, pos_offset[: j + 1]]
                    full_idx += 1
                else:
                    slot = cache.states[state_id, swa_idx]
                    slot[0, j % c.ring_size] = k
                    slot[1, j % c.ring_size] = v
                    lo = max(j - c.window + 1, seq.swa_valid_start, 0)
                    ring_idx = torch.arange(lo, j + 1, device=self.device) % c.ring_size
                    keys, values = slot[0, ring_idx], slot[1, ring_idx]
                    swa_idx += 1
                scores = torch.einsum("hd,nhd->hn", q, keys) * self.scale
                attn = torch.einsum("hn,nhd->hd", torch.softmax(scores, dim=-1), values)
                x = x + attn.reshape(-1) @ layer["wo"]
                x = x + torch.nn.functional.silu(_rms_norm(x, layer["mlp_norm"]) @ layer["w1"]) @ layer["w2"]
            if j == end - 1 and (not meta.is_prompt or meta.is_last_prompt_chunk):
                logits = _rms_norm(x, self.final_norm) @ self.lm_head
        return logits
