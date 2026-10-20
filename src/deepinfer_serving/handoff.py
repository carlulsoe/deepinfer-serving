"""What a PROMPT engine gives a COMPLETION engine (contract 10.1). Transports change how it moves, not what it is."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Handoff:
    request_id: str
    user_id: str
    sampling_params: dict
    prompt_len: int
    token_ids: list[int]
    """Prompt plus the first token sampled by the PROMPT engine."""
    num_cached_tokens: int
    """Equals ``prompt_len``: the first sampled token has no KV yet."""
    block_kv: torch.Tensor
    """``[ceil(num_cached_tokens / block_size), n_full_layers, 2, block_size, n_heads, head_dim]``, table order."""
    swa_state: torch.Tensor
    """SWA record at ``num_cached_tokens``: ``[n_swa_layers, 2, window + backtrack, n_heads, head_dim]``."""
    config_fingerprint: str

    def to_bytes(self) -> bytes:
        """Wire format of ADR 0007: JSON header + raw tensors, SHA-256 checked, no pickle."""
        from deepinfer_serving.wire import encode_handoff

        return encode_handoff(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> Handoff:
        """Parse and checksum-verify. Raises ``WireError``. Shapes are checked by the receiver."""
        from deepinfer_serving.wire import decode_handoff

        return decode_handoff(data)
