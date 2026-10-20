"""Cache layout. Mirrors the shape of upstream ``deepinfer/worker/cache_engine.py`` (block KV + per-seq state slot)."""

import torch

from deepinfer_serving.engine.config import MiniConfig


class CacheEngine:
    """Owns the device cache tensors.

    - ``blocks[num_gpu_blocks, n_full_layers, 2, block_size, n_heads, head_dim]``: paged block KV. One block holds
      ``block_size`` consecutive positions for every full-attention layer, so it is the unit of persistence.
    - ``states[num_state_slots, n_swa_layers, 2, ring_size, n_heads, head_dim]``: one slot per sequence. Position
      ``p`` lives at ring index ``p % ring_size``.

    An SWA *record at position p* is the inference window ending at ``p``: positions
    ``[p - window - backtrack, p)`` in position order, zero-filled below 0.
    Shape ``[n_swa_layers, 2, window + backtrack, n_heads, head_dim]``.
    """

    def __init__(self, config: MiniConfig, device: torch.device) -> None:
        c = config
        self.config = config
        self.device = device
        self.blocks = torch.zeros(
            c.num_gpu_blocks, c.n_full_layers, 2, c.block_size, c.n_heads, c.head_dim, device=device
        )
        self.states = torch.zeros(
            c.num_state_slots, c.n_swa_layers, 2, c.ring_size, c.n_heads, c.head_dim, device=device
        )

    def read_blocks(self, block_ids: list[int]) -> torch.Tensor:
        return self.blocks[torch.tensor(block_ids, dtype=torch.long, device=self.device)].cpu()

    def write_blocks(self, block_ids: list[int], kv: torch.Tensor) -> None:
        self.blocks[torch.tensor(block_ids, dtype=torch.long, device=self.device)] = kv.to(self.device)

    def _window_positions(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(position - self.config.swa_window_len, position, device=self.device)
        valid = pos >= 0
        return pos, valid

    def swa_record(self, slot: torch.Tensor, position: int) -> torch.Tensor:
        """Extract the record at ``position`` from a slot (``states[i]`` or a replica of it)."""
        pos, valid = self._window_positions(position)
        c = self.config
        record = torch.zeros(c.n_swa_layers, 2, c.swa_window_len, c.n_heads, c.head_dim, device=self.device)
        record[:, :, valid] = slot[:, :, pos[valid] % c.ring_size]
        return record.cpu()

    def load_swa_record(self, state_id: int, record: torch.Tensor, position: int) -> None:
        """Zero the slot and write the record so that the ring holds the window ending at ``position``."""
        slot = self.states[state_id]
        slot.zero_()
        pos, valid = self._window_positions(position)
        slot[:, :, pos[valid] % self.config.ring_size] = record.to(self.device)[:, :, valid]

    def zero_state(self, state_id: int) -> None:
        self.states[state_id].zero_()
