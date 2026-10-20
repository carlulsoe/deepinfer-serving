import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MiniConfig:
    """Collapses upstream's model/cache/parallel/scheduler configs into what mini-DeepInfer uses."""

    # model
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 2
    head_dim: int = 32
    d_ff: int = 128
    layer_pattern: tuple[str, ...] = ("full", "swa", "full", "swa")
    eos_token_id: int = 1
    seed: int = 0
    # cache
    block_size: int = 128
    window: int = 128
    backtrack: int = 6
    store_extra: int = 128
    num_gpu_blocks: int = 64
    num_state_slots: int = 8
    # scheduler
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 4096
    chunked_prefill: bool = True
    max_model_len: int = 4096

    def __post_init__(self) -> None:
        if set(self.layer_pattern) - {"full", "swa"}:
            raise ValueError(f"unknown layer kind in {self.layer_pattern}")
        if self.window > self.block_size:
            raise ValueError("window must be <= block_size (the one-block back-off relies on it)")
        if self.backtrack > self.block_size:
            raise ValueError("backtrack must be <= block_size")
        if self.store_extra < self.block_size:
            raise ValueError("store_extra must be >= block_size so the after-prefill record fits in the ring")
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError("max_num_batched_tokens must be >= block_size (upstream check)")

    @property
    def grain(self) -> int:
        """Upstream ``KVSTORE_GRAIN``: SWA persists only at multiples of this. Equal to block size."""
        return self.block_size

    @property
    def n_full_layers(self) -> int:
        return self.layer_pattern.count("full")

    @property
    def n_swa_layers(self) -> int:
        return self.layer_pattern.count("swa")

    @property
    def swa_window_len(self) -> int:
        """Inference window, upstream ``get_inference_swa_block_size``: what persists and is handed off."""
        return self.window + self.backtrack

    @property
    def ring_size(self) -> int:
        """Slot ring, upstream ``get_kvstore_swa_block_size`` with extra state enabled."""
        return self.window + self.backtrack + self.store_extra

    def fingerprint(self) -> str:
        """Fields that must match between engines exchanging cache contents."""
        fields = asdict(self)
        keep = [
            "vocab_size",
            "d_model",
            "n_heads",
            "head_dim",
            "d_ff",
            "layer_pattern",
            "eos_token_id",
            "seed",
            "block_size",
            "window",
            "backtrack",
        ]
        return json.dumps({k: fields[k] for k in keep}, sort_keys=True)
