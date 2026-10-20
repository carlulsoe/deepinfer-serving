"""LLMEngine subclass that implements the hooks: prefix-cache hits, persistence, and PD handoff (contract 10)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.llm_engine import LLMEngine
from deepinfer_serving.engine.sampling_params import SamplingParams
from deepinfer_serving.engine.sequence import Sequence, SequenceStatus
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.kvstore import Hit, KVStore, block_hashes, compute_hit, swa_key


class ServingLLMEngine(LLMEngine):
    def __init__(
        self,
        config: MiniConfig,
        engine_type: EngineType = EngineType.NORMAL,
        device: str | torch.device = "cpu",
        *,
        store: KVStore | None = None,
        handoff_sink: Callable[[Handoff], None] | None = None,
    ) -> None:
        self.store = store
        self.handoff_sink = handoff_sink
        self._pending_prefill_saves: list[Sequence] = []
        super().__init__(config, engine_type, device)
        if store is not None:
            self.copy_state = torch.zeros_like(self.cache_engine.states)

    # ---- prefix-cache hit on request arrival ----

    def add_request(
        self,
        /,
        user_id: str,
        request_id: str,
        sampling_params: dict | SamplingParams | None,
        prompt_token_ids: np.ndarray,
        multi_modal_data: dict | None = None,
        arrival_time: float | None = None,
        extra: dict | None = None,
        enqueue: bool = True,
        chunk_checkpoints: list[int] | None = None,
    ) -> Sequence | None:
        seq = super().add_request(
            user_id,
            request_id,
            sampling_params,
            prompt_token_ids,
            multi_modal_data,
            arrival_time,
            extra,
            enqueue=False,
            chunk_checkpoints=chunk_checkpoints,
        )
        if seq is None or not enqueue:
            return seq
        if self.store is not None:
            c = self.config
            hit = compute_hit(
                self.store, seq.get_token_ids(), block_size=c.block_size, window=c.window, backtrack=c.backtrack
            )
            if hit.cached_hit_tokens > 0:
                self._apply_hit(seq, hit)
        self.scheduler.add_sequence(seq)
        return seq

    def _apply_hit(self, seq: Sequence, hit: Hit) -> None:
        assert self.store is not None
        if not self.scheduler.try_allocate_seq(seq):
            return  # no room now; serve it cold rather than block
        blocks: list[torch.Tensor] = []
        for key in hit.block_keys:
            block = self.store.get_block(key)
            if block is None:
                self.scheduler.free_seq(seq)  # evicted between match and fetch; serve cold
                return
            blocks.append(block)
        self.cache_engine.write_blocks(seq.block_table[: len(blocks)], torch.stack(blocks))
        if hit.swa_position is None:
            self.cache_engine.zero_state(seq.state_id)
        else:
            key = swa_key(hit.block_keys, hit.swa_position, self.config.grain)
            record = self.store.get_swa(key, hit.swa_position)
            assert record is not None
            self.cache_engine.load_swa_record(seq.state_id, record, hit.swa_position)
        seq.num_cached_tokens = hit.num_cached_tokens
        seq.cached_hit_tokens = hit.cached_hit_tokens
        seq.swa_valid_start = hit.swa_valid_start
        seq.kv_reuse_tokens = len(blocks) * self.config.block_size

    # ---- persistence ----

    def _save_blocks(self, seq: Sequence, num_tokens: int) -> None:
        """Store every full block of ``[0, num_tokens)`` that the store does not have yet."""
        assert self.store is not None
        keys = block_hashes(seq.get_token_ids()[:num_tokens], self.config.block_size)
        missing = [i for i, key in enumerate(keys) if not self.store.has_block(key)]
        if not missing:
            return
        kv = self.cache_engine.read_blocks([seq.block_table[i] for i in missing])
        for row, i in enumerate(missing):
            self.store.put_block(keys[i], kv[row])

    def _save_swa(self, seq: Sequence, slot: torch.Tensor, position: int) -> None:
        c = self.config
        assert self.store is not None
        if position < c.grain:
            return
        if seq.swa_valid_start and position - c.swa_window_len < seq.swa_valid_start:
            return  # the window would include positions this slot never held
        keys = block_hashes(seq.get_token_ids()[:position], c.block_size)
        self.store.put_swa(swa_key(keys, position, c.grain), position, self.cache_engine.swa_record(slot, position))

    def pe_prepare_save_kvcache(self, scheduled: list[Sequence], finished: list[Sequence]) -> None:
        # Only a sequence whose last prefill chunk runs this step stores; nothing is stored mid-chunk.
        if self.store is not None:
            self._pending_prefill_saves = [s for s in scheduled if s.chunked_rest_len == 0 and not s.is_finished()]

    def pe_do_save_kvcache(self, scheduled: list[Sequence], finished: list[Sequence]) -> None:
        g = self.config.grain
        for seq in self._pending_prefill_saves:
            length = seq.get_len()  # whole prompt now has KV; num_cached_tokens is not advanced yet
            self._save_blocks(seq, length)
            self._save_swa(seq, self.cache_engine.states[seq.state_id], (length - 1) // g * g)
        self._pending_prefill_saves = []

    def should_copy_state(self, seq: Sequence) -> bool:
        if self.store is None or seq.state_id == -1:
            return False
        g, n = self.config.grain, seq.num_cached_tokens
        boundary = n // g * g
        return boundary >= g and n // g > seq.last_copy_state_token_idx // g and n - boundary < self.config.backtrack

    def do_save_kvcache(self, scheduled: list[Sequence], finished: list[Sequence]) -> None:
        if self.store is None:
            return
        assert self.copy_state is not None
        g = self.config.grain
        for seq in finished:
            if seq.state_id == -1 or seq.status == SequenceStatus.FINISHED_ABORTED:
                continue
            self._save_blocks(seq, seq.num_cached_tokens)
            if seq.last_copy_state_token_idx >= g:
                position = seq.last_copy_state_token_idx // g * g
                self._save_swa(seq, self.copy_state[seq.state_id], position)

    # ---- PD handoff ----

    def send_processed_prompts(self) -> None:
        for seq in self.scheduler.running:
            if not seq.is_finished():
                assert self.handoff_sink is not None, "PROMPT engine needs a handoff_sink"
                self.handoff_sink(self._build_handoff(seq))
            self.scheduler.free_seq(seq)
        self.scheduler.running = []

    def _build_handoff(self, seq: Sequence) -> Handoff:
        n = seq.num_cached_tokens
        num_blocks = (n + self.config.block_size - 1) // self.config.block_size
        return Handoff(
            request_id=seq.request_id,
            user_id=seq.user_id,
            sampling_params=seq.sampling_params.to_dict(),
            prompt_len=seq.get_prompt_len(),
            token_ids=list(seq.get_token_ids()),
            num_cached_tokens=n,
            block_kv=self.cache_engine.read_blocks(seq.block_table[:num_blocks]),
            swa_state=self.cache_engine.swa_record(self.cache_engine.states[seq.state_id], n),
            config_fingerprint=self.config.fingerprint(),
        )

    def validate_handoff(self, handoff: Handoff) -> None:
        """Check a handoff against this engine's config before it touches device memory. Raises ``ValueError``."""
        c = self.config
        if handoff.config_fingerprint != c.fingerprint():
            raise ValueError("handoff was produced by an engine with a different model/cache config")
        n = handoff.num_cached_tokens
        if not (0 < handoff.prompt_len == n < len(handoff.token_ids) <= c.max_model_len):
            raise ValueError("inconsistent prompt_len / num_cached_tokens / token_ids")
        if any(not 0 <= t < c.vocab_size for t in handoff.token_ids):
            raise ValueError("token id out of range")
        num_blocks = (n + c.block_size - 1) // c.block_size
        block_shape = (num_blocks, c.n_full_layers, 2, c.block_size, c.n_heads, c.head_dim)
        swa_shape = (c.n_swa_layers, 2, c.swa_window_len, c.n_heads, c.head_dim)
        if tuple(handoff.block_kv.shape) != block_shape or tuple(handoff.swa_state.shape) != swa_shape:
            raise ValueError("block_kv or swa_state shape does not match this engine's cache layout")

    def receive_handoff(self, handoff: Handoff) -> Sequence | None:
        """Admit a handed-off sequence for decode. Returns ``None`` if there is no capacity (ADR 0005)."""
        c = self.config
        self.validate_handoff(handoff)
        seq = super().add_request(
            handoff.user_id,
            handoff.request_id,
            dict(handoff.sampling_params),
            np.asarray(handoff.token_ids[: handoff.prompt_len], dtype=np.int32),
            enqueue=False,
        )
        assert seq is not None
        for token in handoff.token_ids[handoff.prompt_len :]:
            seq.append_token_id(token)
        if not self.scheduler.try_allocate_seq(seq):
            return None
        self.cache_engine.write_blocks(seq.block_table[: handoff.block_kv.shape[0]], handoff.block_kv)
        self.cache_engine.load_swa_record(seq.state_id, handoff.swa_state, handoff.num_cached_tokens)
        seq.num_cached_tokens = handoff.num_cached_tokens
        # Positions below the handed-off window were never in this slot.
        seq.swa_valid_start = max(0, handoff.num_cached_tokens - c.swa_window_len)
        self.scheduler.add_prompted(seq)
        return seq
