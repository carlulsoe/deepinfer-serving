from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass

from deepinfer_serving.engine.sampling_params import SamplingParams


class SequenceStatus(enum.IntEnum):
    """Mirrors upstream ``csrc/engine/sequence.hpp`` (subset)."""

    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6


_FINISH_REASONS = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH_CAPPED: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
    SequenceStatus.FINISHED_IGNORED: "prompt_length_exceeded",
}


def get_finished_reason(status: SequenceStatus) -> str:
    return _FINISH_REASONS.get(status, "")


class SequenceData:
    def __init__(self, prompt_token_ids: list[int]) -> None:
        self.token_ids = list(prompt_token_ids)
        self.prompt_len = len(self.token_ids)


class Sequence:
    """Python view of upstream ``Sequence`` (fields as bound in ``pybind_sequence.cpp``)."""

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: list[int],
        block_size: int,
        request_id: str,
        user_id: str,
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.seq_id = seq_id
        self.data = SequenceData(prompt_token_ids)
        self.block_size = block_size
        self.request_id = request_id
        self.user_id = user_id
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.status = SequenceStatus.WAITING
        self.schedulable = True

        self.block_table: list[int] = []
        self._num_cached_tokens = 0
        self.old_cached_tokens = 0
        self.cached_hit_tokens = 0
        self.state_id = -1
        self.swa_valid_start = 0
        self.last_copy_state_token_idx = 0
        # mini-only: block KV for [0, kv_reuse_tokens) was loaded from a cache hit and must not be overwritten.
        self.kv_reuse_tokens = 0

        self.chunked_rest_len = 0
        self.chunk_checkpoints: deque[int] = deque()

    @property
    def num_cached_tokens(self) -> int:
        """Number of tokens with KV cache ready."""
        return self._num_cached_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, value: int) -> None:
        # Upstream ``KvCacheState::set_num_tokens``: external injection also resets the baseline.
        self._num_cached_tokens = value
        self.old_cached_tokens = value

    def advance_cached_tokens(self, count: int) -> None:
        self.old_cached_tokens = self._num_cached_tokens
        self._num_cached_tokens += count

    def get_len(self) -> int:
        return len(self.data.token_ids)

    def get_prompt_len(self) -> int:
        return self.data.prompt_len

    def get_output_len(self) -> int:
        return self.get_len() - self.data.prompt_len

    def get_token_ids(self) -> list[int]:
        return self.data.token_ids

    def get_last_token_id(self) -> int:
        return self.data.token_ids[-1]

    def append_token_id(self, token_id: int) -> None:
        self.data.token_ids.append(token_id)

    def get_num_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size

    def is_finished(self) -> bool:
        return self.status >= SequenceStatus.FINISHED_STOPPED


@dataclass
class SequenceMetadata:
    """Per-step input for one scheduled sequence. Computes positions ``[seqlen_k - seqlen_q, seqlen_k)``."""

    seq: Sequence
    is_prompt: bool
    seqlen_q: int
    seqlen_k: int

    @property
    def request_id(self) -> str:
        return self.seq.request_id

    @property
    def is_last_prompt_chunk(self) -> bool:
        return self.is_prompt and self.seqlen_k >= self.seq.get_len()


@dataclass
class RequestOutput:
    """Subset of upstream ``csrc/engine/outputs.hpp`` ``RequestOutput``."""

    request_id: str
    prompt_len: int
    token_id: int
    finish_reason: str
    finished: bool
    num_output_tokens: int
