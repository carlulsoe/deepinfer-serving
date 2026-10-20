"""Mirrors upstream ``csrc/engine/scheduler.hpp`` without MTP, ZOS, forks, preemption, or DP votes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from deepinfer_serving.engine.block_manager import AllocStatus, BlockManager
from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.sequence import Sequence, SequenceMetadata, SequenceStatus


@dataclass
class SchedulerOutputs:
    scheduled_seqs: list[Sequence]
    prompt_run: bool
    num_batched_tokens: int
    finished_seqs: list[Sequence] = field(default_factory=list)
    ignored_seqs: list[Sequence] = field(default_factory=list)
    seqlens_q: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.scheduled_seqs


class Scheduler:
    def __init__(
        self,
        config: MiniConfig,
        engine_type: EngineType,
        block_manager: BlockManager,
        state_manager: BlockManager,
    ) -> None:
        self.config = config
        self.engine_type = engine_type
        self.block_manager = block_manager
        self.state_manager = state_manager
        self.waiting: deque[Sequence] = deque()
        self.prompted: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    # ---- queue management (upstream bindings) ----

    def add_sequence(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def add_prompted(self, seq: Sequence) -> None:
        """Queue a sequence prefilled elsewhere for decode. Upstream has the queue but no binding (contract A4)."""
        self.prompted.append(seq)

    def abort_sequence(self, request_ids: list[str]) -> list[Sequence]:
        aborted = []
        for queue in (self.running, self.waiting, self.prompted):
            for seq in queue:
                if seq.request_id in request_ids and not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    aborted.append(seq)
        return aborted

    def has_unfinished_seqs(self) -> bool:
        return self.get_num_unfinished_seqs() > 0

    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.prompted)

    # ---- allocation ----

    def _can_allocate_with_state(self, num_blocks: int) -> AllocStatus:
        status = self.block_manager.can_allocate(num_blocks)
        if status == AllocStatus.OK and self.state_manager.can_allocate(1) == AllocStatus.LATER:
            return AllocStatus.LATER
        return status

    def _allocate_seq(self, seq: Sequence) -> None:
        seq.block_table = self.block_manager.allocate(seq.get_num_blocks())
        seq.state_id = self.state_manager.allocate_single()

    def try_allocate_seq(self, seq: Sequence) -> bool:
        assert seq.status == SequenceStatus.WAITING, seq.status
        assert not seq.block_table and seq.state_id == -1
        status = self._can_allocate_with_state(seq.get_num_blocks())
        if status == AllocStatus.NEVER:
            raise RuntimeError(f"can never allocate {seq.get_num_blocks()} blocks")
        if status == AllocStatus.LATER:
            return False
        self._allocate_seq(seq)
        return True

    def free_seq(self, seq: Sequence) -> None:
        if seq.state_id != -1:
            self.state_manager.free_blocks([seq.state_id])
            seq.state_id = -1
        self.block_manager.free_blocks(seq.block_table)
        seq.block_table = []

    def _append_slot(self, seq: Sequence) -> None:
        num_blocks = seq.get_num_blocks()
        if len(seq.block_table) < num_blocks:
            if self.block_manager.can_allocate(1) != AllocStatus.OK:
                raise RuntimeError("out of KV blocks during decode (mini-DeepInfer does not preempt)")
            seq.block_table.append(self.block_manager.allocate_single())
        assert len(seq.block_table) == num_blocks

    # ---- scheduling ----

    def schedule(self) -> tuple[list[SequenceMetadata], SchedulerOutputs]:
        if self.engine_type == EngineType.PROMPT:
            outputs = self._schedule_prefill()
        elif self.engine_type == EngineType.COMPLETION:
            outputs = self._schedule_decode()
        else:
            outputs = None
            if self.waiting and self.waiting[0].schedulable:
                outputs = self._schedule_prefill()
                if not outputs.scheduled_seqs and not outputs.ignored_seqs:
                    outputs = None
            if outputs is None:
                outputs = self._schedule_decode()

        metadata = []
        for seq, seqlen_q in zip(outputs.scheduled_seqs, outputs.seqlens_q, strict=True):
            seqlen_k = seq.num_cached_tokens + seqlen_q
            metadata.append(SequenceMetadata(seq, outputs.prompt_run, seqlen_q, seqlen_k))
        if not outputs.prompt_run:
            self.running = list(outputs.scheduled_seqs)
        return metadata, outputs

    def _schedule_prefill(self) -> SchedulerOutputs:
        cfg = self.config
        scheduled: list[Sequence] = []
        ignored: list[Sequence] = []
        seqlens_q: list[int] = []
        num_curr_seqs = len(self.running)
        num_batched_tokens = 0

        while self.waiting:
            seq = self.waiting[0]
            if seq.is_finished():  # aborted before prefill
                if seq.block_table:
                    self.free_seq(seq)
                ignored.append(self.waiting.popleft())
                continue
            if not seq.schedulable:
                break
            num_prompt_tokens = seq.get_len() - seq.num_cached_tokens
            if not seq.block_table:
                if seq.get_len() > cfg.max_model_len:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored.append(self.waiting.popleft())
                    continue
                status = self._can_allocate_with_state(seq.get_num_blocks())
                if status == AllocStatus.LATER:
                    break
                if status == AllocStatus.NEVER:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored.append(self.waiting.popleft())
                    continue
            if num_curr_seqs + 1 > cfg.max_num_seqs:
                break

            seqlen_q = min(num_prompt_tokens, cfg.max_num_batched_tokens - num_batched_tokens)
            start = seq.num_cached_tokens
            while seq.chunk_checkpoints and seq.chunk_checkpoints[0] <= start:
                seq.chunk_checkpoints.popleft()
            if seq.chunk_checkpoints and start < seq.chunk_checkpoints[0] <= start + seqlen_q:
                seqlen_q = seq.chunk_checkpoints.popleft() - start
            if seqlen_q <= 0:
                break
            is_chunked = seqlen_q < num_prompt_tokens
            if is_chunked and not cfg.chunked_prefill:
                break

            # Committed: side effects below.
            num_batched_tokens += seqlen_q
            if not seq.block_table:
                self._allocate_seq(seq)
            if not is_chunked:
                num_curr_seqs += 1
                seq.status = SequenceStatus.RUNNING
            seq.chunked_rest_len = seq.get_len() - (start + seqlen_q) if is_chunked else 0
            scheduled.append(seq)
            seqlens_q.append(seqlen_q)
            self.waiting.popleft()
            if is_chunked and self.engine_type == EngineType.NORMAL:
                break

        return SchedulerOutputs(scheduled, True, num_batched_tokens, ignored_seqs=ignored, seqlens_q=seqlens_q)

    def _schedule_decode(self) -> SchedulerOutputs:
        finished = [seq for seq in self.running if seq.is_finished()]
        self.running = [seq for seq in self.running if not seq.is_finished()]

        ignored = [seq for seq in self.waiting if seq.is_finished()]
        self.waiting = deque(seq for seq in self.waiting if not seq.is_finished())

        while self.prompted and len(self.running) < self.config.max_num_seqs:
            seq = self.prompted.popleft()
            if seq.status == SequenceStatus.WAITING:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                finished.append(seq)

        for seq in self.running:
            self._append_slot(seq)
        scheduled = sorted(self.running, key=lambda s: s.get_len(), reverse=True)
        return SchedulerOutputs(scheduled, False, len(scheduled), finished, ignored, [1] * len(scheduled))

    # ---- outputs ----

    def process_outputs(self, outputs: SchedulerOutputs, sampled: list[int | None]) -> list[tuple[Sequence, int]]:
        """Commit one step. Returns ``(seq, token)`` for every sequence that produced a token."""
        produced = []
        for seq, seqlen_q, token in zip(outputs.scheduled_seqs, outputs.seqlens_q, sampled, strict=True):
            if seq.is_finished():  # aborted while in flight
                if outputs.prompt_run:
                    self.running.append(seq)
                continue
            seq.advance_cached_tokens(seqlen_q)
            if seq.num_cached_tokens >= seq.get_len():
                assert token is not None
                if outputs.prompt_run:
                    self.running.append(seq)
                seq.append_token_id(token)
                self._check_stop(seq)
                produced.append((seq, token))
            else:
                assert outputs.prompt_run and seq.status == SequenceStatus.WAITING
                self.waiting.appendleft(seq)
        return produced

    def _check_stop(self, seq: Sequence) -> None:
        if not seq.sampling_params.ignore_eos and seq.get_last_token_id() == self.config.eos_token_id:
            seq.status = SequenceStatus.FINISHED_STOPPED
        elif seq.get_len() > self.config.max_model_len or seq.get_output_len() == seq.sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
