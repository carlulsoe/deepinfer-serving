"""mini-DeepInfer ``LLMEngine``. Mirrors upstream ``deepinfer/engine/llm_engine.py`` (see the interface contract).

The "virtual functions" block is kept exactly as upstream: no-ops (``finalize_seq`` frees the sequence). Serving code
fills them in by subclassing.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from deepinfer_serving.engine.block_manager import BlockManager
from deepinfer_serving.engine.cache_engine import CacheEngine
from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.model import MiniModel
from deepinfer_serving.engine.sampling_params import SamplingParams
from deepinfer_serving.engine.scheduler import Scheduler, SchedulerOutputs
from deepinfer_serving.engine.sequence import (
    RequestOutput,
    Sequence,
    SequenceMetadata,
    get_finished_reason,
)


@dataclass
class WorkerStepState:
    """Upstream ``deepinfer/worker/zos_status.py`` ``WorkerStepState`` (subset)."""

    seq_metadata_list: list[SequenceMetadata]
    scheduler_outputs: SchedulerOutputs
    sampled_token_ids: list[int | None]


class LLMEngine:
    def __init__(
        self,
        config: MiniConfig,
        engine_type: EngineType = EngineType.NORMAL,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.engine_type = engine_type
        self.device = torch.device(device)
        self.eos_id = config.eos_token_id
        self.seq_counter = itertools.count()
        self.generate_step = 0

        self.model = MiniModel(config, self.device)
        # _init_cache
        self.cache_engine = CacheEngine(config, self.device)
        if self.engine_type != EngineType.NORMAL:
            self.init_pd_split()

        self.block_manager = BlockManager(config.num_gpu_blocks)
        self.state_manager = BlockManager(config.num_state_slots)
        self.scheduler = Scheduler(config, engine_type, self.block_manager, self.state_manager)

        # Replica region for decode-time state snapshots. None unless a subclass allocates it (as upstream).
        self.copy_state: torch.Tensor | None = None

    # ---- requests ----

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
        del extra  # accepted for signature parity; unused
        if multi_modal_data is not None:
            raise ValueError("mini-DeepInfer does not support multimodal input")
        if sampling_params is None:
            sampling_params = {}
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**{k: v for k, v in sampling_params.items() if v is not None})
        sampling_params.eos_id = self.eos_id
        if arrival_time is None:
            arrival_time = time.perf_counter()

        prompt_token_ids = np.asarray(prompt_token_ids)
        if prompt_token_ids.size == 0:
            raise ValueError("prompt must not be empty")
        if prompt_token_ids.min() < 0 or prompt_token_ids.max() >= self.config.vocab_size:
            return None

        seq = Sequence(
            seq_id=next(self.seq_counter),
            prompt_token_ids=[int(t) for t in prompt_token_ids],
            block_size=self.config.block_size,
            request_id=request_id,
            user_id=user_id,
            sampling_params=sampling_params,
            arrival_time=arrival_time,
        )
        if chunk_checkpoints is not None:
            seq.chunk_checkpoints.extend(chunk_checkpoints)
        if enqueue:
            self.scheduler.add_sequence(seq)
        return seq

    def abort_request(self, request_id: str | Iterable[str]) -> tuple[int, int]:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        seqs = self.scheduler.abort_sequence(request_ids)
        return len(seqs), sum(seq.get_len() for seq in seqs)

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_num_unfinished_seqs()

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_seqs()

    # ---- step ----

    def step(self) -> list[RequestOutput]:
        seq_metadata_list, scheduler_outputs = self.scheduler.schedule()
        self.post_schedule(scheduler_outputs)

        if not scheduler_outputs.prompt_run and self.copy_state is not None:
            to_copy_state_seqs = [
                seq
                for seq in scheduler_outputs.scheduled_seqs + scheduler_outputs.finished_seqs
                if self.should_copy_state(seq)
            ]
            if to_copy_state_seqs:
                self.do_copy_state(to_copy_state_seqs)

        if not scheduler_outputs.is_empty() and self.engine_type == EngineType.PROMPT:
            assert scheduler_outputs.prompt_run, "Prompt engines should not perform decoding"

        # Worker.execute_model_step: forward + sample, pre_sampler_sync, post-process (non-ZOS path).
        sampled = self._execute_model(seq_metadata_list)
        self.pre_sampler_sync(scheduler_outputs)
        request_outputs = self.post_process(WorkerStepState(seq_metadata_list, scheduler_outputs, sampled))

        self.generate_step += 1
        return request_outputs

    def _execute_model(self, seq_metadata_list: list[SequenceMetadata]) -> list[int | None]:
        sampled: list[int | None] = []
        for meta in seq_metadata_list:  # one sequence at a time: trivially batch-invariant
            logits = self.model.forward(meta, self.cache_engine)
            sampled.append(None if logits is None else int(torch.argmax(logits)))
        return sampled

    def _process_model_outputs(
        self, sampled: list[int | None], scheduler_outputs: SchedulerOutputs
    ) -> list[RequestOutput]:
        produced = self.scheduler.process_outputs(scheduler_outputs, sampled)
        outputs = [
            RequestOutput(
                request_id=seq.request_id,
                prompt_len=seq.get_prompt_len(),
                token_id=token,
                finish_reason=get_finished_reason(seq.status),
                finished=seq.is_finished(),
                num_output_tokens=seq.get_output_len(),
            )
            for seq, token in produced
        ]
        outputs.extend(
            RequestOutput(
                request_id=seq.request_id,
                prompt_len=seq.get_prompt_len(),
                token_id=-1,
                finish_reason=get_finished_reason(seq.status),
                finished=True,
                num_output_tokens=seq.get_output_len(),
            )
            for seq in scheduler_outputs.ignored_seqs
        )
        return outputs

    def do_copy_state(self, seqs: list[Sequence]) -> None:
        """Copy each seq's state slot into the replica region and record ``last_copy_state_token_idx``."""
        if self.copy_state is None:
            return
        for seq in seqs:
            if seq.state_id != -1:
                self.copy_state[seq.state_id].copy_(self.cache_engine.states[seq.state_id])
                seq.last_copy_state_token_idx = seq.num_cached_tokens

    # 这里开始都是虚函数 (virtual functions start here)
    def init_pd_split(self) -> None:
        "Virtual: initialize data structures for PD disaggregation."

    def finalize_seq(self, seq: Sequence) -> None:
        "Virtual: handle a finished request."
        self.scheduler.free_seq(seq)

    def post_schedule(self, scheduler_outputs: SchedulerOutputs) -> None:
        "Virtual: handle each step's scheduler result."

    def should_copy_state(self, seq: Sequence) -> bool:
        "Virtual: whether to snapshot the seq's current state."
        return False

    def do_save_kvcache(self, scheduled: list[Sequence], finished: list[Sequence]) -> None:
        "Virtual: save kvcache; called from pre_sampler_sync on decode steps."

    def pe_prepare_save_kvcache(self, scheduled: list[Sequence], finished: list[Sequence]) -> None:
        "Virtual: prompt engine prepares to save kvcache."

    def pe_do_save_kvcache(self, scheduled: list[Sequence], finished: list[Sequence]) -> None:
        "Virtual: prompt engine saves kvcache."

    def process_output(self, output: list[RequestOutput]) -> None:
        "Virtual: handle one step's output tokens."

    def pre_sampler_sync(self, scheduler_outputs: SchedulerOutputs) -> None:
        if not scheduler_outputs.prompt_run:
            self.do_save_kvcache(scheduler_outputs.scheduled_seqs, scheduler_outputs.finished_seqs)
        else:
            self.pe_prepare_save_kvcache(scheduler_outputs.scheduled_seqs, scheduler_outputs.finished_seqs)

        if self.engine_type in (EngineType.COMPLETION, EngineType.NORMAL):
            for seq in scheduler_outputs.finished_seqs:
                self.finalize_seq(seq)

    def send_processed_prompts(self) -> None:
        "Virtual: send prefilled sequences to a completion engine and remove them from the scheduler."

    def post_process_model_outputs(self) -> None:
        "Virtual: called after processing model outputs, before logging."

    # 虚函数结束 (virtual functions end)

    def post_process(self, step_state: WorkerStepState) -> list[RequestOutput]:
        scheduler_outputs = step_state.scheduler_outputs
        if scheduler_outputs.prompt_run:
            self.pe_do_save_kvcache(scheduler_outputs.scheduled_seqs, scheduler_outputs.finished_seqs)
        request_outputs = self._process_model_outputs(step_state.sampled_token_ids, scheduler_outputs)
        self.post_process_model_outputs()
        if self.engine_type == EngineType.PROMPT:
            self.send_processed_prompts()
        return request_outputs

    def flush_inflight_reqs(self) -> list[RequestOutput]:
        """No zero-overhead scheduling, so nothing is ever in flight."""
        request_outputs: list[RequestOutput] = []
        self.process_output(request_outputs)
        return request_outputs
