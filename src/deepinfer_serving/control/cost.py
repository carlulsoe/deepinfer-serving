"""GPU-time model per DP rank (ADR 0014). Every default is an uncalibrated placeholder [ours], derived from
DeepSeek-reported production figures [DS]; fitting it to tier-2 profiles is milestone 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_CHUNK = 32_768
"""Per-GPU chunk cap of the prefill LB's level 4 [DS]."""


@dataclass(frozen=True)
class CostModel:
    prefill_ns_per_token: float = 6_000.0
    """Linear prefill work per token (MLP, MoE, projections)."""
    prefill_ns_per_pair: float = 0.145
    """Attention per (query token, key token) pair. With the linear term this gives 16.7 µs per uncached token at
    74K context, i.e. DeepSeek's 60K uncached tok/s per GPU at 77K average input and 96% hit [DS]."""
    prefill_step_ns: float = 1e6
    """Fixed cost of one prefill step (launches, the EP all-to-all's latency floor)."""
    quota_limit_ns: float = 1e8
    """``compute_quota_limit``: the attention-time budget of one prefill step per rank."""
    decode_step_ns: float = 1e6
    decode_ns_per_request: float = 20_000.0
    decode_ns_per_kv_token: float = 0.45
    """With the two terms above: a 4.35 ms step at 61 requests of 77K context, which is DeepSeek's 14K tok/s per GPU
    at 230 tok/s per user [DS]."""
    decode_kv_capacity: int = 8_000_000
    """KV tokens one decode rank holds."""
    transfer_ns_per_token: float = 0.018
    """P → D handoff: 896 B/token at 50 GB/s per GPU."""

    def cost(self, q: int, c: int) -> float:
        """``cost(q, c)``: predicted prefill time of ``q`` new tokens after ``c`` computed ones, in ns. Each new token
        attends to the prefix and, causally, to the new tokens before it."""
        return q * self.prefill_ns_per_token + self.prefill_ns_per_pair * q * (c + (q + 1) / 2)

    def solve_chunk(self, budget_ns: float, c: int, remaining: int) -> int:
        """Level 4: the largest ``q <= min(remaining, MAX_CHUNK)`` with ``cost(q, c) <= budget_ns``, possibly 0."""
        limit = min(remaining, MAX_CHUNK)
        if budget_ns <= 0 or limit <= 0:
            return 0
        a = self.prefill_ns_per_pair / 2
        b = self.prefill_ns_per_token + self.prefill_ns_per_pair * (c + 0.5)
        q = budget_ns / b if a == 0 else (-b + math.sqrt(b * b + 4 * a * budget_ns)) / (2 * a)
        q = min(limit, max(0, math.floor(q)))
        while q > 0 and self.cost(q, c) > budget_ns:  # float rounding at the boundary
            q -= 1
        while q < limit and self.cost(q + 1, c) <= budget_ns:
            q += 1
        return q

    def decode_step(self, requests: int, kv_tokens: int) -> float:
        """One decode step of a rank, in ns; 0 when the rank is empty."""
        if requests == 0:
            return 0.0
        return self.decode_step_ns + self.decode_ns_per_request * requests + self.decode_ns_per_kv_token * kv_tokens
