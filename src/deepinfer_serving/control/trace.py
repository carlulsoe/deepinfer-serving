"""Request traces: one JSON object per line (ADR 0014).

A trace carries each request's hit length as data. With a shared, equal-cost cache the hit does not depend on which
instance serves the request (ADR 0001), so replaying a trace under a different balancer does not change it.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Request:
    id: str
    arrival_s: float
    session: str
    input_len: int
    hit_len: int
    output_len: int
    patches: int = 0
    """Image patches for the vision encoder; 0 for a text-only request."""

    def __post_init__(self) -> None:
        if not (0 <= self.hit_len < self.input_len and self.output_len >= 1 and self.patches >= 0):
            raise ValueError(f"invalid request {self}")

    @property
    def uncached(self) -> int:
        return self.input_len - self.hit_len


def write_trace(path: Path, requests: Iterable[Request]) -> None:
    with path.open("w") as f:
        for r in requests:
            f.write(json.dumps(asdict(r)) + "\n")


def read_trace(path: Path) -> list[Request]:
    with path.open() as f:
        requests = [Request(**json.loads(line)) for line in f if line.strip()]
    return sorted(requests, key=lambda r: (r.arrival_s, r.id))


class _Record(Protocol):
    request_id: str
    prompt: list[int]
    user_id: str
    submitted_at: float
    hit: tuple[int, int, int] | None
    num_output_tokens: int | None


def trace_from_records(records: Iterable[_Record]) -> list[Request]:
    """A trace from a tier-1 run: the Entrypoint's finished ``RequestRecord`` objects."""
    done = [r for r in records if r.num_output_tokens]
    if not done:
        return []
    t0 = min(r.submitted_at for r in done)
    return sorted(
        (
            Request(
                r.request_id,
                r.submitted_at - t0,
                r.user_id,
                len(r.prompt),
                r.hit[0] if r.hit else 0,
                r.num_output_tokens or 0,
            )
            for r in done
        ),
        key=lambda r: (r.arrival_s, r.id),
    )


def synthetic_trace(
    seed: int,
    *,
    sessions: int,
    duration_s: float,
    turns: float = 4.0,
    first_input: int = 20_000,
    user_tokens: int = 1_000,
    output: int = 500,
    think_s: float = 20.0,
    image_fraction: float = 0.0,
    block: int = 128,
    tail: int = 134,
) -> list[Request]:
    """Multi-turn sessions starting uniformly over ``duration_s``. Lengths are log-normal around the given medians.

    Turn k + 1's input is turn k's input and output plus new user tokens, and it hits what turn k left in the
    store: the block-aligned prefix of turn k's context, capped at ``input - tail`` (the hit rule's cap).
    """
    rng = random.Random(seed)

    def lognormal(median: float, sigma: float = 1.0) -> int:
        return max(1, round(rng.lognormvariate(0, sigma) * median))

    out = []
    for s in range(sessions):
        t = rng.uniform(0, duration_s)
        context, n_turns = 0, max(1, round(rng.expovariate(1 / turns)))
        for k in range(n_turns):
            new = lognormal(first_input if k == 0 else user_tokens)
            input_len = context + new
            hit = min(context // block * block, max(0, input_len - tail) // block * block)
            patches = lognormal(1_000, 0.5) if rng.random() < image_fraction else 0
            out_len = lognormal(output)
            out.append(Request(f"s{s}t{k}", round(t, 6), f"s{s}", input_len, hit, out_len, patches))
            context = input_len + out_len
            t += rng.expovariate(1 / think_s)
    return sorted(out, key=lambda r: (r.arrival_s, r.id))
