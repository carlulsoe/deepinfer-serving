# ADR 0016: The autoscaler as a pure reconcile function

Status: accepted

## Context

DeepSeek scales prefill on uncached tokens and decode on running requests. Adjustments are
rate-limited and bounded, ready machines are kept as reserve, and scale-in is forbidden above 0.8
quota utilization. Decode drains get a one-minute grace period before their remaining requests are
forwarded. A failed instance restarts with EP set to its healthy GPU count [DS]. Roadmap S3 asks for
these rules as `reconcile()`, with invariants checked on bursty and failure traces, and SLO
attainment against GPU-hours compared with a static allocation.

## Decision

**`reconcile(spec, observed, telemetry) -> list[Action]`** (`control/autoscaler.py`) is pure. The
actions are `ScaleUp(role)`, `Drain(name)`, `Forward(name)`, `Replace(name, gpus)`,
`Provision(gpus)` and `Release(gpus)`. The simulator applies them today; a cluster backend can
apply the same ones later.

**Per role, each reconcile:**

1. **Replace** every dead instance that has healthy GPUs, with EP re-formed at that count. An
   instance with none is left for scaling to cover.
2. **Desired count:** `ceil(demand / (capacity_per_gpu × instance_gpus × target_utilization))`,
   clamped to `[min, max]`.
   - Prefill demand is the uncached-token arrival rate plus queued uncached tokens divided by
     `backlog_s`.
   - Decode demand is running plus waiting requests.
3. **Scale up** by at most `max_step` per `cooldown_s`.
4. **Scale down** (drain the instances with the fewest requests) by at most `max_step` per
   `cooldown_s`, and only if all of these hold:
   - quota utilization (demand over what the *serving* instances sustain) is at most
     `no_scale_down_above` (0.8);
   - no instance of the role is starting;
   - none is awaiting replacement.

Across roles: **Forward** a draining decode instance that still has requests once it has drained for
`drain_grace_s`. Keep the warm reserve at `warm_gpus` with **Provision** and **Release**; a scale-up
uses warm GPUs when enough are ready.

The simulator retires a draining instance once it is empty. That is mechanism, not policy, so it is
not an action. Forwarding moves sequences with their KV, as ADR 0013 does, so nothing is
re-prefilled.

**Ambiguities resolved here [ours].**

- **The demand signal includes backlog.** Arrival rate alone under-sizes prefill once a queue has
  built up. That is the moment scaling matters most.
- **No scale-down while an instance starts.** A starting instance counts toward the instance count
  but not toward measured capacity. Without this rule the autoscaler drained the only serving decode
  instance while its replacement was still starting, which left decode capacity at zero for over a
  minute (found in the simulator). The rule also stops flapping during a ramp.
- **Telemetry is averaged over a 60 s window** (`SimConfig.telemetry_window_s`). A 10 s window
  flapped on heavy-tailed prompts.
- **The reserve is counted in GPUs, not nodes.** That avoids mapping node shapes onto instance
  sizes; the spec sketch's `warm_nodes` is a node count.
- **Restarts take minutes [DS].** Replacements and cold scale-ups start in `startup_s`, warm
  scale-ups in `warm_startup_s`. The reserve's GPUs count toward GPU-hours.

## Consequences

On the bursty trace of `tests/test_autoscaler.py`, the load is quiet for 10 minutes, about 15×
higher in uncached tokens for nearly 7, then quiet again. Instances have 2 GPUs, startup is 120 s
(15 s warm) and the SLO is TTFT ≤ 2 s with mean TBT ≤ 10 ms:

| Fleet | GPU-hours | SLO attainment |
|---|---|---|
| autoscaled, 1–6 instances per role, 2 warm GPUs | 3.97 | 0.91 |
| static, 2 prefill + 2 decode | 3.90 | 0.79 |
| static, 3 + 3 | 5.84 | 0.96 |
| static, 4 + 4 | 7.79 | 0.96 |

- **Against static fleets:** no static fleet is both cheaper and better. At about the same
  GPU-hours the autoscaler gets 12 points more attainment. Against the smallest good static fleet it
  uses 32% fewer GPU-hours for 5 points less.
- **Where it loses:** almost all of the loss is at the start of the burst, while instances start. A
  bigger warm reserve trades GPU-hours for that.
- **Invariants checked on the bursty trace, with and without one prefill and one decode failure:**
  - nothing is lost or duplicated;
  - rate limits and bounds hold at every reconcile;
  - no drain happens above 0.8 utilization;
  - every drained instance retires;
  - failed instances are replaced at their healthy GPU count.

  A separate trace checks that long requests are forwarded after the grace period without a
  re-prefill.
- **Not yet built:**
  - a cluster backend;
  - the live Entrypoint driving `reconcile`: the Entrypoint has `drain` and `retire` but no way to
    start instances;
  - the deployment spec's constraint checks.
- All numbers are relative to the uncalibrated cost model (ADR 0014).
