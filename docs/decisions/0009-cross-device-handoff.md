# ADR 0009: Cross-device handoffs are allowed but not oracle-checked

Status: accepted

## Context

`config_fingerprint` covers the model and cache shape and the weight seed, but not the device. A
CPU prefill can hand off to a CUDA decode, and an SM86 prefill to an SM100 decode. The data is
compatible, but the arithmetic differs, so bit-exactness across devices is not guaranteed. In
production, prefill and decode may well run on different SKUs.

## Decision

- Receivers accept handoffs from any device. The fingerprint stays device-free.
- The exact oracles compare runs on one device type: NORMAL on device X against PROMPT on X →
  COMPLETION on X.
- One GPU-marked smoke test runs a CPU → CUDA handoff and checks that the request completes with
  the right length and frees its resources. It does not check token equality.

## Consequences

- Heterogeneous P/D deployments are not blocked by a format check.
- Token equality across devices is not claimed anywhere. If a later milestone needs it (for
  example, comparing a mini-DeepInfer prefill with a real DeepInfer decode), it needs its own
  tolerance-based check, decided before the run.
