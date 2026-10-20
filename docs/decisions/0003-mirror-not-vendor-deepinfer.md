# ADR 0003: Do not vendor DeepInfer; mirror its interface at a pinned commit

Status: accepted

## Context

Our code must match DeepInfer's engine interface exactly, but DeepInfer cannot be built here. Its
`vendor/` submodules (FlashMLA, DeepGEMM, DeepEP and their Ascend variants, DeepSelect) need SM100
or Ascend toolchains [code; DS]. DeepInfer is also young and changing: the Q&A is "updated
continuously" [DS]. Vendoring or submoduling it would put an unbuildable tree in the repo, and the
interface we target would drift with every update.

## Decision

- DeepInfer is not vendored and not a submodule. Contributors read it from a shallow fetch of the
  pinned commit, as described in [`reference/DEEPINFER_PIN.md`](../../reference/DEEPINFER_PIN.md).
- The pin is `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (2026-10-20). mini-DeepInfer mirrors the
  upstream files listed there: names, hook call order, and field meanings.
- Every deviation from upstream semantics is recorded in
  [interface-contract.md](../interface-contract.md). Changing a mirrored name, the hook order or a
  field meaning requires updating the contract in the same change.
- Moving the pin is a deliberate change: update `DEEPINFER_PIN.md`, diff the mirrored files, and
  update mini-DeepInfer and the contract together.

## Consequences

- The repo builds and tests anywhere Python and PyTorch run.
- Claims about DeepInfer's code can be checked against a fixed commit and cited by permalink.
- Upstream fixes and new hooks do not reach us automatically. Someone has to move the pin.
- At milestone 5 the real DeepInfer is installed separately on the SM100 node at the pinned commit.
  It is still not part of this repo.
