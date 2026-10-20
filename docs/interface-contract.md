# mini-DeepInfer interface contract

Status: **draft for review**. Upstream reference: DeepInfer
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (see `reference/DEEPINFER_PIN.md`). Upstream paths
below are relative to that checkout. "Q&A" means `doc/inference_deployment_qa.md`.

mini-DeepInfer is the seam every later serving piece plugs into. It mirrors upstream's
*interface and cache shape*, not its math or performance.

## 1. Layering

```
src/deepinfer_serving/
  engine/            mini-DeepInfer. Mirrors upstream. Hooks are no-ops, exactly as upstream.
  kvstore.py         KV store interface, in-memory store, hash chain, hit rule.      (serving)
  handoff.py         PROMPT → COMPLETION handoff data type.                          (serving)
  serving_engine.py  LLMEngine subclass that implements the hooks using the two above. (serving)
```

`engine/` must not import from the serving modules. Upstream ships `LLMEngine` with no-op
"virtual functions" and withholds the subclass that fills them in; we keep the same split, so the
serving subclass is our reconstruction of the withheld piece.

## 2. `EngineType` (mirrors `deepinfer/engine/engine_type.py`)

| Value | Meaning | Mirrored |
|---|---|---|
| `NORMAL = 0` | prompts and completions | yes |
| `PROMPT = 1` | prefill only | yes |
| `COMPLETION = 2` | decode only | yes |

`default_zero_overhead_scheduling` is not mirrored (no ZOS; see §9).

## 3. `LLMEngine` public surface (mirrors `deepinfer/engine/llm_engine.py`)

### Construction

Upstream: `LLMEngine(model_config, cache_config, parallel_config, scheduler_config, log_stats, engine_type, offline_inference)`.
mini: `LLMEngine(config: MiniConfig, engine_type: EngineType = NORMAL, device: str = "cpu")`.
`MiniConfig` collapses the four config objects into the fields mini-DeepInfer uses (model dims,
layer pattern, `block_size`, `window`, `backtrack`, `store_extra`, `num_gpu_blocks`,
`num_state_slots`, `max_num_seqs`, `max_num_batched_tokens`, `chunked_prefill`, `max_model_len`,
`eos_token_id`, `seed`).

As upstream (`_init_cache`), `init_pd_split()` is called once after the cache is allocated, only
when `engine_type != NORMAL`.

### `add_request` — same signature as upstream

```python
def add_request(self, /, user_id: str, request_id: str,
                sampling_params: dict | SamplingParams | None,
                prompt_token_ids: np.ndarray,          # 1-D int32
                multi_modal_data: dict | None = None,
                arrival_time: float | None = None,
                extra: dict | None = None,
                enqueue: bool = True,
                chunk_checkpoints: list[int] | None = None) -> Sequence | None
```

Same semantics: empty prompt raises `ValueError`; out-of-vocab tokens return `None`;
`enqueue=False` creates the `Sequence` without adding it to the scheduler (the caller schedules it,
e.g. after loading cache). `multi_modal_data` must be `None`. `extra` is accepted and ignored.
`chunk_checkpoints` is honored by the prefill scheduler as upstream.

`SamplingParams`: we keep `max_tokens`, `ignore_eos`, `temperature`, `eos_id`. Only greedy decoding
is implemented; `temperature != 0` is rejected.

### Other methods mirrored

`step()`, `abort_request(request_id)`, `has_unfinished_requests()`,
`get_num_unfinished_requests()`, `flush_inflight_reqs()` (returns no outputs; no ZOS),
`do_copy_state(seqs)`, `pre_sampler_sync(scheduler_outputs)`, `post_process(step_state)`.

`step()` returns `list[RequestOutput]` instead of an array of pooled pointers. `RequestOutput`
keeps `request_id`, `prompt_len`, `token_id`, `finish_reason`, `finished`, `num_output_tokens`.
One output per scheduled sequence that produced a token this step, as upstream.

## 4. One `step()`, and when each hook fires

Upstream spreads one step across `LLMEngine.step`, `Worker.execute_model_step`,
`LLMEngine.pre_sampler_sync` and `LLMEngine.post_process`. mini runs the same sequence inline
(non-ZOS path):

1. `seq_metadata_list, scheduler_outputs = scheduler.schedule()`
2. *(upstream: clone/zero states and blocks for forked sequences — not mirrored, §9)*
3. **`post_schedule(scheduler_outputs)`**
4. Decode steps only, if `self.copy_state is not None`: for each seq in
   `scheduled_seqs + finished_seqs`, if **`should_copy_state(seq)`** → `do_copy_state(seqs)`,
   which copies the seq's state slot into `copy_state[state_id]` and sets
   `seq.last_copy_state_token_idx = seq.num_cached_tokens`.
5. Forward + greedy sample.
6. `pre_sampler_sync(scheduler_outputs)`:
   - decode step → **`do_save_kvcache(scheduled_seqs, finished_seqs)`**
   - prompt step → **`pe_prepare_save_kvcache(scheduled_seqs, finished_seqs)`**
   - if engine is COMPLETION or NORMAL: **`finalize_seq(seq)`** for each finished seq
     (base implementation: `scheduler.free_seq(seq)`).
7. `post_process(step_state)`:
   - prompt step → **`pe_do_save_kvcache(scheduled_seqs, finished_seqs)`**
   - `_process_model_outputs` → `scheduler.process_outputs(...)` appends tokens, advances
     `num_cached_tokens`, checks stop conditions; builds `RequestOutput`s.
   - **`post_process_model_outputs()`**
   - if engine is PROMPT: **`send_processed_prompts()`**
8. Return outputs.

Other hooks:

| Hook | Fires | Base behavior |
|---|---|---|
| `init_pd_split()` | once at construction, non-NORMAL only | no-op |
| `process_output(outputs)` | from `flush_inflight_reqs()`; otherwise the caller (future online server) calls it after `step()` | no-op |

Hook names, arguments, and order are the contract. Upstream behavior we rely on and keep:

- Prefill (`pe_*`) hooks run while the step's sequences still hold their blocks;
  `pe_do_save_kvcache` runs *before* `process_outputs`, so `num_cached_tokens` still has its
  pre-step value there.
- On decode steps, `finished_seqs` are sequences that finished on an *earlier* step and are being
  freed now; `do_save_kvcache` sees them before `finalize_seq` frees their blocks and slot.
- A PROMPT engine never calls `finalize_seq`; prefilled sequences sit in `scheduler.running`
  until `send_processed_prompts` removes them.

## 5. Scheduler (mirrors `csrc/engine/scheduler.hpp`)

Queues: `waiting` (needs prefill), `prompted` (prefilled elsewhere, waiting to start decode),
`running`.

- `_schedule_prefill`: FIFO over `waiting`; allocates blocks + one state slot per sequence
  (`allocate_seq`) unless the block table is already populated; chunks a prompt when it exceeds the
  remaining `max_num_batched_tokens` (if `chunked_prefill`); honors `chunk_checkpoints`; a
  chunked sequence goes back to the front of `waiting`. NORMAL stops after a chunked sequence.
- `_schedule_decode`: frees finished sequences into `finished_seqs`, admits `prompted` into
  `running`, appends a slot for each running sequence, schedules all of them.
- `schedule()` selection: PROMPT → always prefill; COMPLETION → always decode; NORMAL → prefill if
  `waiting` has a schedulable sequence, else decode (simplified from upstream's interval/DP-vote
  policy, §9).
- `process_outputs`: as upstream without MTP. Prompt step: if the prompt is complete,
  `num_cached_tokens = len(prompt)`, append the sampled token, move to `running`; otherwise
  advance `num_cached_tokens` by the chunk and push back to `waiting`. Decode step:
  `num_cached_tokens += 1`, append token. Stop on EOS (unless `ignore_eos`), `max_tokens`, or
  `max_model_len`.

`SchedulerOutputs` fields kept: `scheduled_seqs`, `prompt_run`, `num_batched_tokens`,
`finished_seqs`, `ignored_seqs`, `seqlens_q`, `is_empty()`.

Exposed to Python as upstream: `add_sequence`, `abort_sequence`, `try_allocate_seq`, `free_seq`,
`has_unfinished_seqs`, `get_num_unfinished_seqs`, `waiting`, `running`. Additionally
`add_prompted(seq)` (see ambiguity A4).

## 6. `Sequence` fields (Python view, mirrors `csrc/engine/pybind_sequence.cpp`)

| Field | Meaning |
|---|---|
| `seq_id`, `request_id`, `user_id`, `arrival_time` | identity |
| `data.token_ids`, `get_len()`, `get_prompt_len()` | prompt + generated tokens |
| `status` | `SequenceStatus` (WAITING, RUNNING, FINISHED_STOPPED, FINISHED_LENGTH_CAPPED, FINISHED_ABORTED, FINISHED_IGNORED) |
| `block_size`, `block_table` | paged block KV |
| `num_cached_tokens` | tokens whose KV is computed and present (block KV and SWA ring). Prefill computes `[num_cached_tokens, len)`. Settable; setting it also sets `old_cached_tokens` (upstream `set_num_tokens`) |
| `cached_hit_tokens` | prefix-cache hit length reported for the request. Informational in the engine, as upstream (only set externally) |
| `state_id` | index of the sequence's state slot, `-1` if none |
| `swa_valid_start` | lowest position whose SWA keys are valid; SWA attention ignores keys below it (upstream `prepare_tensor_input_v4.cpp`) |
| `last_copy_state_token_idx` | `num_cached_tokens` at the most recent `do_copy_state` |
| `chunked_rest_len`, `chunk_checkpoints` | chunked-prefill bookkeeping |
| `sampling_params` | as passed to `add_request` |

Dropped: compressed tokens, multimodal, structured output, MTP state, `branch_id`/fork.

## 7. Cache layout (mirrors `deepinfer/worker/cache_engine.py`, block manager, Q&A)

Two layer kinds, configurable pattern (default `["full", "swa", "full", "swa"]`):

- **Block KV** (full-attention layers): `blocks[num_gpu_blocks, n_full_layers, 2, block_size, n_heads, head_dim]`.
  One block holds `block_size` consecutive positions for *all* full-attention layers, so a block is
  the unit of persistence and transfer. Position `p` of a sequence lives in
  `block_table[p // block_size]`, offset `p % block_size`.
- **State slot** (SWA layers): `states[num_state_slots, n_swa_layers, ring, 2, n_heads, head_dim]`,
  one slot per sequence (`state_id`). Position `p` lives at ring index `p % ring`.
  - inference window: `window + backtrack` (default 128 + 6 = 134) positions. This is what
    persists and what the handoff carries.
  - `ring = window + backtrack + store_extra` (default `store_extra = 128`), the upstream
    "kvstore" ring size with `DEEPINFER_PREFILL_STORE_EXTRA_STATE` on
    (`DEEPINFER_PREFILL_STORE_EXTRA_SIZE` defaults to 128). See ambiguity A2 for why this is needed.
- **Replica region** `copy_state`: same shape as `states`. `None` in the base engine (as
  upstream); the serving subclass allocates it.
- **Grain** `KVSTORE_GRAIN`: SWA persists only at positions that are multiples of the grain.
  Upstream hard-codes 128 = block size. mini sets grain = `block_size`.

An SWA *state record at position p* is the inference window ending at `p`: the
`window + backtrack` positions `[p - window - backtrack, p)` in position order (rearranged out of
the ring), with positions `< 0` zero-filled. A query at position `i` attends to SWA keys at
positions `max(i - window + 1, swa_valid_start) … i`.

## 8. The model (mini-only; not a mirror)

A tiny decoder-only transformer, fp32, random weights from a fixed seed, generated on CPU and
copied to the target device so CPU and CUDA share identical weights. RMSNorm, RoPE, MLP, greedy
argmax.

Two deliberate choices make the oracles exact rather than approximately equal:

1. **Token-serial forward.** Sequences run one at a time, and within a sequence, positions run one
   at a time. Every operation for position `i` therefore has the same shapes and inputs whether `i`
   was reached by a single prefill, a chunked prefill, decode, a cache hit, or a handoff. Results
   are bit-identical across all those paths, not merely batch-invariant. Cost: a Python loop per
   token per layer. That is fine at test scale (a few hundred tokens, 4 layers, d_model 64).
2. **K/V are projected from the token-embedding stream, not the residual stream.** Queries come
   from the residual stream as usual. Every cached K/V entry (block or SWA) is then a pure function
   of `(token, position, layer)`. This is what makes the SWA-missing back-off (§10.3) exactly
   recomputable. With residual-stream K/V, recomputing a block from an empty SWA ring cannot be
   exact. Upstream's own replay is approximate (Q&A, "SWA bounded replay").

The engine never *exploits* (2): it always reads K/V from the block table and state slot and never
regenerates them from tokens. Negative tests show the caches are load-bearing: corrupting the
handoff's SWA state or a stored block changes the output.

## 9. Deliberate simplifications

| Upstream | mini | Why |
|---|---|---|
| Batched varlen forward, CUDA graphs, FlashMLA/DeepGEMM/DeepEP | token-serial PyTorch, fp32 | determinism oracle; runs on CPU and SM86 |
| MTP / speculative decoding (`nextn`) | none; `backtrack` exists only as ring/record size | greedy one-token decode is enough to exercise cache positions |
| Zero-overhead scheduling (post-process deferred one step) | always inline | simpler; hook order within a step unchanged |
| NORMAL prefill/decode choice via attempt intervals + DP all-reduce votes | prefill whenever `waiting` is schedulable | single DP rank |
| Sampling (temperature, top-p/k, logprobs, structured output) | greedy only | determinism |
| Forks (`fork_seq`, `clone_states`, `clone_blocks`, `zero_states`) | not mirrored | not needed by milestone 1; the serving layer writes slots directly |
| Preemption / recompute | not mirrored; insufficient blocks at decode → error | tests size the cache |
| Compressed attention (`ca_*`/`ica_*` state), YOCO dropped region, MLA, MoE, FP8 formats | absent; the slot has only the stored SWA region | cache *shape* only |
| Multimodal, compressed tokenizer, tokenizer itself | absent; callers pass token ids | |
| `RequestOutputPool` pointers | plain `RequestOutput` objects | |
| Context/expert/data parallelism, workers | single process, single device | |

## 10. Serving seams built in milestone 1

### 10.1 Handoff (`handoff.py`)

What a PROMPT engine gives a COMPLETION engine. A frozen dataclass with CPU tensors and
`to_bytes()` / `from_bytes()`. Later transports change only how it moves.

| Field | Content |
|---|---|
| `request_id`, `user_id` | identity |
| `sampling_params` | `max_tokens`, `ignore_eos`, `eos_id` |
| `prompt_len` | to split `token_ids` into prompt and output |
| `token_ids` | prompt + the first token sampled by the PROMPT engine |
| `num_cached_tokens` | `= prompt_len` (the first sampled token has no KV yet) |
| `block_kv` | `[ceil(num_cached_tokens / block_size), n_full_layers, 2, block_size, n_heads, head_dim]`, the sequence's blocks in table order |
| `swa_state` | SWA state record at `num_cached_tokens` (inference window, §7) |
| `config_fingerprint` | layer pattern, dims, block/window/backtrack. The receiver rejects a mismatch |

The brief listed token ids, block KV, SWA state and `num_cached_tokens`. Identity, sampling
params and `prompt_len` are added because the COMPLETION engine cannot continue the request
without them.

PROMPT side: `send_processed_prompts()` builds a handoff for every sequence in `running` that
finished prefill, passes it to a sink callable, and frees the sequence.
COMPLETION side: `ServingLLMEngine.receive_handoff(h)` → `add_request(..., enqueue=False)`,
append the output tokens, `scheduler.try_allocate_seq(seq)`, copy blocks into the block table and
the record into the slot, `seq.num_cached_tokens = h.num_cached_tokens`,
`scheduler.add_prompted(seq)`. Note: if the first token already ends the request (EOS or
`max_tokens = 1`), the PROMPT engine's output is final and no handoff is sent.

### 10.2 KV store (`kvstore.py`)

```python
class KVStore(Protocol):
    def put_block(self, key: BlockKey, kv: Tensor) -> None
    def get_block(self, key: BlockKey) -> Tensor | None
    def put_swa(self, key: BlockKey, position: int, state: Tensor) -> None
    def get_swa(self, key: BlockKey, position: int) -> Tensor | None
```

- `block_hashes(token_ids, block_size) -> list[BlockKey]`: a hash chain over full blocks,
  `h_i = sha256(h_{i-1} ‖ int32 bytes of tokens[i*bs:(i+1)*bs])`. A key identifies the whole prefix.
- SWA records are keyed by the chain hash of the block ending at `position`. The position is
  explicit so the store can check that it is grain-aligned.
- `InMemoryKVStore`: two dicts. Both engines in a PD pair share one instance in tests.
- `match(store, token_ids, cfg) -> Hit`: implements the hit rule (§10.3) against a store.

### 10.3 Hit rule

For input length `L` (with `G` = grain = block size, `W` = window, `B` = backtrack):

1. `cap = L - (W + B)`. If `cap < G`, there is no hit.
2. `L1` = the largest multiple of `G` that is `<= cap` such that every block in `[0, L1)` is in the
   store (contiguous prefix).
3. If an SWA record exists at `L1` → hit `L1`, resume at `L1` with that record.
4. Else if a record exists at `L1 - G` → hit `L1`; resume at `L1 - G` with that record, and
   recompute `[L1 - G, L1)` (block KV for those positions is loaded and recomputed identically).
5. Else → hit `L1 - G`; resume at `L1 - G` with a zeroed slot and `swa_valid_start = L1 - G`.
   Blocks `[0, L1)` are loaded.

`Hit` carries `cached_hit_tokens` (the hit length above), `num_cached_tokens` (where compute
resumes), `swa_valid_start`, the blocks, and the record. `ServingLLMEngine.add_request` applies it
through `enqueue=False` → `try_allocate_seq` → load → set fields → `scheduler.add_sequence`.
That uses only upstream-exposed operations.

Step 5 is exact in mini because suffix queries at `>= L1` only see SWA keys at `>= L1 - W + 1 > L1 - G`
(when `W <= G`). The truncated-window outputs at `[L1 - G, L1)` feed only discarded logits and
residual streams that no cached K/V depends on (§8).

### 10.4 Persistence positions (implemented in `ServingLLMEngine` hooks)

- **After prefill** (`pe_prepare_save_kvcache` picks; `pe_do_save_kvcache` writes), only for a
  sequence whose *last* chunk ran this step (`chunked_rest_len == 0`): store every full block not
  already stored, and the SWA record at `P = floor((L - 1) / G) * G` if `P >= G`. Nothing is
  stored for a non-final chunk.
- **During decode** (`should_copy_state`): snapshot when `num_cached_tokens` has crossed a grain
  boundary since `last_copy_state_token_idx` and overshoots it by less than `B` (upstream: "at most
  5 tokens"). Without MTP, decode advances one token per step, so every boundary is snapshotted
  exactly at the boundary.
- **On finish** (`do_save_kvcache`, for `finished_seqs`): store any new full blocks of
  `[0, num_cached_tokens)` and, if a snapshot exists, the record at
  `floor(last_copy_state_token_idx / G) * G` taken from the replica.

## 11. Upstream ambiguities and the reading we chose

- **A1. Hit-rule replay.** The Q&A says that with SWA missing, "the last 128 tokens of the hit
  prefix are recomputed … block KV is reused as-is, neither recomputed nor overwritten", using an
  *approximate* bounded replay. It does not pin down the replay segment (`[L1-128, L1)` vs. the
  block before the reported hit) or how `cached_hit_tokens` vs. `num_cached_tokens` are set.
  Reading: §10.3. `cached_hit_tokens` = reported hit and `num_cached_tokens` = compute start. mini's
  replay is exact by construction (§8); upstream's is not.
- **A2. Record at `P` after prefill.** A `W + B` ring cannot hold the window ending at
  `P = floor((L-1)/G)*G` once prefill has reached `L > P + B`. Upstream's `StateBlock` sizes the slot
  ring with `DEEPINFER_PREFILL_STORE_EXTRA_SIZE` (128) exactly for this: "store only at multiples
  of 128, so prefill may not trigger a store; store an extra segment so multi-turn hits work"
  (`get_kvstore_swa_block_size`). Upstream's default for `DEEPINFER_PREFILL_STORE_EXTRA_STATE` is
  off, and the withheld code may handle this some other way. We adopt the extra ring. This
  corrects the brief: the brief says the slot holds `W + 6`; that is the persisted window, while
  the slot ring is `W + 6 + 128`.
- **A3. SWA record at position 0.** Nothing is stored below `G`, so case 4 of the hit rule with
  `L1 = G` can never find a record at 0. Reading: no record exists at 0; for `L1 = G` with no
  record at `G`, the hit is 0. (The alternative is to treat the empty window at 0 as always
  present. Then `L1 = G` always hits `G`.)
- **A4. How a COMPLETION engine ingests a sequence.** Upstream's `Scheduler` admits PD sequences
  from a `prompted` queue, but no Python binding pushes onto it in the open release (only
  `fork_seq` does, from C++). `try_allocate_seq` and a writable `running` are exposed. We add
  `Scheduler.add_prompted(seq)`. It is the one new scheduler method.
- **A5. Decode snapshot trigger.** `should_copy_state` is a withheld virtual. Reading: §10.4.
  The overshoot bound is `< B`, from the Q&A's "can overshoot by at most 5 tokens, absorbed by the
  6 extra positions".
- **A6. Where SWA records live.** The Q&A says SWA state does not go to the persistent (3FS) store;
  it stays in the producing machine's memory behind an in-memory directory. The brief puts both
  behind one store interface. We keep one `KVStore` protocol with separate block and SWA methods,
  so a later implementation can route them to different backends.
- **A7. `process_output`.** Upstream calls it only from `flush_inflight_reqs`; under non-ZOS the
  withheld online server presumably calls it after each `step()`. mini mirrors the call site and
  leaves per-step invocation to the caller.

## 12. Oracles (the definition of done)

1. **PD**: for a fixed prompt set, NORMAL output equals PROMPT → `Handoff.to_bytes`/`from_bytes` →
   COMPLETION output, token for token. Covers chunked and unchunked prefill and prompts that are
   shorter than one block, exactly one block, and several blocks.
2. **Cache hit**: turn 1 runs and persists through the store. Turn 2 (turn 1 + reply + new tokens)
   with a hit equals a cold turn 2. Cases: record at `L1`; record only at `L1 - G`; no record
   (back-off).
3. **Hit-rule unit tests**: `L - (W+B)` cap, grain alignment, SWA present/absent, prompts shorter
   than one block, non-contiguous block hits.
4. **Persistence positions**: record position after prefill, nothing stored for a non-final chunk,
   decode snapshots exactly at grain crossings, record at finish.
5. **Negative**: corrupting the handoff SWA state or a stored block changes the output.

All oracles run with a small config (e.g. block 16, window 16, backtrack 3, extra 16) and once at
the defaults (128 / 128 / 6 / 128). The PD and cache-hit oracles run on CPU and, under
`@pytest.mark.gpu`, on CUDA.
