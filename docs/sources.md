# Sources

Every external source the docs rely on. Each entry says what we use it for and how far to trust it.

## Evidence labels

The docs separate four kinds of statement. Background documents tag claims inline with these labels:

| Label | Meaning |
|---|---|
| **[code]** | Checked in DeepInfer source at the pinned commit `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`. |
| **[DS]** | DeepSeek's own statement: README, Q&A, Zhihu post, or DeepSeek-authored paper. No one has verified it independently. Production numbers are always [DS]. |
| **[3P]** | Third-party reporting. |
| **[ours]** | Our design decision, inference, or arithmetic. |

A [code] claim means the code exists and has the described shape at the pin. It does not mean we have run it: the code needs SM100 or Ascend 950 hardware, and none of it has been run for this repo.

Links into the DeepInfer repo below use the pinned commit, so they do not drift:
`https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/<path>`.

## Primary: DeepInfer (DeepSeek, Apache-2.0)

| Source | Used for | Reliability |
|---|---|---|
| [DeepInfer repo at the pin](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01) | Engine interface: `LLMEngine` hooks, `EngineType`, scheduler, `Sequence`, cache layout, EPLB, kernel submodules. The source of truth for every [code] claim. | High for what the code does. Comments are mostly in Chinese. |
| [`README.md`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md) | Feature list, hardware and toolchain requirements, the list of what the release omits, environment variables for bounded replay and determinism. | [DS]. Requirements and env vars are checkable; feature and performance claims are not. |
| [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md) ("the Q&A") | Production architecture: parallelism, storage, the hit rule, SWA persistence, the four load balancers, autoscaling, fault tolerance. | [DS]. The most detailed and most recent primary description. DeepSeek says it will keep updating it, so anything not at the pin may change. Contains one internal inconsistency (24-hour storage sizing vs. 72-hour retention; see the background doc). |
| [`doc/images/architecture_overview.png`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/images/architecture_overview.png) | The E/P/D request-flow diagram. | [DS]. A schematic. |
| `doc/images/{prefill,decode,wide_ep_decode}_pareto.png` | Offline benchmark curves. Not reproduced here. | [DS]. Offline benchmark, not production. |

## Primary: other DeepSeek publications

| Source | Used for | Reliability |
|---|---|---|
| Zhihu post "DeepSeek V4.1 Flash 大规模推理部署实践" ([link](https://zhuanlan.zhihu.com/p/2095560295930508953); readable via [r.jina.ai](https://r.jina.ai/https://zhuanlan.zhihu.com/p/2095560295930508953)) | The Chinese original of most of the Q&A. It is the only source for the production statistics table (an image), the 3FS bandwidth figures, the ~112 KiB block size, the 180 s SWA-cache lifetime, and the V4 → V4.1 history of SWA persistence. | [DS]. Two of its figures differ from the Q&A (EPLB imbalance, DSpark gain), and its reference numbering is off by one in places (the 3FS reference points at the V4.1 report and vice versa). Where the two differ, the docs give both. |
| [DeepSeek-V4.1-Flash tech report, arXiv 2609.19969](https://arxiv.org/abs/2609.19969) | Model facts: CED (causal encoder–decoder) architecture, 890 B/token KV in HBM, persistent KV about 1/8 of V4-Flash's. | [DS]. We read only the abstract for these docs. |
| [DualPath, arXiv 2602.21548](https://arxiv.org/abs/2602.21548) | Storage-bandwidth analysis for agentic prefill, dual-path KV loading, traffic isolation, and the inter- and intra-group schedulers the Q&A points to for the decode LB. | [DS]-authored (DeepSeek-AI with PKU and Tsinghua). Peer review: the Zhihu post cites it as SIGCOMM '26 under the title "DualPath: Accelerating Agentic LLM Inference by Harvesting Disaggregated KV-Cache Storage I/O". arXiv v2 (26 Feb 2026) is titled "DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference". We read arXiv v2. Its experiments use V3.2-era models on Hopper, not V4.1. |
| [DSpark, arXiv 2607.05147](https://arxiv.org/abs/2607.05147) | Speculative decoding with confidence-scheduled verification. Background only. | [DS]. |
| [DeepSeek-V3/R1 inference system overview (Feb 2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | Earlier generation for comparison: EP32 prefill, EP144 decode, 226.75 average H800 nodes, 545% theoretical margin. | [DS]. |
| [3FS](https://github.com/deepseek-ai/3FS) | The distributed file system behind the block-KV tier. | [DS]. Open source (MIT). We have not deployed it. |
| [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | API ↔ token conversion. Rust libraries with Python bindings, MIT. Mentioned only. | [DS]. |
| Kernel repos: [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP), [DeepSelect](https://github.com/deepseek-ai/DeepSelect) | Hardware requirements: FlashMLA SM100/SM103 only since its 2026-09-30 release; DeepGEMM SM90 or SM100; DeepEP Hopper or newer. | [DS]. Read from the repos' current READMEs (2026-10-20), not from the submodule commits DeepInfer pins. |

## Third-party reporting

| Source | Used for | Reliability |
|---|---|---|
| [The New Stack](https://thenewstack.io/deepseek-deepinfer-huawei-nvidia/) | Release context: NVIDIA and Huawei both supported; the release is the engine without the serving layer. | [3P]. Restates DeepSeek's material; no independent measurement. |
| [Crypto Briefing](https://cryptobriefing.com/deepseek-open-sources-deepinfer-inference-framework/) | Release context. | [3P]. Restates DeepSeek's material. |
| Quartz, MarkTechPost | Named in our research notes as covering the release. | [3P]. We did not locate the specific articles and cite nothing from them. |

None of the docs rely on third-party reporting for a technical claim. No third party has published independent measurements of DeepSeek's production numbers.

## Internal

| Source | Used for |
|---|---|
| [`reference/DEEPINFER_PIN.md`](../reference/DEEPINFER_PIN.md) | The pinned commit, how to fetch it, and which upstream files mini-DeepInfer mirrors. |
| [`docs/interface-contract.md`](interface-contract.md) | The mini-DeepInfer interface, the handoff and store seams, the hit rule as implemented, and every deviation from upstream. |
