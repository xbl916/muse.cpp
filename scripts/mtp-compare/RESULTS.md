# MTP first-step comparison, 2026-09-05

Completed a fixed-prefix comparison using the supplied vLLM 0.28.0 environment and AWQ checkpoint against the local muse.cpp GGUF. This test found a modest first-step proposal-quality difference, not a severe first-step alignment failure. The supplied vLLM configuration also produced a 45.06% aggregate three-token acceptance rate on this reasoning sample.

## Setup

- vLLM: `/workspace/vllmvenv/.venv`, model `/pubdata/llm_models/cyankiwi/Qwen3.8-27B-AWQ-INT4`, GPUs 1 and 3, TP=2, eager, BF16 KV.
- muse: `/workspace/llm/Qwen3.8-27b/Qwen3.8-27B-Q5_K_M.gguf`, GPUs 2 and 6, tensor split 1:1, paged q8_0 KV, outer TP graphs disabled. Used existing `build-package-cuda13` libraries and the standalone capture utility.
- Both: context limit 4096, one sequence. The vLLM source trace uses greedy three-token MTP proposals and target temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, seed 1007843.
- Prompt: prove infinitely many primes congruent to 3 modulo 4 and discuss the analogous argument for 1 modulo 4. The prompt has 89 tokens; the captured output has 256 tokens and is still reasoning text.
- 110 proposal records were captured. The final record samples beyond the requested output limit and is excluded from comparisons. All remaining 109 prefixes match the final accepted sequence exactly.
- Each muse replay teacher-forces the same complete prefix and the same anchor token. Target reference logits come from a subsequent verification at that exact anchor, not an earlier rejected speculative token at the same position.

## Results over all 109 matched positions

| Metric | Main model | First MTP proposal |
|---|---:|---:|
| Cross-backend top-1 agreement | 94.50% | 91.74% |
| Mean top-20 overlap | 92.84% | 91.93% |
| Mean Jensen-Shannon divergence, natural log | 0.00534 | 0.00834 |
| Mean normalized hidden-state cosine | 0.99072 | 0.99139 |

The main-model hidden cosine refers to the row supplied to MTP before the anchor. The main-model logits comparison refers to the prediction after the anchor. These are intentionally different temporal positions.

| First-proposal quality against its own target | vLLM | muse |
|---|---:|---:|
| Draft argmax equals target argmax | 79.82% | 77.98% |
| Mean expected acceptance under target top-k/top-p sampling | 70.79% | 67.36% |

The expected acceptance is the target probability assigned to the greedy draft token. It is not the observed aggregate acceptance of a three-token chain. The measured gap is 3.44 percentage points on this trace.

For 108 fully observed vLLM verification rounds, 146 of 324 draft tokens were accepted: **45.06% aggregate acceptance**. The unconditional per-position rates are 63.89%, 42.59%, and 28.70%. Counts are reconstructed from the advance between successive accepted anchor positions; the final output-truncated round is excluded. This demonstrates that low aggregate acceptance can occur in vLLM at these settings on reasoning text. It does not establish a typical rate across tasks or invalidate observations from other prompts.

Replayed all 109 positions once more with muse device hidden staging enabled. Main and MTP logits were bit-identical to the host-staging replay: maximum absolute difference **0.0** in both cases. This covers first-proposal fixed-prefix replay, not multi-step rollback or live multi-request state reuse.

## Weight and vocabulary checks

The 248,077 defined tokenizer entries have matching IDs. GGUF has 243 additional padding entries up to the model vocabulary size 248,320. The MTP hidden norm, embedding norm, and head norm match exactly after accounting for Qwen's `weight + 1` convention.

The AWQ configuration excludes MTP projection/FFN weights from INT4 quantization. The GGUF uses Q5/Q6 for MTP attention/FFN and Q8 for its input projection. Main-model quantization, KV precision and arithmetic precision also differ. This experiment therefore cannot attribute the small remaining difference solely to quantization or solely to an implementation bug.

## Artifacts and interpretation

- Raw vLLM tensors and token prefixes: `benches/mtp-compare-vllm/`.
- Raw muse host-staging tensors and detailed comparison: `benches/mtp-compare-muse-all/comparison.json`.
- Raw muse device-staging logits: `benches/mtp-compare-muse-device/`.
- Reproduction: `scripts/mtp-compare/README.md`.

The earlier 24-position subset showed a larger 10.32-point first-proposal gap. The full 109-position result supersedes that subset; it illustrates why the subset should not be treated as a stable estimate.

No throughput conclusion can be drawn from these instrumented runs. No 80K/262K context test or 3080 runtime test was performed here. The next discriminating test for a remaining production gap is second/third proposal logits and cache rollback on an identical accepted token trace, with separate thinking/answer counts. These results do not justify changing recommended sampling parameters or claiming that MTP has been fixed.
