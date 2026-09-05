# Fixed-prefix MTP comparison

For continuation on the user's dual RTX 3080 machine, start with [the Chinese handoff](../../docs/3080-handoff-2026-09-05.md). Large raw artifacts referenced by these reports remain on the original machine and are not included in Git.

These diagnostic tools capture vLLM's first MTP proposal and replay exactly the same accepted token prefix and anchor token through muse.cpp. They do not change installed vLLM files or add a server endpoint. Capture adds CPU transfers and extra LM-head evaluations, so its throughput is not a benchmark.

The vLLM script targets the locally installed v0.28.0 API. It uses two GPUs, eager execution, greedy MTP proposals of length three, and target sampling temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 0.0 and repetition penalty 1.0. The current test prompt is `画一个鹈鹕骑自行车的svg`; historical prime-proof artifacts retain their original prompt. Model path and test prompt are in the script. Use free GPUs; do not terminate an existing service just to prepare these tools.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=1,3 MTP_COMPARE_OUT="$PWD/benches/mtp-compare-vllm" \
  /workspace/vllmvenv/.venv/bin/python scripts/mtp-compare/vllm_capture.py

/workspace/vllmvenv/.venv/bin/python scripts/mtp-compare/compare.py \
  benches/mtp-compare-vllm --limit 0

g++ -std=c++17 -O2 -Icommon -Iinclude -Iggml/include -Isrc -Ivendor/nlohmann \
  scripts/mtp-compare/muse_capture.cpp -Lbuild-package-cuda13/bin \
  -Wl,-rpath,"$PWD/build-package-cuda13/bin" -lllama-common \
  build-package-cuda13/common/libllama-common-base.a -lllama -lggml -lggml-base \
  -o /tmp/muse-mtp-capture

CUDA_VISIBLE_DEVICES=2,6 LLAMA_MTP_DEVICE_HIDDEN=0 \
  GGML_CUDA_ALLREDUCE=internal GGML_CUDA_TP_GRAPHS=0 \
  /tmp/muse-mtp-capture /workspace/llm/Qwen3.8-27b/Qwen3.8-27B-Q5_K_M.gguf \
  benches/mtp-compare-vllm/prefixes.json benches/mtp-compare-muse

/workspace/vllmvenv/.venv/bin/python scripts/mtp-compare/compare.py \
  benches/mtp-compare-vllm --muse benches/mtp-compare-muse --limit 0
```

Build the repository libraries first if headers or code have changed. The standalone program must match those libraries. It uses the existing MTP driver, TP mode, paged KV and q8_0 KV. Host hidden transfer is selected to expose actual hidden tensors for comparison; this run alone does not validate device hidden staging.

For a device-staging cross-check, repeat the replay with `LLAMA_MTP_DEVICE_HIDDEN=1` and a different output directory, then supply that directory as `--device` alongside the host result passed to `--muse`. Device mode exports logits only. `--limit 0` includes every valid position; the default selects up to 24.

`step-*.pt` contains raw full-vocabulary logits, normalized hidden states, target positions, target input IDs and the accepted prefix including the anchor. `compare.py` checks every prefix against the final generated token sequence, then finds the target verification row after the identical anchor. The muse outputs are float32 binary vectors. `comparison.json` includes top-20 IDs and probabilities, top-1 agreement, top-20 overlap, Jensen-Shannon divergence, hidden cosine and expected first-proposal acceptance after the specified target sampling filters.

Different checkpoints, weight quantizations, KV types and arithmetic precision are confounders. Distribution mismatch alone does not establish a kernel or temporal-alignment bug. A short capture also does not measure long-context behavior or overall multi-step acceptance. Preserve the raw tensors for additional checks.

## Uninstrumented pelican speed comparison

`pelican_bench.py` runs vLLM and muse sequentially on GPUs 2 and 3 using localhost port 18991. Check that these resources are free first. It never stops unrelated services. All launch commands are prepared before switching servers. It stops only process groups it created, including on an exception.

These are defaults for the original host. Override `MTP_COMPARE_GPUS`, `MTP_COMPARE_PORT`, `MTP_COMPARE_MODEL` (HF checkpoint/tokenizer), `MTP_COMPARE_GGUF`, and `MTP_COMPARE_MUSE_BIN` on another machine. The vLLM subprocess uses the invoking Python unless `MTP_COMPARE_PYTHON` is set. `vllm_capture.py` also supports `MTP_COMPARE_MODEL`; select its GPUs through `CUDA_VISIBLE_DEVICES`. The benchmark refuses to start if its chosen port is already accepting connections. It still requires a readable HF tokenizer even for muse-only cases.

```bash
/workspace/vllmvenv/.venv/bin/python scripts/mtp-compare/pelican_bench.py \
  --out benches/pelican-speed
/workspace/vllmvenv/.venv/bin/python scripts/mtp-compare/pelican_summary.py \
  benches/pelican-speed
```

Both engines receive identical HF-template token IDs, with thinking enabled and the template's default xhigh instruction. Raw completion requests bypass server-specific chat templating and reasoning-budget changes. Sampling is explicit: temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence/frequency penalties 0.0, repetition penalty 1.0. Prefix caching is disabled, concurrency is one, context capacity is 8192, and prefill batch cap is 512. Each case gets a 128-token warmup and two fixed-length 4096-token runs with seeds 1007843 and 1007844. Ignore-EOS keeps output length fixed; continuations can differ across engines and configurations. These are not fixed-output or long-context benchmarks.

The default matrix is vLLM compiled/CUDA-graph MTP3, vLLM eager MTP3, vLLM compiled/CUDA-graph without MTP, muse TP-graph MTP3, muse outer-TP-graph-off MTP3, and muse TP-graph without MTP. Muse outer-TP-graph-off does not disable backend-local CUDA graphs, and vLLM enforce-eager also disables compilation; these switches are not identical ablations. MTP uses greedy drafts, without adaptive disablement or probability truncation. Muse uses internal AllReduce, mirrored output and TP graph maximum batch 8.

Raw JSON includes every SSE event, usage, output text, client TTFT, end-to-end rate, approximate streaming decode rate and per-request changes in speculative metrics. The streaming rate is `(completion_tokens - 1) / (last_text_event - first_text_event)`; speculative output chunks may contain multiple tokens, so it is an approximation. Metric publication waits occur after timing. Server logs provide graph-capture evidence and native timing/acceptance diagnostics. Startup, compilation, capture warmup and metrics-publication waits are excluded from measured generation time. Weight formats and KV formats differ between the two engines; the comparison measures these actual deployments, not isolated graph/kernel efficiency.

The default 16-entry muse TP graph cache fell back during the measured requests. Use `PELICAN_GRAPH_CACHE=256 PELICAN_VERBOSE=4` with a separate output directory to reproduce the larger-cache diagnostic; this is not a long-context deployment recommendation. Full results and limits are in [PELICAN-RESULTS.md](PELICAN-RESULTS.md).
