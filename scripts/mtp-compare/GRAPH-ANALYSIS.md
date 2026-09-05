# Follow-up: sampling defaults, graph ablation and RTX 3080 compilation

## Sampling-default experiment

The question remains `画一个鹈鹕骑自行车的svg`, with the same 61-token thinking prompt and two 4096-token requests on physical GPUs 2 and 3. Requests omit temperature, top-p, top-k, min-p and all penalty overrides. Seed, output length, streaming and ignore-EOS remain benchmark controls. vLLM uses normal `--generation-config auto`, rather than the previous test's `--generation-config vllm` plus explicit request values.

The server log confirms that the supplied model's `generation_config.json` supplies temperature 1.0, top-k 20 and top-p 0.95. The remaining relevant defaults are min-p 0.0, presence/frequency penalty 0.0 and repetition penalty 1.0. Omitting request overrides therefore does not select a radically different sampling policy here.

Both requests remained in thinking. Mean streaming decode rate was 91.57 token/s and aggregate MTP3 acceptance 42.49%, compared with the previous explicit-parameter run's 94.32 token/s and 43.42%. This does not reproduce a high-acceptance thinking regime merely by omitting sampling parameters. Runs occur sequentially on a shared host and are not statistically sufficient to explain the small throughput difference.

Artifacts: `benches/pelican-default-auto/`, including request bodies in each result JSON and the model-default warning in the server log.

## RTX 3080-specific compilation

The actual existing build configuration was inspected, not inferred from package naming:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_ARCHITECTURES=75;86;89
CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG
GGML_CUDA_GRAPHS=ON
GGML_CUDA_FA=ON
GGML_CUDA_FA_ALL_QUANTS=ON
GGML_NATIVE=OFF
GGML_LTO=OFF
GGML_BACKEND_DL=OFF
```

`build-package-cuda13/build.ninja` also confirms `--generate-code=arch=compute_86,code=[compute_86,sm_86]` and `-use_fast_math`. RTX 3080 is compute capability 8.6, so this package already contains native code for that GPU architecture. Restricting the architecture list to 86 generally reduces build time and package size; it does not by itself give the existing sm_86 kernel another optimization level.

CPU-native compilation and LTO are reasonable controlled experiments, not demonstrated speedups. `GGML_NATIVE` targets the build host's CPU features, not the GPU, and a native build made on the 4090 host may be incompatible with the user's CPU. LTO is guarded by CMake's IPO support check. Neither option substitutes for reducing GPU work or inter-GPU communication. No 3080-specific package was built or performance-tested in this follow-up.

A meaningful 3080 specialization can additionally tune existing q8 attention and quantized matmul launch/tile choices, fuse hot operation sequences and improve graph/input/communication reuse. Those are source/kernel changes requiring correctness and performance testing on the actual 3080 hardware. Do not assume 4090 improvements transfer unchanged, or promise a multiplier from compiler flags. Preserve the user's q8-or-lower KV and full-context constraints.

Reference: NVIDIA's [compute-capability table](https://developer.nvidia.com/cuda/gpus) lists RTX 3080 under 8.6. vLLM's [torch.compile integration](https://docs.vllm.ai/en/stable/design/torch_compile/) describes a different optimization layer from simply compiling this C++/CUDA project in Release mode.

## Separating compilation from CUDA Graphs

Added two explicit-sampling 4096-token runs with `enforce_eager=False`, compilation mode `VLLM_COMPILE`, and `compilation_config.cudagraph_mode=NONE`. The startup log confirms compilation remains enabled and CUDA graph capture sizes are empty. The artifact case stem remains `vllm-graph-mtp` because the diagnostic override is applied after constructing the base command; inspect the saved command/configuration, not that stem, to identify this ablation.

| vLLM execution mode, MTP3 | Mean decode token/s | Aggregate acceptance |
|---|---:|---:|
| Eager, previous run | 21.41 | 44.58% |
| Compiled, CUDA Graphs disabled | 32.29 | 45.50% |
| Compiled, CUDA Graphs enabled, previous run | 94.32 | 43.42% |

Compilation-only is about 1.51x eager, and adding graphs to compiled execution is about 2.92x. The previous 4.41x combined difference must not be presented as either compilation-only or graph-only. These remain two-sample comparisons with independently generated continuations, not exact-workload causal decompositions.

Artifacts: `benches/pelican-compiled-nograph/`.

## Request-scoped CUDA traces

Collected separate Nsight Systems traces for muse TP Graph on, muse TP Graph off, and vLLM compiled/Graph, on the same GPUs. Each trace covers one 512-token request after a 128-token warmup. NVTX ranges in the benchmark driver start and stop collection around the request. CUDA graph node tracing exposes kernel activities; CPU sampling was disabled because the environment does not allow `perf_event_open`. No system profiling permissions were changed.

`profile_summary.py` filters approximately from the first response token to request end, excluding the prompt evaluation. Percentages below are fractions of summed GPU kernel durations across both GPUs, NOT fractions of wall-clock time. CUDA API wait time overlaps GPU work; do not add it to kernel time. Graph-node tracing causes substantial instrumentation overhead: the profiled muse TP-on request took 10.78 seconds versus 8.10 seconds TP-off, reversing the small benefit in the uninstrumented benchmark. Do not use trace timings as deployment throughput or infer a normal-runtime communication regression from that reversal.

For the identical muse output and verification counts:

| Decode-window activity | TP Graph on | TP Graph off |
|---|---:|---:|
| GPU kernel instances, both GPUs | 1,420,865 | 1,420,865 |
| CUDA Graph launch calls | 3,150 | 69,378 |
| `cudaLaunchKernel` calls, excluding extended/driver variants | 81,377 | 162,791 |
| Stream synchronize calls | 93,874 | 93,862 |
| Event synchronize calls | 1,328 | 70,176 |

Thus outer TP capture demonstrably batches submissions and eliminates many host-side event waits, but does not fuse or eliminate the underlying GPU work. TP-off already uses many backend-local CUDA graphs; it is not comparable to vLLM's fully eager baseline. The remaining stream synchronizations include many short calls and some waits for necessary GPU results; their count or summed API time does not establish that all can be removed.

In the muse TP-on trace, matrix-multiplication kernels account for 63.05% of summed GPU kernel time, internal collective kernels 17.56%, flash-attention kernels 4.02%, and the named Gated DeltaNet state-update kernel 1.03%. The remainder includes normalization, activation quantization, copies and other operations. Gated DeltaNet layers also contain projections/matmuls included in the matrix category: 1.03% must not be described as the cost of all Gated DeltaNet layers. Collective kernel duration includes waiting for the other rank and is sensitive to profiler-induced launch skew.

The vLLM trace contains 641,760 decode-window kernels, but also has fewer verification rounds (217 versus muse's 258), different weights/KV and a different continuation. Normalizing roughly by verification rounds yields about 2,957 versus 5,507 kernels per round across both GPUs. This is evidence of different execution granularity, not a pure fusion speedup measurement. Actual names include Marlin INT4 matmul and fused Triton normalization/residual kernels. Muse's hot path instead includes Q5_K/Q6_K `mul_mat_vec_q`, separate `quantize_q8_1`, normalizations and copy kernels. Changing the graph submission boundary alone cannot remove those operations or turn Q5/Q6 kernels into Marlin INT4 kernels.

## Scoped optimization priorities

1. **Microbenchmark the actual Q5/Q6 matrix shapes at batches 1, 2, 3 and 4 on 3080.** Tune MMVQ tile/warp/vectorization choices and compare alternatives with numerical checks. Dispatch in `ggml-cuda.cu` selects MMVQ before checking MMQ; blindly enabling `GGML_CUDA_FORCE_MMQ` is not guaranteed to switch this hot path. Do not promise a generic flag-based improvement.
2. **Fuse measured adjacent operations while retaining q8 KV.** Candidate sequences include normalization/activation quantization and compatible projection/activation operations. Check register pressure, memory traffic and MTP logits; not every combination benefits from fusion on 3080. This is source/kernel work, not ordinary Release compilation.
3. **Fix bounded graph-cache behavior and trace key churn.** The previous default 16-entry cache disables TP capture on overflow. A safe cache policy and stable shape/storage representation are more robust than an arbitrarily large cap. Never drop addresses or operation parameters from the key without proving replay safety. This matters for sustained operation and long contexts, even though the short-request speed benefit of the larger cap was modest.
4. **Inspect necessary versus avoidable synchronization and deferred gather.** Internal AllReduce uses mapped pinned-host exchange on this non-P2P pair. Its gather is explicitly not graph-compatible in `ggml-cuda.cu`, so the meta backend defers it and executes the remaining tail outside the outer graph. Profile critical-path waits with lower-overhead tracing before choosing a communication rewrite; the current trace does not prove that gather alone dominates.
5. **Treat native CPU/LTO builds as secondary A/B tests.** They cannot remove GPU matrix work or PCIe dependencies and must be built for the user's CPU/toolchain. No existing compiler option shown here promises a vLLM-sized multiplier for muse.

These priorities address short-context decoding. They do not establish the bottleneck at 80K/262K, where KV attention work can change substantially. No production source changes, 3080 runtime validation, or speedup implementation was performed in this follow-up.

Trace artifacts are `benches/pelican-profile-{graph,tpoff,vllm}.nsys-rep`, exported SQLite databases and `.summary.json` files. Their matching `*-data/` directories contain request results and server logs. Reproduction example:

```bash
PELICAN_NVTX=1 PELICAN_GRAPH_CACHE=256 PELICAN_VERBOSE=4 \
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
nsys profile --trace=cuda,nvtx --capture-range=nvtx \
  --nvtx-capture=muse-graph-mtp-run0 --capture-range-end=stop \
  --sample=none --cpuctxsw=none --cuda-graph-trace=node --wait=primary \
  --output=/workspace/muse.cpp/benches/pelican-profile-graph \
  /workspace/vllmvenv/.venv/bin/python scripts/mtp-compare/pelican_bench.py \
  --out benches/pelican-profile-graph-data --cases muse-graph-mtp --runs 1 --tokens 512
```

Use new output paths for a new run. The benchmark stops only the server process groups it starts, and switches only after replacement commands are ready.
