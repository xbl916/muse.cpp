"""Summarize a request-scoped Nsight trace; GPU sums are not wall-time shares."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    data = json.loads(args.request.read_text())
    con = sqlite3.connect(args.database)
    strings = dict(con.execute("SELECT id,value FROM StringIds"))
    ranges = con.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
    start, end = next((s, e) for s, e, text, tid in ranges
                      if (text or strings.get(tid)) == args.request.stem)
    # The NVTX push precedes the client timer by a few Python instructions.
    lo = start + round(data["ttft_seconds"] * 1e9)
    hi = end
    where = " WHERE start >= ? AND end <= ?"
    rows = con.execute("SELECT demangledName,count(*),sum(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
                       + where + " GROUP BY demangledName ORDER BY sum(end-start) DESC", (lo, hi)).fetchall()
    total = sum(r[2] for r in rows)
    groups = defaultdict(lambda: [0, 0])
    kernels = []
    for sid, count, ns in rows:
        name = strings[sid]
        if "ggml_cuda_ar_" in name or "nccl" in name.lower():
            kind = "collective kernels (includes waiting)"
        elif any(s in name.lower() for s in ("mul_mat", "gemm", "gemv", "marlin")):
            kind = "matrix multiplication kernels"
        elif "flash_attn" in name or "flash_fwd" in name:
            kind = "flash attention kernels"
        elif "gated_delta_net_cuda" in name:
            kind = "Gated DeltaNet core kernel only"
        else:
            kind = "other kernels"
        groups[kind][0] += count
        groups[kind][1] += ns
        kernels.append(dict(name=name, count=count, summed_ms=ns / 1e6, gpu_sum_percent=100 * ns / total))
    apis = []
    for sid, count, ns in con.execute("SELECT nameId,count(*),sum(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
                                     + where + " GROUP BY nameId ORDER BY sum(end-start) DESC", (lo, hi)):
        apis.append(dict(name=strings[sid], count=count, summed_ms=ns / 1e6))
    result = dict(note="Node-level profiling perturbs execution. Kernel sums span both GPUs and are not wall-time percentages. API waits overlap GPU work.",
                  interval="approximately first-token to request end", interval_seconds=(hi - lo) / 1e9,
                  output_tokens=data["usage"]["completion_tokens"], kernel_count=sum(r[1] for r in rows),
                  summed_kernel_ms=total / 1e6,
                  groups={k: dict(count=v[0], summed_ms=v[1] / 1e6, gpu_sum_percent=100 * v[1] / total)
                          for k, v in groups.items()},
                  kernels=kernels, cuda_runtime_apis=apis)
    output = args.database.with_suffix(".summary.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, indent=2))


if __name__ == "__main__":
    main()
