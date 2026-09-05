"""Sequential, uninstrumented two-GPU HTTP benchmark on identical prompt IDs."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("MTP_COMPARE_MODEL", "/pubdata/llm_models/cyankiwi/Qwen3.8-27B-AWQ-INT4")
GGUF = os.environ.get("MTP_COMPARE_GGUF", "/workspace/llm/Qwen3.8-27b/Qwen3.8-27B-Q5_K_M.gguf")
PYTHON = os.environ.get("MTP_COMPARE_PYTHON", sys.executable)
PROMPT = "画一个鹈鹕骑自行车的svg"


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode()


def metrics(url):
    result = {}
    for line in get(url + "/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(" ", 1)
        if "spec_decode" in key or "draft" in key:
            result[key] = float(value)
    return result


def request(url, tokens, count, seed, out):
    body = dict(model="pelican", prompt=tokens, max_tokens=count,
                temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, frequency_penalty=0.0,
                repetition_penalty=1.0, repeat_penalty=1.0,
                seed=seed, ignore_eos=True, stream=True,
                stream_options={"include_usage": True}, cache_prompt=False)
    if os.environ.get("PELICAN_OMIT_SAMPLING") == "1":
        for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                    "frequency_penalty", "repetition_penalty", "repeat_penalty"):
            body.pop(key)
    req = urllib.request.Request(url + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    before = metrics(url)
    if os.environ.get("PELICAN_NVTX") == "1":
        from torch.cuda import nvtx
        nvtx.range_push(out.stem)
    start = time.perf_counter()
    events = []
    with urllib.request.urlopen(req, timeout=1200) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            events.append({"seconds": time.perf_counter() - start,
                           "data": json.loads(line[6:])})
    elapsed = time.perf_counter() - start
    if os.environ.get("PELICAN_NVTX") == "1":
        nvtx.range_pop()
    # Engine metrics are published asynchronously, outside the measured interval.
    time.sleep(6)
    after = metrics(url)
    chunks = [e for e in events if any(c.get("text") for c in e["data"].get("choices", []))]
    usage = next((e["data"]["usage"] for e in reversed(events) if e["data"].get("usage")), {})
    text = "".join(c.get("text", "") for e in events for c in e["data"].get("choices", []))
    n = usage.get("completion_tokens")
    first = chunks[0]["seconds"] if chunks else None
    last = chunks[-1]["seconds"] if chunks else None
    result = dict(seed=seed, requested_tokens=count, usage=usage, request_body=body,
                  elapsed_seconds=elapsed, ttft_seconds=first,
                  end_to_end_tps=n / elapsed if n else None,
                  stream_decode_tps=(n - 1) / (last - first) if n and last > first else None,
                  thinking_end_character=text.find("</think>"),
                  metric_delta={k: v - before.get(k, 0) for k, v in after.items()},
                  text=text, events=events)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(out.name, json.dumps({k: v for k, v in result.items() if k not in ("text", "events")}), flush=True)


def command(engine, graph, mtp, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("MTP_COMPARE_GPUS", "2,3")
    env["OMP_NUM_THREADS"] = "1" if engine == "vllm" else "16"
    for key in ("GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_TP_GRAPHS", "GGML_CUDA_ALLREDUCE"):
        env.pop(key, None)
    if engine == "vllm":
        cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
               "--served-model-name", "pelican", "--tensor-parallel-size", "2",
               "--max-model-len", "8192", "--max-num-seqs", "1",
               "--max-num-batched-tokens", "512", "--gpu-memory-utilization", "0.65",
               "--no-enable-prefix-caching", "--limit-mm-per-prompt", '{"image":0,"video":0}',
               "--generation-config", os.environ.get("PELICAN_GENERATION_CONFIG", "vllm"), "--seed", "1007843",
               "--host", "127.0.0.1", "--port", str(port)]
        if not graph:
            cmd.append("--enforce-eager")
        if os.environ.get("PELICAN_COMPILE_NO_GRAPH") == "1":
            cmd += ["--compilation-config", '{"cudagraph_mode":"NONE"}']
        if mtp:
            cmd += ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}']
    else:
        env.update(GGML_CUDA_ALLREDUCE="internal", GGML_CUDA_TP_GRAPHS=str(int(graph)),
                   GGML_CUDA_TP_GRAPH_MAX_BATCH="8", LLAMA_META_MIRROR_OUTPUT="1")
        if os.environ.get("PELICAN_GRAPH_CACHE"):
            env["GGML_CUDA_TP_GRAPH_MAX_GRAPHS"] = os.environ["PELICAN_GRAPH_CACHE"]
        cmd = [os.environ.get("MTP_COMPARE_MUSE_BIN", str(ROOT / "build-package-cuda13/bin/llama-server")), "-m", GGUF,
               "--alias", "pelican", "--scheduler", "paged", "--max-model-len", "8192",
               "--max-num-seqs", "1", "-np", "1", "--gpu-memory-utilization", "0.65",
               "--kv-block-size", "32", "--no-kv-prefix-cache", "--cache-ram", "0",
               "--no-cache-idle-slots", "-ctk", "q8_0", "-ctv", "q8_0",
               "-sm", "tensor", "-ts", "1,1", "-fa", "on", "-ngl", "99",
               "--fit", "off", "--batch-size", "512", "--ubatch-size", "512",
               "--paged-prefill-chunk", "512", "--paged-decode-steps", "1",
               "--backend-sampling", "--no-spec-adaptive", "--reasoning-budget", "-1",
               "--spec-type", "draft-mtp" if mtp else "none", "--metrics",
               "--host", "127.0.0.1", "--port", str(port), "--no-webui",
               "--log-timestamps", "--log-prefix", "-t", "16", "-tb", "16"]
        if mtp:
            cmd += ["--spec-draft-n-max", "3", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
                    "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0",
                    "--no-spec-draft-probabilistic", "--spec-draft-backend-sampling"]
        if os.environ.get("PELICAN_VERBOSE"):
            cmd += ["-lv", os.environ["PELICAN_VERBOSE"]]
    return cmd, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="vllm-graph-mtp,vllm-eager-mtp,vllm-graph-base,muse-graph-mtp,muse-tpoff-mtp,muse-graph-base")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    port = int(os.environ.get("MTP_COMPARE_PORT", "18991"))
    url = f"http://127.0.0.1:{port}"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except ConnectionRefusedError:
        pass
    else:
        raise RuntimeError(f"Port {port} is already in use; choose another MTP_COMPARE_PORT")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tokens = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=True,
                                    add_generation_prompt=True, enable_thinking=True, return_dict=False)
    (out / "prompt.json").write_text(json.dumps(dict(question=PROMPT, tokens=tokens,
                                                     rendered=tok.decode(tokens)), ensure_ascii=False, indent=2))
    for case in args.cases.split(","):
        engine, mode, kind = case.split("-")
        cmd, env = command(engine, mode == "graph", kind == "mtp", port)
        (out / (case + "-command.json")).write_text(json.dumps(dict(command=cmd,
            environment={k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_", "CUDA_VISIBLE", "OMP_"))}), indent=2))
        print("START", case, flush=True)
        with (out / (case + ".log")).open("w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=ROOT, start_new_session=True)
            try:
                deadline = time.monotonic() + 1500
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{case} exited {proc.returncode}; see log")
                    try:
                        get(url + "/health")
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise TimeoutError(case)
                        time.sleep(2)
                request(url, tokens, 128, 1007843, out / (case + "-warmup.json"))
                for run in range(args.runs):
                    request(url, tokens, args.tokens, 1007843 + run,
                            out / f"{case}-run{run}.json")
            finally:
                # Only the process group created by this benchmark is stopped.
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        print("DONE", case, flush=True)


if __name__ == "__main__":
    main()
