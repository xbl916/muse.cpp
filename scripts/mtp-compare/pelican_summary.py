"""Summarize raw HTTP measurements without conflating graph-off and eager."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*-run*.json")):
        data = json.loads(path.read_text())
        delta = data["metric_delta"]
        def total(name):
            return sum(v for k, v in delta.items() if k.split("{")[0] == name)
        if path.name.startswith("vllm"):
            accepted = total("vllm:spec_decode_num_accepted_tokens_total")
            drafted = total("vllm:spec_decode_num_draft_tokens_total")
        else:
            accepted = total("llamacpp:spec_decode_num_accepted_tokens_total")
            drafted = total("llamacpp:spec_decode_num_draft_tokens_total")
        rows.append(dict(case=path.name.split("-run")[0], run=path.stem.split("-run")[1],
                         tokens=data["usage"].get("completion_tokens"),
                         ttft_ms=1000 * data["ttft_seconds"],
                         decode_tps=data["stream_decode_tps"], e2e_tps=data["end_to_end_tps"],
                         accepted=accepted, drafted=drafted,
                         acceptance=accepted / drafted if drafted else None,
                         thinking_end_character=data["thinking_end_character"]))
    summary = []
    for case in dict.fromkeys(r["case"] for r in rows):
        rs = [r for r in rows if r["case"] == case]
        accepted, drafted = sum(r["accepted"] for r in rs), sum(r["drafted"] for r in rs)
        summary.append(dict(case=case, runs=len(rs),
                            decode_tps_mean=statistics.mean(r["decode_tps"] for r in rs),
                            e2e_tps_mean=statistics.mean(r["e2e_tps"] for r in rs),
                            ttft_ms_mean=statistics.mean(r["ttft_ms"] for r in rs),
                            acceptance=accepted / drafted if drafted else None))
    result = dict(summary=summary, runs=rows)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
