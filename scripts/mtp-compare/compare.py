"""Prepare fixed prefixes, then compare raw model tensors at matching positions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def distribution_metrics(a, b):
    a = torch.as_tensor(a).float().flatten()
    b = torch.as_tensor(b).float().flatten()
    assert a.shape == b.shape
    p, q = a.softmax(-1), b.softmax(-1)
    m = (p + q) / 2
    js = ((p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum()
          + (q * (q.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum()) / 2
    ia, ib = a.topk(20).indices, b.topk(20).indices
    return {"top1_match": bool(ia[0] == ib[0]),
            "top20_overlap": len(set(ia.tolist()) & set(ib.tolist())) / 20,
            "js": float(js), "vllm_top20": ia.tolist(), "muse_top20": ib.tolist(),
            "vllm_top20_probs": p[ia].tolist(), "muse_top20_probs": q[ib].tolist()}


def proposal_quality(target, draft):
    target = torch.as_tensor(target).flatten().float()
    draft = torch.as_tensor(draft).flatten().float()
    values, indices = target.topk(20)
    p = values.softmax(-1)
    keep = (p.cumsum(-1) - p) < 0.95
    p = p * keep
    p = p / p.sum()
    token = int(draft.argmax())
    return {"argmax_match": token == int(target.argmax()),
            "expected_acceptance": float(p[indices == token].sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--muse", type=Path)
    parser.add_argument("--device", type=Path, help="Optional device-hidden replay for exact logits comparison")
    parser.add_argument("--limit", type=int, default=24, help="0 selects every valid capture")
    args = parser.parse_args()
    torch.set_num_threads(4)
    captures = [(p.stem, torch.load(p, weights_only=False)) for p in sorted(args.capture.glob("step-*.pt"))]
    assert captures, "No captures found"
    generation = json.loads((args.capture / "generation.json").read_text())
    prompt = json.loads((args.capture / "prompt.json").read_text())
    sequence = prompt + generation["tokens"]
    valid = []
    for capture_index, (name, record) in enumerate(captures):
        if len(record["prefix"]) > len(sequence) and record["prefix"][:len(sequence)] == sequence:
            continue  # Final verification may sample past the output-token limit.
        if record["prefix"] != sequence[:len(record["prefix"])]:
            raise RuntimeError(f"Prefix does not match final accepted sequence: {name}")
        # Find the actual target row after this anchor in a later verification.
        pos = record["anchor_position"]
        for _, nxt in captures[capture_index + 1:]:
            if pos in nxt["target_positions"] and nxt["target_logits"] is not None:
                idx = nxt["target_positions"].index(pos)
                if nxt["target_ids"][idx] != record["prefix"][-1]:
                    continue
                logits = nxt["target_logits"]
                if len(logits) == len(nxt["target_positions"]):
                    record["anchor_target_logits"] = logits[idx]
                    valid.append((name, record))
                    break
    stride = max(1, len(valid) // args.limit) if args.limit else 1
    selected = valid[::stride][:args.limit] if args.limit else valid
    (args.capture / "prefixes.json").write_text(json.dumps([
        {"name": name, "prefix": record["prefix"]} for name, record in selected]))
    print(f"Validated {len(captures)} captured prefixes; selected {len(selected)} positions")
    if args.muse is None:
        return
    results = []
    for name, record in selected:
        item = {"name": name, "position": record["anchor_position"]}
        for kind, field in [("target", "anchor_target_logits"), ("mtp", "mtp_logits")]:
            values = np.fromfile(args.muse / f"{name}.{kind}_logits.f32", dtype=np.float32)
            item[kind] = distribution_metrics(record[field], values)
        for kind in ["target", "mtp"]:
            a = record[f"{kind}_hidden"].flatten().float()
            b = torch.from_numpy(np.fromfile(args.muse / f"{name}.{kind}_hidden.f32", dtype=np.float32))
            item[f"{kind}_hidden_cosine"] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        item["vllm_quality"] = proposal_quality(record["anchor_target_logits"], record["mtp_logits"])
        item["muse_quality"] = proposal_quality(
            np.fromfile(args.muse / f"{name}.target_logits.f32", dtype=np.float32),
            np.fromfile(args.muse / f"{name}.mtp_logits.f32", dtype=np.float32))
        results.append(item)
    summary = {}
    for kind in ["target", "mtp"]:
        for metric in ["top1_match", "top20_overlap", "js"]:
            summary[f"{kind}_{metric}"] = float(np.mean([r[kind][metric] for r in results]))
        summary[f"{kind}_hidden_cosine"] = float(np.mean([r[f"{kind}_hidden_cosine"] for r in results]))
    for backend in ["vllm", "muse"]:
        for metric in ["argmax_match", "expected_acceptance"]:
            summary[f"{backend}_{metric}"] = float(np.mean([r[f"{backend}_quality"][metric] for r in results]))
    accepted = [len(b["prefix"]) - len(a["prefix"]) - 1
                for (_, a), (_, b) in zip(captures, captures[1:])
                if len(b["prefix"]) <= len(sequence)]
    assert all(0 <= n <= 3 for n in accepted)
    summary["positions_compared"] = len(results)
    summary["vllm_verified_rounds"] = len(accepted)
    summary["vllm_accepted_drafts"] = sum(accepted)
    summary["vllm_proposed_drafts"] = 3 * len(accepted)
    summary["vllm_observed_aggregate_acceptance"] = sum(accepted) / (3 * len(accepted))
    summary["vllm_observed_unconditional_per_position"] = [
        sum(n >= i for n in accepted) / len(accepted) for i in (1, 2, 3)]
    if args.device:
        for kind in ["target", "mtp"]:
            pairs = [(np.fromfile(args.muse / f"{name}.{kind}_logits.f32", dtype=np.float32),
                      np.fromfile(args.device / f"{name}.{kind}_logits.f32", dtype=np.float32))
                     for name, _ in selected]
            summary[f"{kind}_device_hidden_max_abs_diff"] = max(float(np.max(abs(a - b))) for a, b in pairs)
    (args.muse / "comparison.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
