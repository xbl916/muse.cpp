"""Isolated vLLM capture: run with CUDA_VISIBLE_DEVICES and an output directory."""
import json
import os
from pathlib import Path

import torch
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

OUT = Path(os.environ["MTP_COMPARE_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
state = {"serial": 0, "prefix": []}
original_logits = Qwen3_5ForConditionalGeneration.compute_logits
original_propose = SpecDecodeBaseProposer.propose
original_sample = SpecDecodeBaseProposer._sample_draft_tokens


def target_logits(self, hidden_states, *args, **kwargs):
    result = original_logits(self, hidden_states, *args, **kwargs)
    if result is not None:
        state["target_logits"] = result.detach().float().cpu()
    return result


def propose(self, *args, **kwargs):
    import inspect
    bound = inspect.signature(original_propose).bind(self, *args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    ids = values["target_token_ids"].detach().cpu().tolist()
    pos = values["target_positions"].detach().cpu()
    if pos.ndim > 1:
        pos = pos[0]
    pos = pos.tolist()
    idx = values["token_indices_to_sample"]
    idx = len(ids) - 1 if idx is None else int(idx[0].item())
    anchor = int(values["next_token_ids"][0].item())
    prefix = state["prefix"]
    for p, token in zip(pos[:idx + 1], ids[:idx + 1]):
        if p < len(prefix):
            prefix[p] = token
        elif p == len(prefix):
            prefix.append(token)
        else:
            raise RuntimeError("Non-contiguous token prefix")
    prefix = prefix[:pos[idx] + 1] + [anchor]
    state["prefix"] = prefix
    state["capture"] = {
        "prefix": prefix.copy(),
        "target_positions": pos,
        "target_ids": ids,
        "target_logits": state.get("target_logits"),
        "target_hidden": values["target_hidden_states"][idx].detach().float().cpu(),
        "anchor_position": pos[idx] + 1,
    }
    try:
        return original_propose(self, *args, **kwargs)
    finally:
        state.pop("capture", None)


def sample(self, hidden_states, sampling_metadata):
    capture = state.pop("capture", None)
    if capture is not None:
        logits = self.model.compute_logits(hidden_states)
        if logits is not None:
            capture["mtp_logits"] = logits.detach().float().cpu()
            capture["mtp_hidden"] = hidden_states.detach().float().cpu()
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                torch.save(capture, OUT / f"step-{state['serial']:05d}.pt")
                state["serial"] += 1
    return original_sample(self, hidden_states, sampling_metadata)


Qwen3_5ForConditionalGeneration.compute_logits = target_logits
SpecDecodeBaseProposer.propose = propose
SpecDecodeBaseProposer._sample_draft_tokens = sample


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    model = os.environ.get("MTP_COMPARE_MODEL", "/pubdata/llm_models/cyankiwi/Qwen3.8-27B-AWQ-INT4")
    llm = LLM(model=model, tensor_parallel_size=2, max_model_len=4096,
              max_num_seqs=1, gpu_memory_utilization=0.65, enforce_eager=True,
              enable_prefix_caching=False, max_num_batched_tokens=1024,
              speculative_config={"method": "mtp", "num_speculative_tokens": 3},
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=1007843)
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content": "画一个鹈鹕骑自行车的svg"}]
    tokens = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
    (OUT / "prompt.json").write_text(json.dumps(tokens))
    result = llm.generate([{"prompt_token_ids": tokens}], SamplingParams(
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=1.0,
        max_tokens=256, seed=1007843, ignore_eos=True))
    (OUT / "generation.json").write_text(json.dumps({
        "tokens": list(result[0].outputs[0].token_ids), "text": result[0].outputs[0].text}))
    print("CAPTURE_COMPLETE", flush=True)
