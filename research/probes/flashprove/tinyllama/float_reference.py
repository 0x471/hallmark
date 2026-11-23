"""The float reference of the TinyLlama answer check (scripts/flashprove_tinyllama_answer.py).

Usage: python float_reference.py SCALES_DIR GGUF_DIR MODEL_DIR CASE_JSON OUT_JSON OUT_LOGITS_NPY

It runs SASH's prompt (CASE_JSON's prompt_ids, then answer_ids but the last) through two float32
evaluations of TinyLlama-1.1B-Chat:

- upstream FlashProve's float pass (scales/float_forward.py, with the patched scales/gguf_weights.py
  reading the exported GGUF in GGUF_DIR), which calibrate.py uses to fit the scales;
- Hugging Face transformers' LlamaForCausalLM on the published safetensors in MODEL_DIR.

Their final-norm outputs must agree to float32 rounding: that shows that the export (names, Q/K
row order, norm gains) and the patched reading of the shape are the publisher's model. It also runs
transformers'
greedy decoding from the prompt and saves its logits at the checked positions, which the integer
check compares with the integer model's. The caller pins every input; this program only computes.
"""

import json
import sys
import time

import numpy as np
import torch


def main(scales, gguf_dir, model_dir, case_path, out_json, out_logits):
    sys.path.insert(0, scales)
    import float_forward  # noqa: PLC0415 - from the patched upstream tree
    import gguf_weights  # noqa: PLC0415
    from transformers import LlamaForCausalLM  # noqa: PLC0415

    started = time.monotonic()
    torch.manual_seed(0)
    case = json.loads(open(case_path).read())
    prompt, answer = case["prompt_ids"], case["answer_ids"]
    ids = prompt + answer[:-1]  # the positions whose next tokens are the answer

    model = gguf_weights.open_model(gguf_dir)
    shape = gguf_weights.shape_words(model)
    ours = float_forward.forward(model, shape, torch.tensor([ids]), lambda *a: None)[0]
    hf = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = hf.model(torch.tensor([ids])).last_hidden_state[0]
        logits = hf(torch.tensor([ids])).logits[0].double()
        greedy = list(prompt)
        for _ in range(len(answer)):
            nxt = int(hf(torch.tensor([greedy])).logits[0, -1].argmax())
            greedy.append(nxt)
            if nxt == 2:
                break
    checked = logits[len(prompt) - 1 :].numpy()
    np.save(out_logits, checked)
    result = dict(
        shape={k: v for k, v in shape.items() if k not in ("moe", "rope")},
        moe_layers=shape["moe"],
        rope_layers=len(shape["rope"]),
        final_norm_relative_l2=float((ours - theirs).norm() / theirs.norm()),
        final_norm_max_abs=float((ours - theirs).abs().max()),
        float_greedy=greedy[len(prompt) :],
        float_greedy_is_the_answer=greedy[len(prompt) :] == answer,
        checked_logits=dict(positions=[len(prompt) - 1, len(ids) - 1], shape=list(checked.shape)),
        seconds=time.monotonic() - started,
    )
    with open(out_json, "x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__.split("\n\n")[1])
    main(*sys.argv[1:])
