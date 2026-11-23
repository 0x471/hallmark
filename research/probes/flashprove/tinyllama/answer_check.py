"""The decisive CPU check of decision 0094 (run by scripts/flashprove_tinyllama_answer.py).

Usage: python answer_check.py MODEL_DIR DIGESTS_JSON CASE_JSON FLOAT_LOGITS_NPY OUT_JSON

MODEL_DIR holds integer_model.json, policy.txt and KL.bin for upstream FlashProve's integer model
(integrations/vllm/integer_model, with Hallmark's patches); DIGESTS_JSON pins those three files and
wsums.txt and fold.txt, and integer_model.json sets "verify_sums": true, so the weights'
integers are checked against wsums.txt. The program feeds one whole request window of T tokens:
SASH's prompt ids, SASH's answer ids, then the filler token 0 up to T, as the shipped tools pad.
It records:

- the greedy choice after each position; at positions r .. e - 1 (r = len(prompt) - 1, e = r +
  len(answer)) they must be SASH's answer, which is what a proof of this request would cover;
- that no range condition fails at any of the T positions (the integer model raises
  RangeConditionFailed otherwise, and the prover would make no proof);
- the largest absolute value of each named integer, the gate's lookup key among them (without the
  clamp gcl the provers refuse a key above 32,639);
- at the checked positions, the margin of the top logit over the second, and the slope and
  correlation of the integer logits (divided by 2^s) against the float reference's logits.

Equal tokens are a smoke test that the integer model is TinyLlama, not a proof that FlashProve's and
SASH's integer functions are equal.
"""

import collections
import json
import sys
import time

import numpy as np
import torch

LARGEST_GATE_KEY = 32639  # the provers make no proof for a larger key without the clamp gcl


def main(model_dir, digests_path, case_path, float_logits, out_json):
    import integer_model  # noqa: PLC0415 - from the patched upstream tree, on PYTHONPATH
    from integer_model.model import RangeConditionFailed  # noqa: PLC0415

    started = time.monotonic()
    case = json.loads(open(case_path).read())
    prompt, answer = case["prompt_ids"], case["answer_ids"]
    digests = json.loads(open(digests_path).read())
    model = integer_model.load_from_folder(model_dir, "cpu", {"sha256": digests})
    p = model.parameters
    window = prompt + answer + [0] * (p.T - len(prompt) - len(answer))
    r, e = len(prompt) - 1, len(prompt) + len(answer) - 1
    peaks = collections.defaultdict(int)
    logits = {}

    def record(name, layer, tensor):
        if not tensor.dtype.is_floating_point:
            peaks[name] = max(peaks[name], int(tensor.abs().max()))
        if name == "lam":
            logits["lam"] = tensor[r:e].to(torch.float64) / 2.0**p.s

    sequence = model.new_sequence()
    failure = None
    try:
        choices = model.extend(sequence, window, record=record).tolist()
    except RangeConditionFailed as fault:
        choices, failure = None, dict(position=fault.position, condition=fault.condition)
    result = dict(T=p.T, s=p.s, r=r, e=e, window=window, range_failure=failure)
    if choices is not None:
        got = choices[r:e]
        ours, theirs = logits["lam"].numpy(), np.load(float_logits)
        rows = []
        for i in range(len(got)):
            a, b = ours[i], theirs[i]
            top = np.sort(a)[-2:]
            rows.append(
                dict(
                    position=r + i,
                    token=got[i],
                    margin=float(top[1] - top[0]),
                    float_margin=float(np.diff(np.sort(b)[-2:])[0]),
                    slope=float((a * b).sum() / (b * b).sum()),
                    correlation=float(np.corrcoef(a, b)[0, 1]),
                )
            )
        result |= dict(
            choices=got,
            answer=answer,
            matches=got == answer,
            positions=rows,
            peaks=dict(sorted(peaks.items())),
            gate_key_within_limit=peaks["key"] <= LARGEST_GATE_KEY,
        )
    result["seconds"] = time.monotonic() - started
    with open(out_json, "x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__.split("\n\n")[1])
    main(*sys.argv[1:])
