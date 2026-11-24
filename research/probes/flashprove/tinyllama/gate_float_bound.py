"""The largest gate input of each layer over all prompts, in real units, from the float weights
(decision 0094). A check on worst_case_cpu.py that shares no code or file with it.

Usage: python gate_float_bound.py MODEL_DIR EG OUT_JSON

MODEL_DIR holds Hugging Face's model.safetensors of TinyLlama-1.1B-Chat (pinned by the caller).
After the norm before the feed-forward block, a row x of the stream becomes g * x / rms(x), whose
length is sqrt(d) for every x (up to the norm's epsilon), so the largest input of gate row j over
all prompts is at most sqrt(d) ||W_gate[j] * g||, and that much is reached by some direction of x.
The gate table reads that input as the key round(z 2^eg) (EG: the parameter file's eg), and
without the line gcl the provers make no proof for a key above 32,639 (worst_case.cuh,
QUOTIENT_TOP).
worst_case_cpu.py bounds the same key from the committed integers over the norm's ball, whose radius
is somewhat above 2,048 (the length of a normed row in integers); with the kit's exponents the
integer key is the real input times 2^eg, up to the weights' rounding, so its bound should be this
one times that radius over 2,048. The answer target checks that ratio in every layer.
"""

import json
import math
import sys

KEY_LIMIT = 32639  # QUOTIENT_TOP


def main(model_dir, eg, out_json):
    import torch  # noqa: PLC0415 - in the scales environment
    from safetensors.torch import load_file  # noqa: PLC0415

    eg = int(eg)
    weights = load_file(f"{model_dir}/model.safetensors")
    config = json.loads(open(f"{model_dir}/config.json").read())
    d, layers = config["hidden_size"], config["num_hidden_layers"]
    rows = []
    for layer in range(layers):
        gains = weights[f"model.layers.{layer}.post_attention_layernorm.weight"].double()
        gate = weights[f"model.layers.{layer}.mlp.gate_proj.weight"].double()
        lengths = (gate * gains).norm(dim=1)
        z = math.sqrt(d) * float(lengths.max())
        key = z * 2**eg
        rows.append(
            dict(
                layer=layer,
                row=int(lengths.argmax()),
                largest_input=z,
                key=key,
                rows_beyond_the_limit=int((math.sqrt(d) * lengths * 2**eg > KEY_LIMIT).sum()),
            )
        )
        print(f"layer {layer}: largest gate input {z:.3f}, key {key:.0f}", flush=True)
    over = [r["layer"] for r in rows if r["key"] > KEY_LIMIT]
    result = dict(
        program="research/probes/flashprove/tinyllama/gate_float_bound.py",
        eg=eg,
        key_limit=KEY_LIMIT,
        layers_over_the_limit=over,
        layers=rows,
        torch=torch.__version__,
    )
    open(out_json, "w").write(json.dumps(result, indent=1) + "\n")
    print("layers over the limit:", over)


if __name__ == "__main__":
    main(*sys.argv[1:])
