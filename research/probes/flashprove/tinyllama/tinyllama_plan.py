"""Write a TinyLlama-1.1B-shaped dense parameter file and KL.bin for the patched verifier's planner.

usage: python3 -I -B tinyllama_plan.py TREE OUT

TREE is the pinned upstream source with the three patches applied; OUT must not exist. It starts
from upstream's prover/examples/target_full_f2 (Llama 4 Maverick's shape) and makes it dense the
way the patched verifier/tests/served_form_test.py does for its dense case:

- the shape lines become TinyLlama-1.1B's, with T = 64 tokens (lt 6);
- NE 0 and hr 0, the expert-only weight widths 0, no rcl or gcl lines, and the router's and the
  clamps' segments taken out of the call lines;
- iota_0 .. iota_7 refitted with upstream's verifier/tests/forest_lines.py;
- the three table digests a dense model forms (no tsig), from upstream's scales/policy_tables.py.
  It refuses NE 0, but no table's entries depend on NE (NE only decides whether the sigmoid table
  is formed), so they are computed on a copy with NE 4; the verifier compares them with its own;
- KL.bin: the full kit's first 22 layer records, with rotary positions in every layer.

Every other line is the full kit's, except the few that TinyLlama's shape forces, set in FORCED.
The scales and those lines are placeholders for planning, not a fitted kit: they test that the
verifier's planner admits this shape, not that a proof can be made with them. It reads no weights.
"""

import struct
import sys
from pathlib import Path

tree, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(tree / "scales"), str(tree / "verifier/tests")]
import forest_lines  # noqa: E402
import policy_tables  # noqa: E402

BASE = tree / "prover/examples/target_full_f2"
LAYERS = 22
RECORD = 13  # words of one layer's record in KL.bin; the last is the flag for rotary positions
# TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json: hidden 2048, 32 heads of 64, 4 KV heads,
# intermediate 5632, 22 layers, vocabulary 32000, rope_theta 10000. ll 1: 2 divides 22.
SHAPE = dict(lt=6, L=LAYERS, ll=1, d=2048, H=32, G=4, dh=64, f=5632, V=32000, theta0_lo=10000)
DENSE = dict(NE=0, hr=0, wb_gs=0, wb_us=0, wb_gr=0, wb_ur=0, wb_dnr=0, wb_r=0)
# Lines whose full-kit value the verifier refuses at this shape (verifier/docs/parameter-file.md).
FORCED = {
    "la": 12,  # la - lt stays the full kit's 6 (16 - 10); 2^6 divides d, f, G dh and (H + G) dh
    "lB": 4,  # at most lt - 2 (af-shape-tiles); the full kit's 6 is above it
    "lmS": 12,  # at most log2 G + lt + lB = 2 + 6 + 4 (af-shape-lmS); the full kit's is 16
    "kFD": 15,  # the softmax table's rows must fit 2^kFD with little room to spare (af-policy-kF)
    "iotaA_0": 52,  # 32 + dT of the attention forest, dT 20 at this shape (policy-iotaA)
    "iotaH_0": 0,  # n - dT of the output head's calls, 0 at V 32000 (shape-call-H-*)
    "iotaH_1": 0,
    "iotaH_2": 0,
    "ranges-part": 13,  # a part may read no more rows than the matrix has (parameters-not-admitted)
}
MOE_SEGMENTS = {"XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG"}


def put(lines, name, value):
    hits = [i for i, x in enumerate(lines) if x.split(" ")[0] == name and len(x.split(" ")) == 2]
    if len(hits) != 1:
        sys.exit(f"tinyllama_plan.py: the full kit has {len(hits)} lines {name}")
    lines[hits[0]] = f"{name} {value}"


def policy():
    lines = (BASE / "policy.txt").read_text().splitlines()
    for name, value in (SHAPE | DENSE | FORCED).items():
        put(lines, name, value)
    lines = [x for x in lines if x.split(" ")[0] not in ("rcl", "gcl")]
    calls = []
    for x in lines:
        if x.startswith("call "):
            head, segments = x.split(" ")[:3], x.split(" ")[3:]
            kept = [s for s in segments if s not in MOE_SEGMENTS]
            x = " ".join(head + kept) if kept else None
        if x is not None:
            calls.append(x)
    lines = calls
    value = {
        w[0]: int(w[1]) for w in (x.split(" ") for x in lines) if len(w) == 2 and w[1].isdigit()
    }
    fit = forest_lines.layer(value, forest_lines.calls_of("\n".join(lines)))
    lines = [
        f"{x.split(' ')[0]} {fit[x.split(' ')[0]]}" if x.split(" ")[0] in fit else x for x in lines
    ]
    with_experts = ["NE 4" if x == "NE 0" else x for x in lines]
    digests = policy_tables.digests([x for x in with_experts if not policy_tables.is_table_line(x)])
    tables = [x for x in policy_tables.table_lines(digests) if not x.startswith("tsig")]
    lines = [x for x in lines if not policy_tables.is_table_line(x)]
    at = next(i for i, x in enumerate(lines) if x.startswith("call "))
    return lines[:at] + tables + lines[at:]


def scales():
    data = (BASE / "KL.bin").read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if len(words) != 48 * RECORD + 72:  # 48 layer records, then 3 words for each of 24 MoE layers
        sys.exit("tinyllama_plan.py: the full kit's KL.bin is not as expected")
    records = [list(words[RECORD * i : RECORD * (i + 1)]) for i in range(LAYERS)]
    for record in records:
        record[-1] = 1
    return struct.pack(f"<{RECORD * LAYERS}I", *(w for r in records for w in r))


if __name__ == "__main__":
    if out.exists():
        sys.exit(f"tinyllama_plan.py: {out} exists")
    lines, kl = policy(), scales()
    out.mkdir(parents=True)
    (out / "policy.txt").write_text("\n".join(lines) + "\n")
    (out / "KL.bin").write_bytes(kl)
