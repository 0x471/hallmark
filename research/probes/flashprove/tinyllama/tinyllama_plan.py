"""Write a TinyLlama-1.1B-shaped dense parameter file and KL.bin for the patched verifier's planner.

usage: python3 -I -B tinyllama_plan.py TREE OUT

TREE is the pinned upstream source with the three program patches applied (scales.patch too, in
the answer target: it changes nothing this file reads); OUT must not exist. It starts
from upstream's prover/examples/target_full_f2 (Llama 4 Maverick's shape) and makes it dense the
way the patched verifier/tests/served_form_test.py does for its dense case:

- the shape lines become TinyLlama-1.1B's, with T = 64 tokens (lt 6);
- NE 0 and hr 0, the expert-only weight widths 0, no rcl or gcl lines, and the router's and the
  clamps' segments taken out of the call lines;
- the line eg 7, which verifier.patch admits in the standard form at NE 0 only: without gcl a gate's
  key must fit two limbs, and eg 7 doubles the gate inputs that this covers;
- kFM and kFD, the log2 of the rows of the two tables that hold the softmax table, set by the rule
  of cp_prove's plan of attention (softmax_rows());
- iota_0 .. iota_7 refitted with upstream's verifier/tests/forest_lines.py;
- the three table digests a dense model forms (no tsig), from upstream's scales/policy_tables.py,
  the SiLU table's for eg 7. It refuses NE 0, but no table's entries depend on NE (NE only decides
  whether the sigmoid table is formed), so they are computed on a copy with NE 4; the verifier
  compares them with its own;
- KL.bin: the full kit's first 22 layer records, with rotary positions in every layer.

Every other line is the full kit's, except the few that TinyLlama's shape forces, set in FORCED.
The scales and those lines are placeholders for planning, not a fitted kit: they test that the
verifier's planner admits this shape, not that a proof can be made with them. It reads no weights.
"""

import math
import struct
import sys
from pathlib import Path

LAYERS = 22
RECORD = 13  # words of one layer's record in KL.bin; the last is the flag for rotary positions
# TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json: hidden 2048, 32 heads of 64, 4 KV heads,
# intermediate 5632, 22 layers, vocabulary 32000, rope_theta 10000. ll 1: 2 divides 22 (the full
# kit's ll 2 would need 4 to). patches.json pins this shape; the harness and a test compare them.
SHAPE = dict(lt=6, L=LAYERS, ll=1, d=2048, H=32, G=4, dh=64, f=5632, V=32000, theta0_lo=10000)
DENSE = dict(NE=0, hr=0, wb_gs=0, wb_us=0, wb_gr=0, wb_ur=0, wb_dnr=0, wb_r=0)
# Lines whose full-kit value the verifier refuses at this shape (verifier/docs/parameter-file.md).
FORCED = {
    "la": 12,  # la - lt stays the full kit's 6 (16 - 10); 2^6 divides d, f, G dh and (H + G) dh
    "lB": 4,  # at most lt - 2 (af-shape-tiles); the full kit's 6 is above it
    "lmS": 12,  # at most log2 G + lt + lB = 2 + 6 + 4 (af-shape-lmS); the full kit's is 16
    "iotaA_0": 52,  # 32 + dT of the attention forest, dT 20 at this shape (policy-iotaA)
    "iotaH_0": 0,  # n - dT of the output head's calls, 0 at V 32000 (shape-call-H-*)
    "iotaH_1": 0,
    "iotaH_2": 0,
    # The largest value admitted: from 15 a part reads more rows than the matrix has, and at 14 a
    # single mask row of once-ranges has a code of a larger rate than its folded rows (both
    # parameters-not-admitted).
    "ranges-part": 13,
}
BASE = "prover/examples/target_full_f2"
GATE_KEY = "eg 7"  # a key c of the SiLU table means c / 2^7 (parameter-file.md, 9)
MOE_SEGMENTS = {"XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG"}


def softmax_rows(words):
    """kFM and kFD: for each, the one k with u0 + n <= 2^k <= u0 + 2 n at every attention shift sh
    from 4 to 12, where n is nHM (kFM) or nHD (kFD). cp_prove's plan of attention refuses a file
    that misses it at a layer's shift (APlan::build(), prover/attn/plan.cuh); cp_verify checks it
    for kFD only (af-policy-kF) and does not read kFM. u0 is the length of the exponential table
    (APlan::expTable()), nHM = (2 Smax + rho) // rho and nHD = (4 Smax + rho) // rho with
    Smax = dh 2047^2 and rho = 65536 - 2^sh (APlan::shd())."""
    s, rx, smax = words["S"] % 32, words["rx"], words["dh"] * 2047 * 2047
    u0 = 0
    while round(math.ldexp(math.exp(-u0 / (1 << rx)), s)) >= 1:  # nearbyint: half to even
        u0 += 1
    rows = {}
    for name, times in (("kFM", 2), ("kFD", 4)):
        sizes = [(times * smax + rho) // rho for rho in (65536 - (1 << sh) for sh in range(4, 13))]
        fits = [k for k in range(31) if all(u0 + n <= 1 << k <= u0 + 2 * n for n in sizes)]
        if len(fits) != 1:
            sys.exit(f"tinyllama_plan.py: {name}: {len(fits)} values fit every shift")
        rows[name] = fits[0]
    return rows


def put(lines, name, value):
    hits = [i for i, x in enumerate(lines) if x.split(" ")[0] == name and len(x.split(" ")) == 2]
    if len(hits) != 1:
        sys.exit(f"tinyllama_plan.py: the full kit has {len(hits)} lines {name}")
    lines[hits[0]] = f"{name} {value}"


def upstream(tree):
    """Upstream's verifier/tests/forest_lines.py and scales/policy_tables.py, from the tree."""
    sys.path[:0] = [str(tree / "scales"), str(tree / "verifier/tests")]
    import forest_lines
    import policy_tables

    return forest_lines, policy_tables


def policy(tree):
    forest_lines, policy_tables = upstream(tree)
    lines = (tree / BASE / "policy.txt").read_text().splitlines()
    for name, value in (SHAPE | DENSE | FORCED).items():
        put(lines, name, value)
    words = {
        w[0]: int(w[1]) for w in (x.split(" ") for x in lines) if len(w) == 2 and w[1].isdigit()
    }
    for name, value in softmax_rows(words).items():
        put(lines, name, value)
    lines = [x for x in lines if x.split(" ")[0] not in ("rcl", "gcl")]
    at = next(
        i for i, x in enumerate(lines) if x.startswith("nminH ")
    )  # eg comes directly after it
    lines = lines[: at + 1] + [GATE_KEY] + lines[at + 1 :]
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


def scales(tree):
    data = (tree / BASE / "KL.bin").read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if len(words) != 48 * RECORD + 72:  # 48 layer records, then 3 words for each of 24 MoE layers
        sys.exit("tinyllama_plan.py: the full kit's KL.bin is not as expected")
    records = [list(words[RECORD * i : RECORD * (i + 1)]) for i in range(LAYERS)]
    for record in records:
        record[-1] = 1
    return struct.pack(f"<{RECORD * LAYERS}I", *(w for r in records for w in r))


if __name__ == "__main__":
    tree, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        sys.exit(f"tinyllama_plan.py: {out} exists")
    lines, kl = policy(tree), scales(tree)
    out.mkdir(parents=True)
    (out / "policy.txt").write_text("\n".join(lines) + "\n")
    (out / "KL.bin").write_bytes(kl)
