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
- the line table, the committed table's size, set for this model's weight rows (table_numbers());
  the full kit's line would have cp_commit encode Maverick's table for a model 300 times smaller;
- iota_0 .. iota_7 refitted with upstream's verifier/tests/forest_lines.py;
- the three table digests a dense model forms (no tsig), from upstream's scales/policy_tables.py,
  the SiLU table's for eg 7. It refuses NE 0, but no table's entries depend on NE (NE only decides
  whether the sigmoid table is formed), so they are computed on a copy with NE 4; the verifier
  compares them with its own;
- KL.bin: the full kit's first 22 layer records, with rotary positions in every layer.

Every other line is the full kit's, except the few that TinyLlama's shape forces, set in FORCED.
The scales and those lines are placeholders for planning, not a fitted kit: they test that the
verifier's planner admits this shape, not that a proof can be made with them. It reads no weights.
The other proof-system lines (group, stages and the matrices' sizes) stay the full kit's: they are
admitted, and larger than this model needs.
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
# The committed table's masks (the fourth number of the line table) come from upstream's small
# example kit, 65,535. The full kit's 44,040,191 are sized for hundreds of thousands of batches on
# one stored copy and would force a row of at least 2^26 message entries, 51 times this model's
# data. The masks bound how many batches a stored copy bears; the planner prints the number.
SMALL = "prover/examples/target_small_f2"
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


def weight_rows(words):
    """The rows of W, the matrix of weight digits that cp_commit commits to: upstream's layoutW()
    (prover/common/weight_layout.h) as HfShape::from_policy() calls it
    (prover/hash_commitment/hf_shape.cuh), for the row count alone. Each unit (one layer's
    tensors of one kind, the embedding, the output head) gives some of its index bits to W's
    2^lkW columns and the rest to rows; the sum is rounded up to a multiple of the ring degree D.
    The answer target compares the result with layoutW() itself (prover_setup.cpp)."""
    lk, ring, experts = words["table"][5], words["D"], words["NE"]
    d, dh, f, fe = words["d"], words["dh"], words["f"], words["fe"]
    keys, queries = words["G"] * dh, words["H"] * dh
    moe_layers = (words["prof"] & 15) >= 3 and experts > 0

    def rows(digits, *groups):
        left, total = lk, digits
        for extent, kind in groups:
            if kind == "A":  # the whole index goes to the columns, or none of it
                bits = (extent - 1).bit_length()
                take = bits if bits <= left and extent > 1 else 0
                total *= 1 if take or extent == 1 else extent
            else:  # "S", "P": as many of the index's trailing zero bits as are left
                take = min((extent & -extent).bit_length() - 1, left)
                total *= extent >> take
            left -= take
        return total

    total = 0
    for layer in range(words["L"]):
        moe = moe_layers and layer % 2 == 1
        total += rows(2, (64, "A"), (dh, "A"), (d, "S"))  # ShapeW::hdR is 64
        total += rows(2, (d, "S"), (keys, "P")) + rows(2, (queries, "S"), (d, "P"))
        if moe:
            total += 2 * rows(2, (d, "S"), (fe, "P"))
            total += 2 * rows(2, (d, "S"), (experts, "S"), (fe, "P"))
            total += rows(2, (fe, "S"), (d, "P"))
            total += rows(2, (fe, "S"), (experts, "S"), (d, "P"))
            total += rows(2, (d, "S"), (experts, "P"))
        else:
            total += 2 * rows(2, (d, "S"), (f, "P")) + rows(2, (f, "S"), (d, "P"))
    if words["prof"] >> 4:  # the embedding and the output head
        total += rows(4, (words["V"], "A"), (d, "P")) + rows(2, (words["V"], "A"), (d, "S"))
    return -(-total // ring) * ring


def table_numbers(rows, row_bits, masks, column_bits, answers=1):
    """The six numbers of the line `table` (fb fln fi0 fnm 0 lkW) for a matrix W of `rows` rows of
    2^column_bits columns, held in 2^row_bits long rows with `masks` random entries each.

    - fi0, a long row's message entries: the least number that HfTableParams::table_derive()
      (prover/hash_commitment/hf_table_params.h) admits: at least the row's data entries,
      rows 2^(lkW - fb), and three pads for each answer, and a multiple of both 2^(lkW - fb) and
      the least power of two that is at least fnm;
    - fln, the log2 of the code's length: three more than the bits of fi0 + fnm - 1, as in each
      of upstream's example kits (table_derive() asks for at least one more).

    With upstream's own rows, row_bits and masks this gives the line of each of upstream's example
    kits (target_small_f2, target_small_file_f2, target_full_f2; tested). The rule is read off
    those kits and that function; upstream states none."""
    low = column_bits - row_bits
    # table_lines_first() and table_derive(): lkW from 4 to 18, fb from 3 to 13 and below lkW, fb
    # at most the bits of a row number of W, fnm odd.
    shape = 4 <= column_bits <= 18 and 3 <= row_bits <= 13 and row_bits < column_bits
    if not (shape and row_bits <= (rows - 1).bit_length() and masks % 2):
        sys.exit("tinyllama_plan.py: table: fb, lkW or fnm is not admitted")
    step = max(1 << low, 1 << (masks - 1).bit_length(), 2)
    needed = (rows << low) + 3 * answers
    data = -(-needed // step) * step
    length = (data + masks - 1).bit_length() + 3
    if length > 32 or data >> 32:
        sys.exit("tinyllama_plan.py: table: the code would be longer than 2^32")
    return [row_bits, length, data, masks, 0, column_bits]


def frame_numbers(lines, name):
    """The numbers of the one line that begins with `name` and a blank."""
    hits = [x for x in lines if x.startswith(name + " ")]
    if len(hits) != 1:
        sys.exit(f"tinyllama_plan.py: {len(hits)} lines {name}")
    return [int(w) for w in hits[0].split(" ")[len(name.split(" ")) :]]


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
    # The committed table: the full kit's fb and lkW, the small kit's number of masks.
    full = frame_numbers(lines, "table")
    masks = frame_numbers((tree / SMALL / "policy.txt").read_text().splitlines(), "table")[3]
    answers = frame_numbers(lines, "answers")[0]
    table = table_numbers(weight_rows(words | {"table": full}), full[0], masks, full[5], answers)
    lines[lines.index("table " + " ".join(map(str, full)))] = "table " + " ".join(map(str, table))
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
