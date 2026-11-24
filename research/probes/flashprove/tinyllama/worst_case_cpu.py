"""cp_commit's worst-case check on a CPU, for a kit without experts (decision 0094). Hallmark's port
of upstream's prover/weights_commitment/worst_case.cuh, not upstream's code.

Usage: python worst_case_cpu.py MODEL_DIR DIGESTS_JSON OUT_JSON

MODEL_DIR and DIGESTS_JSON are as for answer_check.py: the integer model's directory (the kit's
policy.txt and KL.bin, the GGUF file, wsums.txt and fold.txt), every file pinned, the weights'
integers compared with wsums.txt. So the integers read here are the ones that wsums.txt lists.

Upstream computes the check on a GPU inside cp_commit (worst_case.cuh: struct Check, take() and
lines()); scales/worst-case-bounds.md derives its bounds. This program follows that file for a
parameter file with NE 0 and writes one line per site and layer, as lines() does, with the bound,
the limit, the shortfall in whole bits (0: the site fits) and the headroom in bits:

- the sites that a row bound decides (q, k, v, gate, up; take(), cases 0, 1, 3 and 4). Every row's
  bound is computed in floating point; the rows that can decide a site are computed again in pure
  Python with the GPU's directed rounding emulated exactly (each step rounded to the least binary64
  value at or above its exact result, as __dmul_ru, __ddiv_ru, __dadd_ru, __ll2double_ru and
  __dsqrt_ru round), and that value is kept. So each of these sites' bounds is the integer that
  k_row_bounds() gives, if the GPU rounds as IEEE 754 states;
- the sites that public parameters decide (the norms' quotients, the head's margin), as lines()
  computes them;
- the residual stream's sites, as lines() computes them but with each matrix's largest singular
  value replaced by its Frobenius norm, rounded up. That is never below the GPU's bound, whose first
  level is the Frobenius norm (operator_norm()), and every later step is monotone. So a stream site
  that fits here fits on the GPU; one that does not fit here is not shown to fail there.

Not covered: the guards on cuBLAS's products (product_holds(), gram_is_exact()), which can only
make a GPU site fail; that cp_commit's integers equal the ones read here (cp_commit converts the
file with its own code and is compared with wsums.txt only through that file); anything else the
GPU program does that this file does not restate.
"""

import json
import math
import sys
import time
from fractions import Fraction

# worst_case.cuh's constants.
WEIGHT_MOST = 4095
QUOTIENT_TOP = 32639
QUOTIENT_LOW = -32768
CELL_TOP = 2047
SQUARES_BITS = 59
GATE_KEY_BITS = 8
GATE_OUT_STEPS = 64
LAST_STREAM_BITS = 30
HEAD_LOW_BITS = 7
HEAD_LIFT_BITS = 37
BINS_MOST = 250
GATE_TOPS = (4095, 2047, 1023, 511, 255)
WIDE_STREAM_TOP = 547599908735
WIDE_SQUARES_BITS = 78
WIDE_GATE_TOP = 256 * 16383 + 127
LAYER_WORDS = ("kq", "kk", "kv", "ko", "kg", "ku", "kd", "M1", "K1", "M2", "K2", "sh", "rp")
ROW_SITES = ("q", "k", "v", "gate", "up")
PUBLIC_SITES = ("norm 1 quotient", "norm 2 quotient", "last norm quotient", "head margin")
INF = math.inf


# The host arithmetic of worst_case.cuh: round to nearest, then one step up.
def up(x):
    return math.nextafter(x, INF)


def add_up(a, b):
    return up(a + b)


def mul_up(a, b):
    return up(a * b)


def div_up(a, b):
    return up(a / b)


def sqrt_up(a):
    return up(math.sqrt(a))


def whole_up(v):
    return float(v) if v < 1 << 53 else up(float(v))


def quotient(a, k):
    """round(A / 2^k): the largest |round(s / 2^k)| over the sums |s| <= A."""
    return (a + (1 << (k - 1))) >> k


# The GPU's rounding upwards, emulated exactly.
def ru(q):
    """The least binary64 value at or above the rational q."""
    f = float(q)  # correctly rounded to nearest
    return f if Fraction(f) >= q else math.nextafter(f, INF)


def dsqrt_ru(a):
    """The least binary64 value at or above sqrt(a), for a binary64 value a >= 0."""
    f = math.sqrt(a)  # correctly rounded to nearest
    return f if Fraction(f) ** 2 >= Fraction(a) else math.nextafter(f, INF)


def tau_grid():
    """tau_grid(): every integer up to 67, then steps of 3 %, up to WEIGHT_MOST."""
    edge = [0]
    while edge[-1] < WEIGHT_MOST:
        edge.append(min(WEIGHT_MOST, max(edge[-1] + 1, edge[-1] * 103 // 100)))
    return edge


EDGES = tau_grid()
BINS = len(EDGES) - 1
BIN_OF = [0] * (WEIGHT_MOST + 1)
for _b in range(1, BINS + 1):
    for _v in range(EDGES[_b - 1] + 1, EDGES[_b] + 1):
        BIN_OF[_v] = _b


def row_bound_exact(row, ball2, box):
    """k_row_bounds() for one row of integers: (the bound, sum w^2). Pure Python; every step as the
    GPU rounds it. The bound is an integer at or above every |w.a| with ||a||^2 <= ball2 and
    |a_k| <= box: the least of the dual bound at each tau of the grid, the bound by the box only
    and the bound by the ball only (worst_case.cuh, top comment, 1)."""
    count, sum1, sum2 = [0] * (BINS + 1), [0] * (BINS + 1), [0] * (BINS + 1)
    for value in row:
        a = abs(int(value))
        if a > WEIGHT_MOST:
            raise ValueError("a weight exceeds 4,095: the GPU's check gives no bound")
        if a:
            b = BIN_OF[a]
            count[b] += 1
            sum1[b] += a
            sum2[b] += a * a
    ball2_q, box_q = Fraction(ball2), Fraction(box)
    above, above1 = sum(count), sum(sum1)
    below2, least = 0, INF
    for b in range(1, BINS + 1):
        below2 += sum2[b]
        above -= count[b]
        above1 -= sum1[b]
        tau = EDGES[b]
        inner = below2 + 2 * tau * above1 - tau * tau * above  # exact, not negative
        first = ru(Fraction(ru(Fraction(tau * tau) * ball2_q)) / box_q)
        second = ru(box_q * Fraction(ru(Fraction(inner))))
        least = min(least, ru(Fraction(ru(Fraction(first) + Fraction(second))) / (2 * tau)))
    all1, all2 = sum(sum1), sum(sum2)
    by_box = ru(box_q * Fraction(ru(Fraction(all1))))
    by_ball = dsqrt_ru(ru(ball2_q * Fraction(ru(Fraction(all2)))))
    return math.floor(min(least, by_box, by_ball)), all2


def ball(M, K, eps, d, xl, one_cell=False):
    """Check::ball(): the radius B after a norm, or with one_cell a bound on one unclamped cell."""
    first = mul_up(math.ldexp(1.0, M - K), add_up(1.0, div_up(1.0, float(math.isqrt(eps)))))
    if xl == 5:
        root = add_up(math.ldexp(1.0, WIDE_SQUARES_BITS // 2), whole_up(math.isqrt(eps) + 2))
    else:
        root = whole_up(math.isqrt((1 << SQUARES_BITS) + eps) + 2)
    second = math.ldexp(root, -(K + 1))
    last = 0.5 if one_cell else div_up(sqrt_up(float(d)), 2.0)
    return add_up(add_up(first, second), last)


def gate_most(eg=GATE_KEY_BITS, smax=GATE_TOPS[0]):
    """lay_the_gate_table(): entry g is the gate table's largest |output| on the keys -g .. g, plus
    one. A key whose output lies within 1e-9 of a rounding boundary takes the larger value, so that
    no library's exp() gives more."""
    most, out = 0, []
    key_top = (GATE_OUT_STEPS << eg) - 1
    for g in range(-QUOTIENT_LOW + 1):
        for c in (g, -g):
            c = max(-key_top - 1, min(key_top, c))
            z = math.ldexp(float(c), -eg)
            s = abs(z / (1 + math.exp(-z)) * GATE_OUT_STEPS)
            v = math.floor(s + 0.5)
            if abs(s + 0.5 - round(s + 0.5)) < 1e-9:
                v = round(s + 0.5)
            most = max(most, min(smax, v + 1))
        out.append(most)
    return out


def sum_limit(k, top=QUOTIENT_TOP):
    """lines(), sum(): round(A / 2^k) <= top means A <= (2 top + 1) 2^(k - 1) - 1."""
    return math.ldexp(float(2 * top + 1), k - 1) - 1


def shortfall(bound, limit):
    """shortfall_of(): the least b >= 0 with bound <= limit 2^b (99 for a bound that is NaN)."""
    b = 0
    while b < 99 and not (bound <= math.ldexp(limit, b)):
        b += 1
    return b


def pair_bounds(b, s, dot, B1, rp, turn):
    """take(), case 0: the bound of each pair (2 i, 2 i + 1) of q's or k's rows, as the host
    computes it from the rows' bounds b, sums of squares s and dot products."""
    out = []
    for i in range(len(dot)):
        b0, b1 = b[2 * i], b[2 * i + 1]
        if not rp:
            out.append(4096 * max(b0, b1))
            continue
        n0, n1, p = float(s[2 * i]), float(s[2 * i + 1]), abs(float(dot[i]))  # below 2^53: exact
        gap = abs(n0 - n1)
        root = sqrt_up(add_up(mul_up(gap, gap), mul_up(4.0, mul_up(p, p))))
        top = div_up(add_up(n0 + n1, root), 2.0)
        by_ball = mul_up(B1, sqrt_up(top))
        f0, f1 = whole_up(b0), whole_up(b1)
        by_rows = sqrt_up(add_up(mul_up(f0, f0), mul_up(f1, f1)))
        out.append(math.floor(mul_up(turn, min(by_ball, by_rows))))
    return out


def row_bounds(w, ball2, box):
    """Every row's bound in floating point, and the exact bound of the rows that can be the
    largest. Returns (bounds, squares, exact, make_exact): bounds holds an int per row; exact maps
    the rows computed exactly to their bound; make_exact(rows) computes more rows exactly."""
    import numpy as np  # noqa: PLC0415 - in the integer model's environment only

    rows = w.shape[0]
    a = np.abs(w.astype(np.int64))
    if int(a.max()) > WEIGHT_MOST:
        raise SystemExit("a weight exceeds 4,095: the GPU's check gives no bound")
    bin_of = np.array(BIN_OF, dtype=np.int64)
    flat = (np.arange(rows, dtype=np.int64)[:, None] * (BINS + 1) + bin_of[a]).ravel()
    size = rows * (BINS + 1)
    count = np.bincount(flat, minlength=size).reshape(rows, BINS + 1)
    # Each bin's sums are integers below 2^53 (n0 < 2^20 entries below 2^24), so exact in float.
    sum1 = np.bincount(flat, weights=a.ravel().astype(np.float64), minlength=size)
    sum2 = np.bincount(flat, weights=(a * a).ravel().astype(np.float64), minlength=size)
    sum1 = sum1.reshape(rows, BINS + 1).astype(np.int64)
    sum2 = sum2.reshape(rows, BINS + 1).astype(np.int64)
    count[:, 0] = sum1[:, 0] = sum2[:, 0] = 0
    tau = np.array(EDGES[1:], dtype=np.int64)
    below2 = np.cumsum(sum2[:, 1:], axis=1)
    above = count[:, 1:].sum(1, keepdims=True) - np.cumsum(count[:, 1:], axis=1)
    above1 = sum1[:, 1:].sum(1, keepdims=True) - np.cumsum(sum1[:, 1:], axis=1)
    inner = (below2 + 2 * tau * above1 - tau * tau * above).astype(np.float64)
    t = tau.astype(np.float64)
    near = ((t * t * ball2) / box + box * inner) / (2 * t)
    squares = sum2.sum(1)
    by_box = box * sum1.sum(1).astype(np.float64)
    by_ball = np.sqrt(ball2 * squares.astype(np.float64))
    approx = np.floor(np.minimum(near.min(1), np.minimum(by_box, by_ball))).astype(np.int64)
    bounds = [int(v) for v in approx]
    exact = {}

    def make_exact(chosen):
        for row in chosen:
            if row in exact:
                continue
            value, row_squares = row_bound_exact(w[row].tolist(), ball2, box)
            # The floating-point value is within a few units in the last place of the exact one,
            # so their floors differ by at most 1: more is an error in one of the two computations.
            if abs(value - bounds[row]) > 1 or row_squares != int(squares[row]):
                raise SystemExit(f"row {row}: the two computations of its bound disagree")
            bounds[row] = exact[row] = value

    top = max(bounds)
    make_exact([r for r in range(rows) if bounds[r] >= top - 2])
    return bounds, [int(v) for v in squares], exact, make_exact


def frobenius_up(w, scale=None):
    """At or above the GPU's operator_norm(): its first level is the Frobenius norm times
    sqrt(1 + 2^-20) (frobenius()); here the exact sum of squares, rounded up, times 1 + 2^-19."""
    import numpy as np  # noqa: PLC0415

    squares = (w.astype(np.int64) ** 2).sum(1).tolist()  # each row's sum is below 2^53
    total = (
        sum(squares)
        if scale is None
        else sum(s * s * q for s, q in zip(scale, squares, strict=True))
    )
    return sqrt_up(mul_up(ru(Fraction(total)), 1.0 + 2.0**-19))


def main(model_dir, digests_path, out_json):
    import integer_model  # noqa: PLC0415 - from the patched upstream tree, on PYTHONPATH
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    started = time.monotonic()
    digests = json.loads(open(digests_path).read())
    model = integer_model.load_from_folder(model_dir, "cpu", {"sha256": digests})
    p, weights = model.parameters, model.weights
    if not weights.sums_verified:
        raise SystemExit("the weights were not compared with wsums.txt")
    policy = {}
    for text in open(f"{model_dir}/policy.txt").read().splitlines():
        if text:
            policy.setdefault(text.split()[0], text.split()[1:])
    kl = np.frombuffer(open(f"{model_dir}/KL.bin", "rb").read(), dtype="<u4").astype(np.int64)
    if p.NE != 0 or len(kl) != 13 * p.L:
        raise SystemExit("this port covers a parameter file without experts (NE 0) only")
    words = [
        dict(zip(LAYER_WORDS, map(int, kl[13 * i : 13 * i + 13]), strict=True)) for i in range(p.L)
    ]
    if any(words[i][n] != getattr(p.layers[i], n) for i in range(p.L) for n in LAYER_WORDS):
        raise SystemExit("KL.bin's words differ from the integer model's")
    for name in ("smax", "rcl", "dcl", "gcl"):
        if name in policy:
            raise SystemExit(f"this port does not cover the line {name}")
    if "cw" in policy and policy["cw"] != ["12"]:
        raise SystemExit("the check supports only 12-bit cells")
    if (policy["cs"], policy["ut"], policy["prof"]) != (["12"], ["1"], ["20"]):
        raise SystemExit("the check supports only cs 12, ut 1 and prof 20")
    xl = int(policy["xl"][0]) if "xl" in policy else 4
    eg = int(policy["eg"][0]) if "eg" in policy else GATE_KEY_BITS
    if (p.NX, p.eg, p.smax, p.gate_clamp, p.router_clamp) != (xl, eg, 4095, False, False):
        raise SystemExit("the parameter file's xl, eg, smax, gcl or rcl is not as read here")
    eps1, eps2, epsF, MF, KF = p.eps1, p.eps2, p.epsF, p.MF, p.KF
    d, dh, H, G, f, box = p.d, p.dh, p.n_h, p.n_g, p.f, CELL_TOP
    table = gate_most(eg)
    half = div_up(sqrt_up(float(d)), 2.0)
    half_dh = div_up(sqrt_up(float(dh)), 2.0)
    turn = sqrt_up(4096.0 * 4096.0 + 5794.0)

    def matrix(name, layer):
        return weights.matrix(name, layer).cpu().numpy().astype(np.int16)

    # The embedding's longest row (take(), unit 12). The GPU sums the squares rounding upwards;
    # here they are exact, then rounded up with room for its 2,047 upward additions.
    longest = 0
    for first in range(0, p.V, 2000):
        rows = weights.embedding_rows(torch.arange(first, min(p.V, first + 2000))).numpy()
        if int(np.abs(rows).max()) >= 1 << 26:
            raise SystemExit("an embedding entry is too large for exact int64 squares")
        longest = max(longest, int((rows * rows).sum(1).max()))
    X = embedding = sqrt_up(mul_up(ru(Fraction(longest)), 1.0 + 2.0**-30))

    lines, layers, exact_rows = [], [], {}

    def line(site, layer, word, value, bound, limit):
        headroom = math.log2(limit) - math.log2(bound) if bound > 0 else None
        lines.append(
            dict(
                site=site,
                layer=layer,
                word=word,
                value=value,
                bound=bound,
                limit=limit,
                shortfall=shortfall(bound, limit),
                headroom_bits=headroom,
            )
        )

    def site_sum(site, layer, word, k, A):
        line(site, layer, word, k, mul_up(1.0, whole_up(A)), sum_limit(k))

    stream_most = float(WIDE_STREAM_TOP) if xl == 5 else float(math.isqrt((1 << SQUARES_BITS) - 1))
    for i in range(p.L):
        x = words[i]
        B1, B2 = ball(x["M1"], x["K1"], eps1, d, xl), ball(x["M2"], x["K2"], eps2, d, xl)
        ball1, ball2 = math.ceil(mul_up(B1, B1)), math.ceil(mul_up(B2, B2))  # squared_whole()
        y = {}
        for name, site in (("Wq", "q"), ("Wk", "k")):
            w = matrix(name, i)
            b, s, exact, make_exact = row_bounds(w, ball1, box)
            w64 = w.astype(np.int64)
            dot = (w64[0::2] * w64[1::2]).sum(1).tolist()
            # A row not computed exactly may be 1 off its exact bound. Until no such row is left
            # in a pair that, with those rows raised by 1, reaches the largest pair's bound, those
            # rows are computed exactly. Then the largest is exact, and no other pair can exceed it.
            while True:
                top = max(pair_bounds(b, s, dot, B1, x["rp"], turn))
                raised = [v if r in exact else v + 1 for r, v in enumerate(b)]
                close = pair_bounds(raised, s, dot, B1, x["rp"], turn)
                pending = {j for n, v in enumerate(close) if v >= top for j in (2 * n, 2 * n + 1)}
                if pending <= exact.keys():
                    break
                make_exact(sorted(pending - exact.keys()))
            y[site] = top
            exact_rows[f"{name} {i}"] = len(exact)
        w = matrix("Wv", i)
        b, _, exact, _ = row_bounds(w, ball1, box)
        exact_rows[f"Wv {i}"] = len(exact)
        y["v"] = max(b)
        # Delta_o's root (case 1), each row's bound raised by 1 unless exact: at or above the GPU's.
        total = v_norm = 0.0
        for g in range(G):
            norm = frobenius_up(w[g * dh : (g + 1) * dh])
            v_norm = max(v_norm, norm)
            cells = 0.0
            for j in range(g * dh, (g + 1) * dh):
                c = float(min(box, quotient(b[j] + (0 if j in exact else 1), x["kv"])))
                cells = add_up(cells, c * c)
            V = min(sqrt_up(cells), add_up(math.ldexp(mul_up(B1, norm), -x["kv"]), half_dh))
            total = add_up(total, mul_up(add_up(V, half_dh), add_up(V, half_dh)))
        attention_in = sqrt_up(mul_up(float(H // G), total))
        o_norm = frobenius_up(matrix("Wo", i))
        wg = matrix("Wg", i)
        bg, _, exact_g, _ = row_bounds(wg, ball2, box)
        exact_rows[f"Wg {i}"] = len(exact_g)
        y["gate"] = max(bg)
        raised_g = [v if r in exact_g else v + 1 for r, v in enumerate(bg)]
        S = [table[min(-QUOTIENT_LOW, quotient(v, x["kg"]))] for v in raised_g]
        wu = matrix("Wu", i)
        bu, _, exact_u, _ = row_bounds(wu, ball2, box)
        exact_rows[f"Wu {i}"] = len(exact_u)
        y["up"] = max(bu)
        top_s = GATE_TOPS[0]
        gated_up_norm, up_norm = frobenius_up(wu, S), frobenius_up(wu)
        cells = gates = 0.0
        for j in range(f):
            sj = min(S[j], top_s)
            u = min(-QUOTIENT_LOW, quotient(bu[j] + (0 if j in exact_u else 1), x["ku"]))
            h = float(sj * u)
            cells = add_up(cells, mul_up(h, h))
            gates = add_up(gates, float(sj * sj))
        norm = min(gated_up_norm, mul_up(float(top_s), up_norm))
        by_norm = add_up(math.ldexp(mul_up(B2, norm), -x["ku"]), div_up(sqrt_up(gates), 2.0))
        hidden = min(by_norm, sqrt_up(cells))
        down_norm = frobenius_up(matrix("Wd", i))

        q1 = ball(x["M1"], x["K1"], eps1, d, xl, True)
        q2 = ball(x["M2"], x["K2"], eps2, d, xl, True)
        line("stream at norm 1", i, "e_res", 0, X, stream_most)
        line("norm 1 quotient", i, "K1", x["K1"], q1, QUOTIENT_TOP)
        site_sum("q", i, "kq", x["kq"], y["q"])
        site_sum("k", i, "kk", x["kk"], y["k"])
        site_sum("v", i, "kv", x["kv"], y["v"])
        X = add_up(X, add_up(math.ldexp(mul_up(o_norm, attention_in), -x["ko"]), half))
        line("stream at norm 2", i, "e_res", 0, X, stream_most)
        line("norm 2 quotient", i, "K2", x["K2"], q2, QUOTIENT_TOP)
        site_sum("gate", i, "kg", x["kg"], y["gate"])  # no line gcl: the limit is QUOTIENT_TOP
        site_sum("up", i, "ku", x["ku"], y["up"])
        X = add_up(X, add_up(math.ldexp(add_up(mul_up(down_norm, hidden), 0.0), -x["kd"]), half))
        layers.append(
            dict(
                layer=i,
                ball_1=B1,
                ball_2=B2,
                gate_key_bound=quotient(y["gate"], x["kg"]),
                gate_rows_beyond_the_limit=sum(v > sum_limit(x["kg"]) for v in bg),
                largest_S=max(S),
                frobenius=dict(
                    o=o_norm, v_group=v_norm, gated_up=gated_up_norm, up=up_norm, down=down_norm
                ),
                attention_in=attention_in,
                hidden=hidden,
            )
        )
        print(f"layer {i}: gate key bound {quotient(y['gate'], x['kg'])}", flush=True)
    line("stream at the last norm", p.L, "e_res", 0, X, stream_most)
    last_cell = float(WIDE_STREAM_TOP) if xl == 5 else math.ldexp(1.0, LAST_STREAM_BITS) - 1
    line("last stream, a cell", p.L, "e_res", 0, X, last_cell)
    line("last norm quotient", p.L, "KF", KF, ball(MF, KF, epsF, d, xl, True), QUOTIENT_TOP)
    margin = 2.0 * d * box * WEIGHT_MOST + 1 + math.ldexp(1.0, HEAD_LIFT_BITS) + math.ldexp(1.0, 32)
    line("head margin", p.L, "-", 0, margin, math.ldexp(65537.0, HEAD_LOW_BITS + 15) - 1)

    worst = {}
    for row in lines:
        held = worst.get(row["site"])
        if held is None or row["bound"] / row["limit"] > held["bound"] / held["limit"]:
            worst[row["site"]] = row
    stream = [r for r in lines if r["site"].startswith(("stream", "last stream"))]
    result = dict(
        program="research/probes/flashprove/tinyllama/worst_case_cpu.py",
        port_of="prover/weights_commitment/worst_case.cuh at upstream 9e2bdd8",
        xl=xl,
        eg=eg,
        short=sorted([r["site"], r["layer"]] for r in lines if r["shortfall"]),
        row_sites_fit=all(not r["shortfall"] for r in lines if r["site"] in ROW_SITES),
        public_sites_fit=all(not r["shortfall"] for r in lines if r["site"] in PUBLIC_SITES),
        stream_sites_fit_by_frobenius=all(not r["shortfall"] for r in stream),
        worst_by_site=worst,
        rows_recomputed_exactly=exact_rows,
        embedding_longest_row=embedding,
        layers=layers,
        lines=lines,
        seconds=round(time.monotonic() - started),
    )
    open(out_json, "w").write(json.dumps(result, indent=1) + "\n")
    for site, row in worst.items():
        print(
            f"{site:24s} layer {row['layer']:2d} {row['word']:5s} {row['value']:3d}  "
            f"log2 bound {math.log2(row['bound']):7.3f}  limit {math.log2(row['limit'] + 1):7.3f}  "
            f"short {row['shortfall']}"
        )
    print("short:", result["short"])


if __name__ == "__main__":
    main(*sys.argv[1:])
