"""Offline checks of worst_case_cpu.py's arithmetic: the CPU port of cp_commit's worst-case check.

No model runs: `make verify-flashprove-tinyllama-answer` runs the port on the kit. These tests use
only the port's pure-Python parts, against values derived by hand.
"""

import math
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "research/probes/flashprove/tinyllama")
)
import worst_case_cpu as wc


def test_the_tau_grid_is_upstreams_rule():
    edges = wc.EDGES
    assert edges[:68] == list(range(68))  # every integer up to 67
    assert edges[-1] == wc.WEIGHT_MOST and wc.BINS == len(edges) - 1 <= wc.BINS_MOST
    for a, b in zip(edges, edges[1:], strict=False):
        assert b == min(wc.WEIGHT_MOST, max(a + 1, a * 103 // 100))
    for v in range(1, wc.WEIGHT_MOST + 1):  # bin b holds edge[b - 1] < |w| <= edge[b]
        b = wc.BIN_OF[v]
        assert edges[b - 1] < v <= edges[b]


def test_rounding_upwards_gives_the_least_double_at_or_above():
    for q in (Fraction(1, 3), Fraction(2**60 + 1), Fraction(-7, 10), Fraction(5, 1)):
        f = wc.ru(q)
        assert Fraction(f) >= q and Fraction(math.nextafter(f, -math.inf)) < q
    for a in (2.0, 3.0, 1e17 + 7, 0.5):
        f = wc.dsqrt_ru(a)
        assert Fraction(f) ** 2 >= Fraction(a)
        assert Fraction(math.nextafter(f, -math.inf)) ** 2 < Fraction(a)
    assert wc.dsqrt_ru(4.0) == 2.0 and wc.ru(Fraction(5)) == 5.0


def test_a_rows_bound_is_the_least_of_the_ball_the_box_and_the_dual():
    # The ball binds: max (3, 4).a over ||a|| <= 5 is 25, and the box (100) does not bind.
    assert wc.row_bound_exact([3, 4], 25, 100) == (25, 25)
    # The box binds: |a_k| <= 1 and ||a||^2 <= 100 give 4 for four weights 1.
    assert wc.row_bound_exact([1, -1, 1, 1], 100, 1) == (4, 4)
    # Both: |a_k| <= 1, ||a||^2 <= 4; a = (1, 1) gives 6, the box alone also 6.
    assert wc.row_bound_exact([5, 1], 4, 1) == (6, 26)
    assert wc.row_bound_exact([0, 0, 0], 9, 2) == (0, 0)


def test_a_rows_bound_holds_for_points_of_the_ball_and_the_box():
    rng = random.Random(186)
    for _ in range(40):
        n = rng.randint(1, 12)
        row = [rng.randint(-wc.WEIGHT_MOST, wc.WEIGHT_MOST) for _ in range(n)]
        ball2, box = rng.randint(1, 5000), rng.randint(1, 60)
        bound, squares = wc.row_bound_exact(row, ball2, box)
        assert squares == sum(w * w for w in row)
        assert bound <= min(box * sum(abs(w) for w in row), math.isqrt(ball2 * squares) + 1)
        for _ in range(50):  # integer points a with |a_k| <= box and ||a||^2 <= ball2
            a = [rng.randint(-box, box) for _ in range(n)]
            while sum(x * x for x in a) > ball2:
                a = [x // 2 for x in a]
            assert abs(sum(w * x for w, x in zip(row, a, strict=True))) <= bound


def test_the_ball_and_the_limits():
    # M 43, K 32, eps 85,899 (isqrt 293), d 2,048, five limbs: the kit's norms.
    exact = 2048 * (1 + 1 / 293) + (2**39 + 295) / 2**33 + math.sqrt(2048) / 2
    radius = wc.ball(43, 32, 85899, 2048, 5)
    assert exact <= radius <= exact * (1 + 1e-14)
    assert wc.ball(43, 32, 85899, 2048, 5, one_cell=True) < radius
    assert wc.sum_limit(10) == 65279 * 512 - 1  # a gate's key round(A / 2^10) <= 32,639
    assert wc.quotient(65279 * 512 - 1, 10) == 32639 and wc.quotient(65279 * 512, 10) == 32640
    assert wc.shortfall(1.0, 1.0) == 0 and wc.shortfall(2.0, 1.0) == 1
    assert wc.shortfall(float("nan"), 1.0) == 99


def test_the_gate_table_bound_rises_to_smax():
    table = wc.gate_most()
    assert len(table) == 32769 and table[0] == 1  # silu(0) = 0, plus one
    assert all(a <= b for a, b in zip(table, table[1:], strict=False))
    assert table[-1] == 4095  # 64 silu(64) is 4,096, clamped to smax


def test_pairs_without_rope_take_the_larger_row():
    assert wc.pair_bounds([3, 9, 4, 2], [0] * 4, [0, 0], 1.0, 0, 4096.0) == [4096 * 9, 4096 * 4]
