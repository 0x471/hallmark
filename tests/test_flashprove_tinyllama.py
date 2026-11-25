"""Offline checks of the TinyLlama patches' pins and of the caller-side code that tests them.

No archive is fetched and no verifier is built: `make verify-flashprove-tinyllama` does that.
"""

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "research/probes/flashprove/tinyllama")
)
import dual_verify as dual
import flashprove as fp
import flashprove_tinyllama as harness

ID = "ab" * 32
PATCHED_BUILD = harness.PINS["verifier_source_digest"]


def changed_files(patch):
    return re.findall(r"^diff --git a/(\S+) b/\1$", patch.read_text(), re.M)


def test_pins_match_the_patch_files():
    # The fourth patch, scales.patch, is the kit's, pinned by kit.json
    # (tests/test_flashprove_tinyllama_answer.py).
    kit = json.loads((harness.PROBE / "kit.json").read_text())
    on_disk = sorted(p.name for p in harness.PROBE.glob("*.patch"))
    assert sorted([*harness.PINS["apply_order"], kit["scales_patch"]["name"]]) == on_disk
    assert sorted(harness.PINS["apply_order"]) == sorted(harness.PINS["patches"])
    for name, pin in harness.PINS["patches"].items():
        data = (harness.PROBE / name).read_bytes()
        assert len(data) == pin["bytes"]
        assert hashlib.sha256(data).hexdigest() == pin["sha256"]
    assert harness.PINS["upstream_revision"] == fp.PIN["revision"]
    assert PATCHED_BUILD != fp.PIN["verifier_source_digest"]


def test_each_patch_touches_only_its_part_and_marks_every_changed_file():
    parts = {
        "verifier.patch": ("verifier/",),
        "prover.patch": ("prover/",),
        "tools.patch": ("weight_sums/", "integrations/vllm/integer_model/"),
    }
    for name, prefixes in parts.items():
        patch = harness.PROBE / name
        files = changed_files(patch)
        assert files and all(f.startswith(prefixes) for f in files), files
        text = patch.read_text()
        assert text.isascii()
        # Apache-2.0 section 4(b): every changed file says that it was changed.
        for path in files:
            hunk = text.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("\ndiff --git ", 1)[0]
            assert "Modified by the Hallmark project" in hunk, path
    # The prover's and the tools' patches leave cp_verify's source digest alone: none touches the
    # verifier's sources or the headers that flashprove.source_digest() hashes.
    for name in ("prover.patch", "tools.patch"):
        for path in changed_files(harness.PROBE / name):
            assert not path.startswith(("verifier/src/", "blocks/"))
            assert not path.endswith(("plan_rounds.h", "hiding/token_plan.h"))


def bits(x):
    return int.from_bytes(struct.pack(">f", x), "big")


def test_exact_constants_reproduce_upstream_and_the_pins():
    assert harness.exact_constants(5120, 128) == (0x3F8F1BBD, 0x3DCA62C2)
    assert harness.exact_constants(2048, 64) == (0x3FB504F3, 0x3E3504F3)
    for shape, words in harness.PINS["fold_constants"].items():
        d, dh = (int(n) for n in shape.split(","))
        assert harness.exact_constants(d, dh) == tuple(int(w, 16) for w in words)


@pytest.mark.parametrize(
    "q, expected",
    [
        (Fraction(1), 0x3F800000),
        (Fraction(4), 0x40000000),
        (Fraction(1, 4), 0x3F000000),
        (Fraction(2), 0x3FB504F3),
        (Fraction(1, 2), 0x3F3504F3),
        (Fraction(3), bits(3**0.5)),
        (Fraction(10**6 - 1), bits((10**6 - 1) ** 0.5)),
    ],
)
def test_rn32_sqrt(q, expected):
    assert harness.rn32_sqrt(q) == expected


def fake_tree(tmp_path, wide=(0x3FB504F3, 0x3E3504F3)):
    sources = {
        "prover/weights_commitment/fold_weights.cuh": (
            "  if (d == 5120 && dh == 128) return {0x3F8F1BBDu, 0x3DCA62C2u};\n"
            f"  if (d == 2048 && dh == 64) return {{0x{wide[0]:08X}u, 0x{wide[1]:08X}u}};\n"
        ),
        "weight_sums/weight_sums.cpp": (
            "    {5120, 128, 0x3F8F1BBDu, 0x3DCA62C2u},\n"
            f"    {{2048, 64, 0x{wide[0]:08X}u, 0x{wide[1]:08X}u}},\n"
        ),
        "weight_sums/weight_sums.py": (
            "    (5120, 128): (0x3F8F1BBD, 0x3DCA62C2),\n"
            f"    (2048, 64): (0x{wide[0]:08X}, 0x{wide[1]:08X}),\n"
        ),
        "integrations/vllm/integer_model/integer_model/block_weights.py": (
            "_C0 = 0x3F8F1BBD  # x\n_C0_QUERY = 0x3DCA62C2  # x\n"
            "    (5120, 128): (_C0, _C0_QUERY),\n"
            f"    (2048, 64): (0x{wide[0]:08X}, 0x{wide[1]:08X}),\n"
        ),
    }
    for path, text in sources.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    return tmp_path


def test_constants_are_read_from_all_four_implementations(tmp_path):
    found = harness.check_constants(fake_tree(tmp_path))
    assert found == {where: ["2048,64", "5120,128"] for where in found}
    assert len(found) == 4


def test_a_differing_constant_in_any_implementation_fails(tmp_path):
    with pytest.raises(ValueError, match="other fold constants"):
        harness.check_constants(fake_tree(tmp_path, wide=(0x3FB504F2, 0x3E3504F3)))


def test_a_differing_constant_in_one_implementation_alone_fails(tmp_path):
    tree = fake_tree(tmp_path)
    files = [p for p in tree.rglob("*") if p.is_file() and "3FB504F3" in p.read_text().upper()]
    assert len(files) == 4
    for path in files:
        text = path.read_text()
        path.write_text(re.sub("3FB504F3", "3FB504F2", text, count=1, flags=re.I))
        with pytest.raises(ValueError, match="other fold constants"):
            harness.check_constants(tree)
        path.write_text(text)
    harness.check_constants(tree)


def test_the_caller_takes_a_patched_result_only_with_the_patched_build():
    def output(build):
        line = f"outcome=verified commitment_id={ID} range=24:32 claim=exact-match build={build}"
        return f"notes\nresult: {line}\ncheck_proof: verified\n"

    assert fp.accepted(0, output(PATCHED_BUILD), ID, "24:32", PATCHED_BUILD)
    assert not fp.accepted(0, output(fp.PIN["verifier_source_digest"]), ID, "24:32", PATCHED_BUILD)
    # The default stays upstream's digest, as `make verify-flashprove` requires.
    assert not fp.accepted(0, output(PATCHED_BUILD), ID, "24:32")
    assert fp.accepted(0, output(fp.PIN["verifier_source_digest"]), ID, "24:32")


# One line of each timed format, from cp_verify's output on upstream's stored proofs.
TIMED_LINES = [
    "---- stage key ---- 1 thread(s), 0.623 s",
    "ok public inputs: parameter file, shape, no-overflow conditions; profile 4; 2 blocks of"
    " layers  0.004 s",
    "read: sub-proof A_0: 767 claims listed (tested at the chain's end), 4190 slots  0.157 s",
    "read: sub-proof G_0: 8 lookup calls in one forest, 2 more in the smallness test's, 9 matrices,"
    " 803 claims listed (tested at the chain's end), 4059 slots  0.548 s",
    "read: sub-proof H: 6 lookup calls in one forest, 2 more in the smallness test's, 8 matrices,"
    " 297 claims listed (tested at the chain's end), 3570 slots  0.280 s",
    "   seconds: the frame's first() 0.000 | the sub-proof's own checks 0.491 | the frame's later()"
    " 0.000 | the equation's terms made 0.011 | the frame's close() 0.073",
    "seconds: the links' last proximity test (the frame's finish()) 0.104",
    "seconds: before the proof is read (the scan, the plans, blk::configure(), the request's form,"
    " the accounting) 3.129",
    "seconds: the links with their last test 0.433 | behind them (the rows on the table, the place,"
    " the shared chain, the second chain) 0.054",
]


def test_each_timed_format_has_a_real_line():
    assert len(TIMED_LINES) == len(dual.TIMED_FORMATS)
    for line, timed in zip(TIMED_LINES, dual.TIMED, strict=True):
        assert timed.match(line.encode()), line


@pytest.mark.parametrize("line", TIMED_LINES)
def test_times_are_hidden_and_nothing_else(line):
    d = "f" * 64
    times = re.findall(r"\d+\.\d{3}", line)
    other = re.sub(r"\d+\.\d{3}", "9.999", line)
    assert dual.normal(line.encode(), d) == dual.normal(other.encode(), d)
    # Any other number on the line is compared: a thread count, a block or claim count, a slot.
    for number in re.findall(r"(?<![\d.])\d+(?![\d.])", line):
        changed = line.replace(f" {number} ", f" {int(number) + 1} ", 1)
        if changed != line:
            assert dual.normal(line.encode(), d) != dual.normal(changed.encode(), d), number
    assert times


def test_two_runs_of_one_build_compare_equal():
    """The pair that showed the gap in the comparator's earlier form: stage header and public
    inputs line of one binary on one proof, from two runs."""
    d = "f" * 64
    a = "---- stage key ---- 1 thread(s), 0.623 s\nok public inputs: parameter file, shape,"
    a += " no-overflow conditions; profile 4; 2 blocks of layers  0.004 s\n"
    b = a.replace("0.623", "0.981").replace("0.004", "0.016")
    assert dual.normal(a.encode(), d) == dual.normal(b.encode(), d)
    assert dual.normal(a.encode(), d) != dual.normal(a.replace("1 thread", "2 thread").encode(), d)


def test_untimed_lines_are_compared_byte_for_byte():
    d = "f" * 64
    for line in (
        f"result-v1 status=0 bits=128 least-round=128.00 build={d}",
        "plan-round: 128.001105 0 1 0 BASE lambda (Omega, last stage)",
        "note: 2.470 seconds",
        "seconds: elapsed 1.000 | score 0.125",  # not a format of cp_verify's
    ):
        changed = re.sub(r"\d(?!\d*$)", "7", line, count=1) if "build" not in line else line + "x"
        assert dual.normal(line.encode(), d) != dual.normal(changed.encode(), d), line


def fake_verifier(path, text, code):
    path.write_text(f"#!{sys.executable}\nimport sys\nprint({text!r})\nsys.exit({code})\n")
    path.chmod(0o700)
    return path


@pytest.mark.parametrize(
    "patched_text, patched_code, same",
    [
        ("NO: shape-NE build=" + "e" * 64, 1, True),
        ("YES build=" + "e" * 64, 0, False),
        ("NO: shape-NE build=" + "e" * 64, 2, False),
        ("NO: shape-L build=" + "e" * 64, 1, False),
    ],
)
def test_the_stand_in_passes_on_agreement_and_stops_on_difference(
    tmp_path, patched_text, patched_code, same
):
    up = fake_verifier(tmp_path / "up", "NO: shape-NE build=" + "d" * 64, 1)
    down = fake_verifier(tmp_path / "patched", patched_text, patched_code)
    env = os.environ | dict(
        DUAL_UPSTREAM=str(up),
        DUAL_PATCHED=str(down),
        DUAL_UPSTREAM_DIGEST="d" * 64,
        DUAL_PATCHED_DIGEST="e" * 64,
        DUAL_LOG=str(tmp_path / "log.jsonl"),
    )
    script = harness.PROBE / "dual_verify.py"
    done = subprocess.run(
        [sys.executable, script, "a", "b"], env=env, capture_output=True, text=True
    )
    row = json.loads((tmp_path / "log.jsonl").read_text())
    assert row["same"] is same and row["arguments"] == ["a", "b"]
    assert done.returncode == (patched_code if same else 99)
    assert done.stdout.strip() == patched_text


def test_served_form_cases_are_read_from_the_test_output():
    text = "\n".join(
        [
            "routed-experts.dense: NE 0 ...: status 4, not met: (); expected: met: not as expected",
            "table-digests.no-tsig: the lines ...: status 1, ['policy-tables']; as expected",
            "routed-experts.NE-0: NE 0 and nothing else: status 1; expected: x: not as expected",
            "served_form_test.py: 2 not as expected",
        ]
    )
    found = harness.NOT_AS_EXPECTED.findall(text)
    assert found == ["routed-experts.dense", "routed-experts.NE-0"]


# A planning call's output on the TinyLlama-shaped files, cut to the lines that matter here; the
# planner's last lines are the patched build's, as -plan accounting and -plan request print them.
H_PAR = "252ffdf7478c1d69609b5412dd1a540d66aac0916fd29298da52a5cbcc6cb8b9"
CONFIGURATION = "f7fd2f25691d8c4712549b600d541add91c0aa669892d26fcba552dab9b86ccc"
THREE = (
    "plan: this program's three public tables match the parameter file's digests"
    " (lines tsilu0 .. texp7, no tsig)"
)
PLANNED = [
    "accounting: as one number, by the compiled bound: 126.21 bits, proof of work counted"
    " (without it: 96.23 bits): the largest L such that a prover with at most 2^L hash calls,"
    " the verifier's counted in, succeeds with probability at most (its calls) / 2^L",
    "length: proof: at most 14061196 bytes, 13617133 on average over the drawn positions",
    "plan: profile 4: the profile rules out no result",
    "accounting: one-time files: the plan of a key: 53 closeness rounds rest on the theorem on"
    " codes (ePrint 2025/2055, Thm 4.6), taken as given, 0 on the bound within a third of the"
    " distance; the table has a record",
    "accounting: one-time files: the plan of an entry: 23 closeness rounds rest on the theorem on"
    " codes (ePrint 2025/2055, Thm 4.6), taken as given, 0 on the bound within a third of the"
    " distance; the table has a record",
    "plan: lidb 28",
    "plan: nminH 27",
    "plan: a value of the residual stream has five limbs (line xl)",
    THREE,
    "statement: 177 words, y at word 171",
    "h_par=" + H_PAR,
    "configuration=" + CONFIGURATION,
    "dw=8",
    "PLAN ONLY: no proof was checked",
]
NOT_STANDARD = (
    "plan: not the standard parameter form (both-clamps): no run with these public inputs"
    " can be a verification (at best status 14)"
)
REFUSED = "NO: policy-tables\n"
POLICY_KIND = (
    "plan: the policy's kind is X: every whole call under this parameter file ends NO: policy"
)


def text(lines):
    return "".join(x + "\n" for x in lines)


def test_tinyllama_planning_summary():
    got = harness.planning_summary(text(PLANNED), text(PLANNED), REFUSED)
    assert got == dict(
        proof_bytes_at_most=14061196,
        bits_compiled_bound=126.21,
        h_par=H_PAR,
        configuration=CONFIGURATION,
    )


def changed(at, new):
    """PLANNED with the line that starts with `at` replaced by `new` (a list, maybe empty)."""
    (i,) = [i for i, x in enumerate(PLANNED) if x.startswith(at)]
    return PLANNED[:i] + new + PLANNED[i + 1 :]


@pytest.mark.parametrize(
    ("accounting", "planned", "control", "why"),
    [
        # Output that ends like a planning call but lacks the planner's last lines.
        (PLANNED, PLANNED[-2:], [], "plan: profile"),
        (PLANNED, [], [], "plan: profile"),
        (PLANNED[:-1], PLANNED, [], "missing|not as expected"),
        (changed("configuration=", []), PLANNED, [], "not as expected"),
        (changed("accounting: one-time files: the plan of a key", []), PLANNED, [], "one-time"),
        (PLANNED, changed("plan: profile", []), [], "plan: profile"),
        (PLANNED + ["PLAN ONLY: no proof was checked"], PLANNED, [], "unexpected|not as expected"),
        (PLANNED, changed("plan: lidb", ["group: something else"]), [], "unexpected"),
        (PLANNED, changed("plan: profile", PLANNED[2:3] * 2), [], "plan: profile"),
        (changed("dw=", ["dw=4"]), PLANNED, [], "not as expected"),
        # The standard form not met, or every call ending NO: policy, in either call.
        (changed("plan: lidb", [NOT_STANDARD]), PLANNED, [], "standard parameter form"),
        (PLANNED, changed("plan: lidb", [NOT_STANDARD]), [], "standard parameter form"),
        (
            PLANNED,
            changed("plan: lidb", [POLICY_KIND]),
            [],
            "NO: policy",
        ),
        # The tables, the two calls' files, and the accounting's figures.
        (changed(THREE, [THREE.replace("three", "four")]), PLANNED, [], "three tables"),
        (PLANNED, changed("configuration=", ["configuration=" + "0" * 64]), [], "other files"),
        (PLANNED, changed("h_par=", ["h_par=" + "0" * 64]), [], "other files"),
        (changed("length: proof", []), PLANNED, [], "proof length"),
        (changed("accounting: as one number", []), PLANNED, [], "proof length or bits"),
        # The unpatched build must refuse with policy-tables, as its last line.
        (PLANNED, PLANNED, ["NO: plan-layout"], "unpatched"),
        (PLANNED, PLANNED, PLANNED, "unpatched"),
        (PLANNED, PLANNED, ["NO: policy-tables", "NO: policy-tables"], "unpatched"),
        (PLANNED, PLANNED, ["NO: policy-tables", "something after it"], "last line"),
    ],
)
def test_tinyllama_planning_summary_refuses(accounting, planned, control, why):
    refused = text(control) if control else REFUSED
    with pytest.raises(ValueError, match=why):
        harness.planning_summary(text(accounting), text(planned), refused)


def test_the_generator_writes_the_pinned_shape():
    import tinyllama_plan

    shape = tinyllama_plan.SHAPE | {"NE": tinyllama_plan.DENSE["NE"]}
    assert shape == harness.PINS["tinyllama_plan"]["shape"]


def test_the_generator_sets_the_softmax_tables_rows_by_cp_prove_s_rule():
    import tinyllama_plan

    # The upstream kits in prover/examples (head width 128) have kFM 15 and kFD 16; the same rule
    # gives one less of each at TinyLlama's head width 64.
    assert tinyllama_plan.softmax_rows(dict(S=12, rx=8, dh=128)) == dict(kFM=15, kFD=16)
    assert tinyllama_plan.softmax_rows(dict(S=12, rx=8, dh=64)) == dict(kFM=14, kFD=15)


# Upstream's example kits at the pinned commit: the model lines that decide W's rows, the rows
# that upstream's code names (59,392 and 3,076,096: prover/hash_commitment/hf_opening_params.h,
# toy_shape() and full()), and each kit's line `table`.
UPSTREAM_KITS = {
    "target_small_f2": (
        dict(prof=20, L=8, d=512, H=4, G=2, dh=128, f=1024, fe=512, NE=4, V=512, D=2048),
        59392,
        [8, 23, 524288, 65535, 0, 11],
    ),
    "target_small_file_f2": (
        dict(prof=20, L=4, d=5120, H=40, G=8, dh=128, f=2048, fe=1024, NE=16, V=2048, D=2048),
        114688,
        [8, 26, 7864320, 524287, 0, 14],
    ),
    "target_full_f2": (
        dict(prof=20, L=48, d=5120, H=40, G=8, dh=128, f=16384, fe=8192, NE=128, V=202048, D=2048),
        3076096,
        [11, 32, 402653184, 44040191, 0, 18],
    ),
}
TINYLLAMA_TABLE = [11, 24, 1376256, 65535, 0, 18]


@pytest.mark.parametrize("kit", sorted(UPSTREAM_KITS))
def test_the_table_rule_gives_each_upstream_kit_its_own_line(kit):
    import tinyllama_plan

    words, rows, table = UPSTREAM_KITS[kit]
    assert tinyllama_plan.weight_rows(words | {"table": table}) == rows
    assert tinyllama_plan.table_numbers(rows, table[0], table[3], table[5]) == table


def test_the_table_is_sized_for_tinyllama_s_weight_rows():
    import tinyllama_plan

    words = tinyllama_plan.SHAPE | tinyllama_plan.DENSE | dict(prof=20, fe=8192, D=2048)
    full = UPSTREAM_KITS["target_full_f2"][2]
    rows = tinyllama_plan.weight_rows(words | {"table": full})
    assert rows == 10240
    # The full kit's fb and lkW, the small kit's masks.
    small = UPSTREAM_KITS["target_small_f2"][2]
    table = tinyllama_plan.table_numbers(rows, full[0], small[3], full[5])
    assert table == TINYLLAMA_TABLE
    fb, fln, fi0, fnm, _, lkw = table
    data = rows << (lkw - fb)
    # table_derive()'s conditions (prover/hash_commitment/hf_table_params.h), and the least fi0.
    step = max(1 << (lkw - fb), 1 << (fnm - 1).bit_length())
    assert fi0 % step == 0 and fi0 % 2 == 0 and fnm % 2 == 1
    assert fi0 - step < data + 3 <= fi0
    assert fln == (fi0 + fnm - 1).bit_length() + 3 <= 32
    # The full kit's line would hold this model's data in a 307th of a row's entries.
    assert full[2] // data == 307


@pytest.mark.parametrize(
    ("rows", "fb", "fnm", "lkw"),
    [
        (10240, 15, 65535, 18),  # fb above 13 (policy-rows)
        (1000, 11, 65535, 18),  # more long rows than W has rows (policy-rows)
        (10240, 18, 65535, 18),  # fb is not below lkW
        (10240, 11, 65535, 19),  # lkW above 18 (w-shape-lk)
        (10240, 11, 65536, 18),  # an even number of masks (policy-mask-shape)
        (1 << 22, 4, 65535, 18),  # a code longer than 2^32
    ],
)
def test_the_table_rule_refuses_what_the_provers_refuse(rows, fb, fnm, lkw):
    import tinyllama_plan

    with pytest.raises(SystemExit, match="table"):
        tinyllama_plan.table_numbers(rows, fb, fnm, lkw)


# The same TinyLlama-shaped file without its digest lines, as the unpatched build plans it.
UNMET_BARE = (
    "plan: not the standard parameter form (routed-experts, both-clamps, table-digests,"
    " no-range-words): no run with these public inputs can be a verification (at best status 14)"
)
BARE = changed(THREE, [UNMET_BARE])
WITHOUT_DIGESTS = harness.PINS["tinyllama_plan"]["unmet_without_digests"]


def test_planner_ending_names_the_unmet_conditions():
    unmet = WITHOUT_DIGESTS["unpatched"]
    ids = harness.planner_ending(text(BARE), "bare", unmet=unmet, three_tables=False)
    assert ids == (H_PAR, CONFIGURATION)


@pytest.mark.parametrize(
    ("lines", "unmet", "three_tables"),
    [
        (BARE, WITHOUT_DIGESTS["patched"], False),
        (BARE, (), False),
        (BARE, WITHOUT_DIGESTS["unpatched"], True),
        (BARE, ["routed-experts", "table-digests", "both-clamps"], False),
        (
            changed(THREE, [UNMET_BARE, UNMET_BARE]),
            WITHOUT_DIGESTS["unpatched"],
            False,
        ),
        (PLANNED, WITHOUT_DIGESTS["unpatched"], True),
    ],
)
def test_planner_ending_refuses_other_unmet_conditions(lines, unmet, three_tables):
    with pytest.raises(ValueError):
        harness.planner_ending(text(lines), "bare", unmet=unmet, three_tables=three_tables)
