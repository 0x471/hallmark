"""Caller-side checks of a dense TinyLlama proof on fake files; no verifier is built here."""

import shutil
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flashprove_tinyllama_proof as proof

ID = "ab" * 32
KIT_DIR = proof.PROBE / "kit"


@pytest.fixture
def case(tmp_path):
    folder = tmp_path / "public"
    for name in proof.PUBLIC:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x" * 64)
    for name in proof.CONFIG:
        shutil.copyfile(KIT_DIR / name, folder / "config" / name)
    (folder / "example.req").write_bytes(proof.request_bytes())
    return folder


def test_the_request_is_sashs_window_and_range():
    words = struct.unpack("<66I", proof.request_bytes())
    ids = proof.CASE["prompt_ids"] + proof.CASE["answer_ids"]
    assert list(words[: len(ids)]) == ids and set(words[len(ids) : 64]) == {0}
    # r is the last prompt token, e the last answer token: the proof is about tokens r + 1 .. e.
    assert words[64:] == (24, 32)
    assert list(words[25:33]) == proof.CASE["answer_ids"]


def test_checked_case_copies_only_the_callers_own_files(case, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (case / "isk").mkdir()
    (case / "isk/isk.txt").write_text("the provider's")
    copy = proof.checked_case(case, out)
    assert sorted(str(p.relative_to(copy)) for p in copy.rglob("*") if p.is_file()) == sorted(
        proof.PUBLIC
    )


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (lambda c: (c / "config/policy.txt").write_bytes(b"version H821\n"), "policy.txt"),
        (lambda c: (c / "config/KL.bin").write_bytes(b"\0" * 4), "KL.bin"),
        (lambda c: (c / "example.req").write_bytes(b"\0" * 264), "request"),
        (lambda c: (c / "ivk/leaf/0.bin").unlink(), "missing"),
        (lambda c: (c / "example.proof").unlink(), "missing"),
    ],
)
def test_checked_case_refuses_a_worker_s_substitution(case, tmp_path, change, why):
    change(case)
    with pytest.raises(ValueError, match=why):
        proof.checked_case(case, tmp_path)


def test_the_altered_copies_cover_the_six_of_decision_0093_and_two_more():
    altered = proof.altered_copies(ID)
    assert list(altered) == [
        "proof-middle-bit",
        "proof-last-bit",
        "token-file",
        "request-first-token",
        "request-answer-token",
        "key-bit",
        "leaf-bit",
        "commitment-id",
    ]
    other = altered["commitment-id"][2]
    assert other != ID and len(other) == 64 and other[1:] == ID[1:]
    assert proof.altered_copies("0" * 64)["commitment-id"][2][0] == "1"


def test_each_altered_copy_changes_one_bit_of_one_file(case, tmp_path):
    for name, (file, change, _) in proof.altered_copies(ID).items():
        if file is None:
            continue
        copy = tmp_path / name
        shutil.copytree(case, copy)
        change(copy / file)
        differing = [n for n in proof.PUBLIC if (copy / n).read_bytes() != (case / n).read_bytes()]
        assert differing == [file], name
        a, b = (case / file).read_bytes(), (copy / file).read_bytes()
        assert (
            len(a) == len(b) and sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True)) == 1
        )
    # The answer-token copy changes the first checked token, position r + 1.
    copy = tmp_path / "answer"
    shutil.copytree(case, copy)
    proof.altered_copies(ID)["request-answer-token"][1](copy / "example.req")
    words = struct.unpack("<66I", (copy / "example.req").read_bytes())
    assert (
        words[25] == proof.CASE["answer_ids"][0] ^ 1 and words[24] == proof.CASE["prompt_ids"][-1]
    )


def fake_verifier(path, verdicts):
    """A stand-in for cp_verify: its last line and status depend on which copy it is given."""
    lines = ["#!/bin/sh", 'case "$1" in']
    for key, (status, last) in verdicts.items():
        lines += [f'  *{key}*) echo "{last}"; exit {status} ;;']
    lines += ['  *) echo "NO: proof-refused"; exit 1 ;;', "esac"]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o700)
    return path


def test_rejections_require_no_and_status_1_of_every_altered_copy(case, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rows = proof.rejections(fake_verifier(tmp_path / "v", {}), case, ID, out)
    assert [r["altered"] for r in rows] == list(proof.altered_copies(ID))
    assert all(r["exit_code"] == 1 and r["last_line"].startswith("NO:") for r in rows)


@pytest.mark.parametrize(
    "verdict", [(0, "YES"), (14, "NOT-A-VERIFICATION"), (1, "YES"), (2, "NO: usage")]
)
def test_rejections_fail_if_an_altered_copy_is_not_refused(case, tmp_path, verdict):
    out = tmp_path / "out"
    out.mkdir()
    verifier = fake_verifier(tmp_path / "v", {"key-bit": verdict})
    with pytest.raises(ValueError, match="key-bit"):
        proof.rejections(verifier, case, ID, out)
