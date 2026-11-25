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
        (lambda c: (c / "config/layout.txt").write_bytes(b"x\n"), "layout.txt"),
        (lambda c: (c / "example.req").write_bytes(b"\0" * 264), "request"),
        (lambda c: (c / "ivk/leaf/0.bin").unlink(), "missing"),
        (lambda c: (c / "example.proof").unlink(), "missing"),
    ],
)
def test_checked_case_refuses_a_worker_s_substitution(case, tmp_path, change, why):
    change(case)
    with pytest.raises(ValueError, match=why):
        proof.checked_case(case, tmp_path)


def test_checked_case_refuses_a_link_in_place_of_a_file(case, tmp_path):
    kept = tmp_path / "kept.proof"
    (case / "example.proof").rename(kept)
    (case / "example.proof").symlink_to(kept)
    with pytest.raises(ValueError, match="missing"):
        proof.checked_case(case, tmp_path)


def test_checked_case_compares_the_copy_that_the_verifiers_read(case, tmp_path, monkeypatch):
    # A folder that changes after it is read: the copy's bytes decide, not the folder's.
    real = proof.shutil.copyfile

    def swapping(src, dst, **kwargs):
        made = real(src, dst, **kwargs)
        if str(dst).endswith("config/policy.txt"):
            (tmp_path / "case/config/policy.txt").write_bytes(b"version H821\n")
        return made

    monkeypatch.setattr(proof.shutil, "copyfile", swapping)
    with pytest.raises(ValueError, match="policy.txt"):
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
    # Each copy has its own expected refusal; only the token file's is a refusal on the
    # statement's form, made before a proof is read.
    assert list(proof.REFUSALS) == list(altered)
    assert [n for n, line in proof.REFUSALS.items() if line == "NO: statement-range"] == [
        "token-file"
    ]


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


def fake_verifier(path, verdicts=None):
    """A stand-in for cp_verify. It answers each altered copy with that copy's own refusal
    (REFUSALS) and status 1, any other folder with YES and status 0; `verdicts` overrides copies.
    The commitment ID's copy is told by the ID that it is called with, not by its folder."""
    answers = {name: (1, line) for name, line in proof.REFUSALS.items()} | (verdicts or {})
    lines = [
        "#!/bin/sh",
        f'[ "$7" = "{ID}" ] || {{ echo "NO: model-statement"; exit 1; }}',
        'case "$1" in',
    ]
    for key, (status, last) in answers.items():
        if key != "commitment-id":
            lines += [f'  */altered/{key}/config) echo "{last}"; exit {status} ;;']
    lines += ['  *) echo "YES"; exit 0 ;;', "esac"]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o700)
    return path


def test_rejections_require_each_copy_s_own_refusal(case, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rows = proof.rejections(fake_verifier(tmp_path / "v"), case, ID, out)
    assert [r["altered"] for r in rows] == list(proof.altered_copies(ID))
    assert all(r["exit_code"] == 1 and r["last_line"] == r["expected"] for r in rows)
    assert [r["last_line"] for r in rows] == list(proof.REFUSALS.values())


def test_the_commitment_id_copy_is_run_under_another_id(case, tmp_path):
    # A verifier that takes every ID answers YES to that copy, whose files are unchanged.
    verifier = tmp_path / "v"
    verifier.write_text(
        fake_verifier(tmp_path / "w").read_text().replace(f'[ "$7" = "{ID}" ]', "true")
    )
    verifier.chmod(0o700)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="commitment-id"):
        proof.rejections(verifier, case, ID, out)


@pytest.mark.parametrize(
    "verdict",
    [
        (0, "YES"),
        (14, "NOT-A-VERIFICATION"),
        (1, "YES"),
        (2, "NO: usage"),
        (1, "NO: statement-range"),  # a refusal, but not the one that a changed key must get
        (1, "NO: proof-refused"),
    ],
)
def test_rejections_fail_if_an_altered_copy_is_not_refused_for_its_reason(case, tmp_path, verdict):
    out = tmp_path / "out"
    out.mkdir()
    verifier = fake_verifier(tmp_path / "v", {"key-bit": verdict})
    with pytest.raises(ValueError, match="key-bit"):
        proof.rejections(verifier, case, ID, out)
