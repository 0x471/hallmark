"""The FlashProve side of the composed demo on stand-in checks; no verifier is built here."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import composed_input
import flashprove_composed as side

CASE = composed_input.case()
ENROLLED = side.proof.ENROLLED


def checked(**changes):
    return (
        dict(
            patched_verdict="YES",
            unpatched_verdict="NO: policy-tables",
            altered_copies_rejected=8,
            commitment_id=ENROLLED["commitment_id"],
            commitment_id_source="stated by the caller; not recomputed here",
            checked_tokens=CASE["answer_ids"],
            range="24:32",
            files={},
            builds=dict(patched=dict(source_digest=ENROLLED["verifier_source_digest"])),
            seconds=1.0,
        )
        | changes
    )


@pytest.fixture
def stand_ins(monkeypatch):
    calls = []

    def text(output):
        calls.append("text")
        return dict(prompt_tokens=25)

    def proof(case_dir, commitment_id, output):
        calls.append(("proof", commitment_id))
        return checked(**stand_ins_changes)

    stand_ins_changes = {}
    monkeypatch.setattr(side.request_text, "check_text", text)
    monkeypatch.setattr(side.proof, "verify", proof)
    return calls, stand_ins_changes


def test_the_check_runs_text_then_proof_under_the_enrolled_id(tmp_path, stand_ins):
    calls, _ = stand_ins
    output = side.verify(tmp_path / "case", tmp_path / "llm")
    result = json.loads((output / "result.json").read_text())
    assert calls == ["text", ("proof", ENROLLED["commitment_id"])]
    assert result["passed"] is True and result["proofs"] == 1
    assert result["response"] == CASE["response"] and result["checked_tokens"] == CASE["answer_ids"]
    assert result["altered_copies_rejected"] == 8 and result["hiding_claim"] is False


@pytest.mark.parametrize(
    "change",
    [
        dict(patched_verdict="NO: proof-refused"),
        dict(unpatched_verdict="YES"),
        dict(altered_copies_rejected=7),
        dict(commitment_id="0" * 64),
        dict(checked_tokens=CASE["answer_ids"][:-1]),
        dict(builds=dict(patched=dict(source_digest="0" * 64))),
    ],
)
def test_a_check_that_is_not_the_enrolled_one_writes_no_result(tmp_path, stand_ins, change):
    stand_ins[1].update(change)
    with pytest.raises(ValueError, match="incomplete FlashProve verification"):
        side.verify(tmp_path / "case", tmp_path / "llm")
    assert not (tmp_path / "llm/result.json").exists()


def test_the_verifier_identity_follows_every_listed_source():
    inventory = side.source_inventory()
    assert "research/probes/flashprove/tinyllama/enrollment.json" in inventory
    assert "research/probes/flashprove/tinyllama/verifier.patch" in inventory
    assert "scripts/flashprove_composed.py" in inventory and len(inventory) == len(side.SOURCES)
    changed = dict(inventory, **{"scripts/flashprove.py": "0" * 64})
    assert side.verifier_identity(inventory) != side.verifier_identity(changed)
    assert side.verifier_identity(inventory) == side.verifier_identity(dict(inventory))


def test_the_enrollment_names_the_case_the_kit_and_the_patched_verifier():
    assert ENROLLED["case_sha256"] == composed_input.CASE_SHA256
    assert ENROLLED["kit_json_sha256"] == side.sha(side.ft.PROBE / "kit.json")
    assert ENROLLED["verifier_source_digest"] == side.ft.PINS["verifier_source_digest"]
    assert (ENROLLED["h_par"], ENROLLED["configuration"]) == (
        side.proof.KIT["h_par"],
        side.proof.KIT["configuration"],
    )
