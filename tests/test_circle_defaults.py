"""Native enrollment parity and default dispatch; synthetic fixtures make no proof claim."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import circle_backend as backend
import circle_inputs
import classifier_comparison as comparison
import run_circle_e2e

pytestmark = pytest.mark.train


def test_native_templates_carry_no_keys_and_enroll(synthetic_exports):
    native, policy = circle_inputs.native_templates(synthetic_exports)
    assert all("key" not in n for n in native)
    engines = {
        kind: comparison.encoded(
            dict(
                schema="hallmark.circle-enrollment/1",
                external_multiplicities=[1] * 103,
                program_sha256=str(i) * 64,
            )
        )
        for i, kind in enumerate(backend.KINDS)
    }
    raw = backend.enroll(engines, native)
    records, _, _ = backend.resolve(raw, comparison.sha(raw), native, policy)
    assert [r["classifier"].backend for r in records] == [backend.source_identity()[0]] * 3
    assert policy.max_prompt_bytes == 96


def test_native_preparation_rejects_changed_request_and_pin(
    synthetic_exports, tmp_path, monkeypatch
):
    exports = synthetic_exports
    monkeypatch.setattr(comparison, "exports", lambda _: copy.deepcopy(exports))
    monkeypatch.setattr(comparison, "EXPORTS", {})
    pin = circle_inputs.create(tmp_path)
    circle_inputs.admit(tmp_path, pin)
    with pytest.raises(ValueError, match="pin mismatch"):
        circle_inputs.admit(tmp_path, "0" * 64)
    path = tmp_path / "requests/allow-request.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="request mismatch"):
        circle_inputs.admit(tmp_path, pin)


def test_circle_unsupported_host_stops_before_output(monkeypatch):
    monkeypatch.setattr(run_circle_e2e.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="Apple Silicon"):
        run_circle_e2e.run()


@pytest.mark.parametrize(
    "target,driver", [("e2e", "run_circle_e2e.py"), ("e2e-composed", "run_circle_composed.py")]
)
def test_make_targets_run_circle(target, driver):
    root = Path(__file__).resolve().parents[1]
    command = ["make", "--no-print-directory", "-n", target]
    assert driver in subprocess.check_output(command, cwd=root, text=True)


@pytest.mark.parametrize("bad_authority", [True, False])
def test_composition_rejects_authority_or_incomplete_llm_before_classifier_proving(
    tmp_path, monkeypatch, bad_authority
):
    from types import SimpleNamespace

    import run_circle_composed as composed

    monkeypatch.setattr(composed, "ROOT", tmp_path)
    monkeypatch.setattr(
        composed.shutil, "disk_usage", lambda _: SimpleNamespace(free=128 * 1024**3)
    )
    monkeypatch.setattr(composed.verify_llm, "read_enrollment", lambda *a: (None, None))
    monkeypatch.setattr(composed.verify_llm, "source_inventory", lambda: {})
    monkeypatch.setattr(composed.verify_llm, "verifier_identity", lambda _: bytes(32))
    monkeypatch.setattr(composed.run_e2e, "source_pins", lambda: {})
    invoked = []

    def authority(*args, **kwargs):
        if bad_authority:
            raise ValueError("invalid authority")
        return {"kind": "checkout-case"}

    monkeypatch.setattr(composed.verify_llm, "parameter_authority", authority)
    verification = tmp_path / "native"
    verification.mkdir()
    (verification / "result.json").write_text('{"passed": true, "proofs": 22}')
    monkeypatch.setattr(composed.verify_llm, "run", lambda *a, **k: verification)
    monkeypatch.setattr(composed.run_circle_e2e, "run", lambda *a, **k: invoked.append("prove"))
    with pytest.raises(ValueError, match="invalid authority|incomplete fresh LLM"):
        composed.run(llm_run=tmp_path)
    assert invoked == []
    assert len(list((tmp_path / "runs").glob("composed-circle-*/failure.json"))) == 1


def test_flashprove_composition_stops_on_an_incomplete_check_before_classifier_proving(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import flashprove_composed
    import run_circle_composed as composed

    monkeypatch.setattr(composed, "ROOT", tmp_path)
    monkeypatch.setattr(
        composed.shutil, "disk_usage", lambda _: SimpleNamespace(free=128 * 1024**3)
    )
    monkeypatch.setattr(composed.verify_llm, "read_enrollment", lambda *a: (None, None))
    monkeypatch.setattr(composed.run_e2e, "source_pins", lambda: {})
    invoked = []

    def checked(case_dir, output):
        # Outside the composed evidence folder: the join's output budget counts all of that one.
        assert output.parent.parent == tmp_path / "runs"
        assert output.parent.name.startswith("flashprove-composed-")
        output.mkdir()
        record = dict(passed=True, proofs=1, fresh_verifications=1, altered_copies_rejected=7)
        record["response"] = composed.composed_input.case()["response"]  # all but the count holds
        (output / "result.json").write_text(json.dumps(record))
        return output

    monkeypatch.setattr(flashprove_composed, "verify", checked)
    monkeypatch.setattr(composed.run_circle_e2e, "run", lambda *a, **k: invoked.append("prove"))
    with pytest.raises(ValueError, match="incomplete fresh LLM"):
        composed.run(llm_backend="flashprove", flashprove_case=tmp_path)
    assert invoked == []
    (failure,) = (tmp_path / "runs").glob("composed-circle-*/failure.json")
    protocol = json.loads((failure.parent / "protocol.json").read_text())
    assert protocol["llm_backend"] == "flashprove" and protocol["fresh_llm_proving"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        dict(llm_backend="flashprove"),
        dict(flashprove_case=Path("x")),
        dict(llm_backend="flashprove", flashprove_case=Path("x"), llm_run=Path("y")),
        dict(llm_backend="other"),
    ],
)
def test_composition_refuses_a_mixed_choice_of_llm_backend(arguments):
    import run_circle_composed as composed

    with pytest.raises(ValueError, match="FlashProve backend takes|unknown LLM backend"):
        composed.run(**arguments)
