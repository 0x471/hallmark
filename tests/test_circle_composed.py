"""Real caller bindings with synthetic models/proof bytes; no crypto acceptance."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import circle_backend as circle
import circle_inputs
import composed_input
import trained_common_input as common
from check_circle_composed import binding_negatives, joined_jobs
from classifier_comparison import encoded, sha

pytestmark = pytest.mark.train


def test_circle_composed_exact_statement_and_rejected_bindings(synthetic_exports):
    originals, old_policy = circle_inputs.native_templates(synthetic_exports)
    engines = {
        kind: encoded(
            dict(
                schema="hallmark.circle-enrollment/1",
                external_multiplicities=[1] * 103,
                program_sha256=str(i) * 64,
            )
        )
        for i, kind in enumerate(circle.KINDS)
    }
    enrollment = circle.enroll(engines, originals)
    records, policy, _ = circle.resolve(enrollment, sha(enrollment), originals, old_policy)
    expected, opening = composed_input.request(policy)
    rows = common.evaluate(records, policy, opening, expected)
    bundle = dict(
        schema_id="hallmark.circle-classifier-bundle/1",
        proofs=[
            dict(kind=k, public=r["public"], proof_hex=b"fake".hex())
            for k, r in zip(circle.KINDS, rows, strict=True)
        ],
    )
    assert (
        len(
            joined_jobs(
                enrollment, sha(enrollment), originals, old_policy, opening, expected, bundle
            )
        )
        == 3
    )
    rejected = binding_negatives(
        enrollment, sha(enrollment), originals, old_policy, opening, expected, bundle
    )
    assert len(rejected) == 16 and len(set(rejected)) == 16
    other_expected, other_opening = composed_input.request(policy)
    with pytest.raises(ValueError, match="binding mismatch"):
        joined_jobs(
            enrollment,
            sha(enrollment),
            originals,
            old_policy,
            other_opening,
            other_expected,
            bundle,
        )
    changed = copy.deepcopy(bundle)
    changed["proofs"][0]["public"][98] += 1
    with pytest.raises(ValueError, match="binding mismatch"):
        joined_jobs(enrollment, sha(enrollment), originals, old_policy, opening, expected, changed)


def test_circle_composed_binds_the_flashprove_model_identity(synthetic_exports):
    originals, old_policy = circle_inputs.native_templates(synthetic_exports)
    engines = {
        kind: encoded(
            dict(
                schema="hallmark.circle-enrollment/1",
                external_multiplicities=[1] * 103,
                program_sha256=str(i) * 64,
            )
        )
        for i, kind in enumerate(circle.KINDS)
    }
    enrollment = circle.enroll(engines, originals)
    pin = sha(enrollment)
    records, policy, _ = circle.resolve(enrollment, pin, originals, old_policy)
    case = composed_input.case()
    assert composed_input.model_identity(case) != composed_input.model_identity(case, "flashprove")
    with pytest.raises(ValueError, match="unknown LLM backend"):
        composed_input.model_identity(case, "other")
    bundles = {}
    for backend in composed_input.BACKENDS:
        expected, opening = composed_input.request(policy, backend)
        rows = common.evaluate(records, policy, opening, expected)
        proofs = [
            dict(kind=k, public=r["public"], proof_hex=b"fake".hex())
            for k, r in zip(circle.KINDS, rows, strict=True)
        ]
        bundles[backend] = (
            opening,
            expected,
            dict(schema_id="hallmark.circle-classifier-bundle/1", proofs=proofs),
        )
    opening, expected, bundle = bundles["flashprove"]
    args = (enrollment, pin, originals, old_policy, opening, expected, bundle)
    assert len(joined_jobs(*args, "flashprove")) == 3
    rejected = binding_negatives(*args, "flashprove")
    assert len(rejected) == 16 and len(set(rejected)) == 16
    # Classifier proofs that bind one path's model do not join the other path's LLM proof.
    with pytest.raises(ValueError, match="request mismatch"):
        joined_jobs(*args)
    opening, expected, bundle = bundles["sash"]
    with pytest.raises(ValueError, match="request mismatch"):
        joined_jobs(enrollment, pin, originals, old_policy, opening, expected, bundle, "flashprove")
