"""The FlashProve side of the composed demo: a fresh check of one dense TinyLlama proof.

scripts/run_circle_composed.py calls this in place of SASH's native verification when it is given
--llm-backend flashprove. Nothing is proved here and no GPU is used: the folder holds the public
files of a run of scripts/run_flashprove_tinyllama_gpu.py (or the published archive), and every
check of scripts/flashprove_tinyllama_proof.py runs again on them under the enrolled commitment
ID, after the pinned tokenizer and chat template have tied the request's ids to the case's text.
A saved result never stands in for these checks.

What the composition can then say is narrower than with SASH: one FlashProve proof that the
enrolled commitment's integer model, in a 64-token window, chooses the case's eight answer tokens
after its 25 prompt tokens. Only Hallmark's patched cp_verify accepts such a proof (decision
0094), and the commitment ID is an enrolled value that no CPU program here recomputes.
"""

from pathlib import Path

import composed_input
import flashprove as fp
import flashprove_tinyllama as ft
import flashprove_tinyllama_proof as proof
import flashprove_tinyllama_request as request_text
from run_e2e import sha, write_json
from run_llm_e2e import require

from hallmark.wire import H

ROOT = fp.ROOT
SOURCES = (
    *(
        ROOT / "scripts" / name
        for name in (
            "flashprove.py",
            "flashprove_tinyllama.py",
            "flashprove_tinyllama_answer.py",
            "flashprove_tinyllama_proof.py",
            "flashprove_tinyllama_request.py",
            "flashprove_composed.py",
            "composed_input.py",
        )
    ),
    fp.PROBE / "source.json",
    *(
        ft.PROBE / name
        for name in (
            "patches.json",
            "verifier.patch",
            "prover.patch",
            "tools.patch",
            "kit.json",
            "enrollment.json",
            "request_text.py",
        )
    ),
    ROOT / "research/probes/sash/response-case.json",
)


def source_inventory():
    return {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}


def verifier_identity(inventory):
    return H(
        "program", dict(schema_id="hallmark/flashprove-llm-verifier-source/1", files=inventory)
    )


def verify(case_dir, output):
    """Text, then proof, both afresh; writes result.json in `output` and returns the folder."""
    case = composed_input.case()
    enrolled = proof.ENROLLED
    require(enrolled["case_sha256"] == composed_input.CASE_SHA256, "enrolled for another case")
    output.mkdir()
    text = request_text.check_text(output)
    checked = proof.verify(Path(case_dir), enrolled["commitment_id"], output)
    require(
        checked["patched_verdict"] == "YES"
        and checked["unpatched_verdict"] == "NO: policy-tables"
        and checked["altered_copies_rejected"] == 8
        and checked["commitment_id"] == enrolled["commitment_id"]
        and checked["checked_tokens"] == case["answer_ids"]
        and checked["builds"]["patched"]["source_digest"] == enrolled["verifier_source_digest"],
        "incomplete FlashProve verification",
    )
    result = dict(
        schema_id="hallmark.flashprove-llm-result/1",
        passed=True,
        response=case["response"],
        proofs=1,
        fresh_verifications=1,
        altered_copies_rejected=checked["altered_copies_rejected"],
        commitment_id=checked["commitment_id"],
        commitment_id_source=checked["commitment_id_source"],
        range=checked["range"],
        checked_tokens=checked["checked_tokens"],
        request_text=text,
        files=checked["files"],
        builds=checked["builds"],
        unpatched_verdict=checked["unpatched_verdict"],
        seconds=checked["seconds"],
        cloud_cost_usd=0,
        qualified_backend=False,
        security_review=False,
        hiding_claim=False,
    )
    write_json(output / "result.json", result)
    return output
