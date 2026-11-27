"""Check a dense TinyLlama proof from a GPU run with caller-held values, and its altered copies.

usage: python scripts/flashprove_tinyllama_proof.py --case DIR [--commitment-id ID]

DIR holds the public files of one run of research/probes/flashprove/tinyllama/gpu_job.sh (its
folder public/): config/policy.txt, KL.bin and layout.txt, ivk/key.bin, ivk/leaf/0.bin,
example.req, example.y and example.proof. ID is the commitment ID that cp_commit printed in that
run: valid only for a commitment the caller made itself, from the public weights, as here. Without
it the enrolled one is taken (research/probes/flashprove/tinyllama/enrollment.json). This
script makes no commitment or proof and uses no GPU. It downloads the pinned archive, builds both
verifiers and requires, in order:

1. the case's configuration is the committed kit, byte for byte, and its request is SASH's
   (kit.json: 25 prompt ids, 8 answer ids, filler to the window, r and e);
2. upstream's integrations/check_proof.py, with the patched cp_verify, accepts the proof under the
   caller's commitment ID, range, end token, exact-match claim and the patched build's digest;
3. a plain call of the patched cp_verify ends YES with status 0;
4. the unpatched cp_verify refuses the same files (NO: policy-tables): only the patched build
   answers YES to a dense proof;
5. each altered copy ends with status 1 and the refusal named for it (REFUSALS) under the patched
   build. Seven are refused by the checks of the proof, the key, the leaf file or the ID: two proof
   bits, a prompt token, an answer token, a bit of the key, a bit of the leaf file, and another
   commitment ID. The eighth, a bit of the token file, is refused on the statement's form
   (statement-range: y must be 0 while e is below T), by a rule that does not depend on the
   proof's content.

A pass says that these files fit the patched verifier's checks for this request under the
commitment ID that the caller states. The ID is not recomputed here: it is cp_commit's own print
on the GPU host, so a pass does not tie the proof to TinyLlama's weights; enrollment.json holds
it as a value. It says nothing of hiding (the seed
is public), of the checkpoint the weights were converted from, or of SASH's integer function
(decision 0094).
"""

import argparse
import json
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import flashprove as fp
import flashprove_tinyllama as ft
from run_e2e import sha, write_json
from run_llm_e2e import require

ROOT, PROBE = ft.ROOT, ft.PROBE
KIT = json.loads((PROBE / "kit.json").read_text())
CASE = json.loads((ROOT / KIT["model"]["case"]).read_text())
ENROLLED = json.loads((PROBE / "enrollment.json").read_text())
CONFIG = ("policy.txt", "KL.bin", "layout.txt")
PUBLIC = (
    *(f"config/{name}" for name in CONFIG),
    "ivk/key.bin",
    "ivk/leaf/0.bin",
    "example.req",
    "example.y",
    "example.proof",
)


def request_bytes():
    """SASH's request as FlashProve takes it: the window of T tokens, then r and e."""
    req = KIT["request"]
    ids = CASE["prompt_ids"] + CASE["answer_ids"]
    require(req["r"] == len(CASE["prompt_ids"]) - 1 and req["e"] == len(ids) - 1, "r and e")
    window = ids + [req["filler"]] * (req["T"] - len(ids))
    return struct.pack(f"<{req['T'] + 2}I", *window, req["r"], req["e"])


def checked_case(case, output):
    """A private copy of the case's public files. What the caller holds itself (the kit's three
    configuration files and the request) is compared on the copy, which the verifiers then read,
    not on the case's folder, which may change after it is read."""
    for name in PUBLIC:
        path = case / name
        require(path.is_file() and not path.is_symlink(), f"{name} is missing from the case")
    copy = output / "case"
    for name in PUBLIC:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case / name, copy / name, follow_symlinks=False)
        require(not (copy / name).is_symlink(), f"{name} is missing from the case")
    for name in CONFIG:
        require(
            sha(copy / "config" / name) == KIT["files"][name],
            f"the case's {name} is not the committed kit's",
        )
    require(
        (copy / "example.req").read_bytes() == request_bytes(), "the case's request is not SASH's"
    )
    return copy


def check_proof(tree, verifier, case, commitment_id, log):
    """Upstream's check_proof.py with the caller's own range, end token and claim, under the
    commitment ID that the caller states."""
    req = KIT["request"]
    checked = f"{req['r']}:{req['e']}"
    args = [sys.executable, "-I", tree / "integrations/check_proof.py"]
    args += [case / "example.proof", case / "example.req", "--verifier", verifier]
    args += ["--public-key-dir", case / "ivk", "--parameter-file", case / "config/policy.txt"]
    args += ["--scales-file", case / "config/KL.bin", "--commitment-id", commitment_id]
    args += ["--range", checked, "--end-token", str(CASE["answer_ids"][-1])]
    args += ["--next-token", "none"]  # e < T: the proof is about the window's tokens alone
    started = time.monotonic()
    done = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=600)
    log.write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    build = ft.PINS["verifier_source_digest"]
    ok = fp.accepted(done.returncode, done.stdout, commitment_id, checked, build)
    return ok, dict(
        exit_code=done.returncode,
        seconds=time.monotonic() - started,
        result=done.stdout.rstrip("\n").splitlines()[-2:],
    )


def flip(path, offset):
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(bytes(data))


def altered_copies(commitment_id):
    """name -> (file to change, change, commitment ID to state)."""
    other = ("0" if commitment_id[0] != "0" else "1") + commitment_id[1:]
    answer = 4 * len(CASE["prompt_ids"])  # the first checked token's low byte
    return {
        "proof-middle-bit": ("example.proof", lambda p: flip(p, p.stat().st_size // 2), None),
        "proof-last-bit": ("example.proof", lambda p: flip(p, p.stat().st_size - 1), None),
        "token-file": ("example.y", lambda p: flip(p, 0), None),
        "request-first-token": ("example.req", lambda p: flip(p, 0), None),
        "request-answer-token": ("example.req", lambda p: flip(p, answer), None),
        "key-bit": ("ivk/key.bin", lambda p: flip(p, p.stat().st_size // 2), None),
        "leaf-bit": ("ivk/leaf/0.bin", lambda p: flip(p, p.stat().st_size // 2), None),
        "commitment-id": (None, None, other),
    }


# The last line that cp_verify must give for each altered copy. statement-range is a refusal on the
# statement's form, by a rule that does not depend on the proof's content: the copy of the token
# file shows that the caller's y is checked, and nothing about the proof.
REFUSALS = {
    "proof-middle-bit": "NO: proof-refused",
    "proof-last-bit": "NO: proof-refused",
    "token-file": "NO: statement-range",
    "request-first-token": "NO: proof-refused",
    "request-answer-token": "NO: proof-refused",
    "key-bit": "NO: key-refused",
    "leaf-bit": "NO: entry-refused",
    "commitment-id": "NO: model-statement",
}


def rejections(verifier, case, commitment_id, output):
    """Each altered copy must end with status 1 and its own refusal (REFUSALS); returns the rows."""
    rows = []
    for name, (file, change, cid) in altered_copies(commitment_id).items():
        copy = output / "altered" / name
        shutil.copytree(case, copy)
        if file:
            change(copy / file)
        code, last = fp.direct(verifier, copy, cid or commitment_id, output / f"altered-{name}.log")
        rows.append(dict(altered=name, exit_code=code, last_line=last, expected=REFUSALS[name]))
        require(code == 1 and last == REFUSALS[name], f"the altered copy {name}: {code} {last}")
    return rows


def built_verifiers(output):
    """The pinned source and both verifiers, built here: (tree, verifiers, builds). The tree is
    left with the three program patches applied."""
    copies = ft.checked_patches(output)
    tree = fp.source_tree(output)
    verifiers, builds = {}, {}
    verifiers["upstream"], builds["upstream"] = fp.build_verifier(
        tree, output, ft.UPSTREAM_DIGEST, "bin-upstream"
    )
    ft.apply_patches(tree, output, copies)
    verifiers["patched"], builds["patched"] = fp.build_verifier(
        tree, output, ft.PINS["verifier_source_digest"], "bin-patched"
    )
    return tree, verifiers, builds


def verify(case, commitment_id, output, built=None):
    """built: built_verifiers()'s return, for a caller that built them before a paid step."""
    require(re.fullmatch("[0-9a-f]{64}", commitment_id), "the commitment ID is 64 hex digits")
    started = time.monotonic()
    # The case is looked at before the minutes of building; a caller that built before a paid
    # step has the case only afterwards.
    held = checked_case(case, output)
    tree, verifiers, builds = built or built_verifiers(output)
    ok, checked = check_proof(
        tree, verifiers["patched"], held, commitment_id, output / "check-proof.log"
    )
    require(ok, f"check_proof.py did not accept the proof: {checked['result']}")
    code, last = fp.direct(verifiers["patched"], held, commitment_id, output / "cp-verify.log")
    require(code == 0 and last == "YES", f"the patched cp_verify: {code} {last}")
    text = (output / "cp-verify.log").read_text()
    require(not ft.NOT_STANDARD.search(text), "not in the standard parameter form")
    code, refused = fp.direct(
        verifiers["upstream"], held, commitment_id, output / "cp-verify-upstream.log"
    )
    require(
        code == 1 and refused == "NO: policy-tables", f"the unpatched cp_verify: {code} {refused}"
    )
    rows = rejections(verifiers["patched"], held, commitment_id, output)
    req = KIT["request"]
    return dict(
        builds=builds,
        commitment_id=commitment_id,
        commitment_id_source="stated by the caller; not recomputed here",
        range=f"{req['r']}:{req['e']}",
        checked_tokens=CASE["answer_ids"],
        files={
            name: dict(bytes=(held / name).stat().st_size, sha256=sha(held / name))
            for name in PUBLIC
        },
        check_proof=checked,
        patched_verdict=last,
        result_line=fp.result_fields(
            (output / "check-proof.log").read_text().split("\n--- stderr")[0]
        ),
        unpatched_verdict=refused,
        altered_copies=rows,
        altered_copies_rejected=len(rows),
        altered_copies_refused_on_the_statement_s_form=sum(
            r["last_line"] == "NO: statement-range" for r in rows
        ),
        seconds=time.monotonic() - started,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--commitment-id", default=ENROLLED["commitment_id"])
    a = p.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="flashprove-tinyllama-proof-", dir=ROOT / "runs"))
    print("Outputs: " + str(output), flush=True)
    try:
        result = verify(a.case.resolve(), a.commitment_id, output)
        result |= dict(
            passed=True,
            schema_id="hallmark.flashprove-tinyllama-proof/1",
            source=fp.PIN,
            patches=ft.PINS["patches"],
            kit=dict(files=KIT["files"], h_par=KIT["h_par"], configuration=KIT["configuration"]),
            host=dict(system=platform.system(), machine=platform.machine()),
            proof_made_here=False,
            gpu_used_here=False,
            hiding_claim=False,
            qualified_backend=False,
        )
        write_json(output / "result.json", result)
        print(
            json.dumps(
                {k: result[k] for k in ("passed", "patched_verdict", "altered_copies_rejected")}
            )
        )
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
