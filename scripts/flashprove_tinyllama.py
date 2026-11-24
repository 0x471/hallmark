"""Check Hallmark's TinyLlama patches (dense-model support) on pinned FlashProve, on the CPU.

`make verify-flashprove-tinyllama` fetches the pinned upstream archive (research/probes/flashprove/
source.json), builds the unpatched cp_verify, applies the three pinned patches of
research/probes/flashprove/tinyllama/ and builds the patched one. It then checks:

1. upstream's two stored proofs and 99 altered copies through a stand-in (tinyllama/dual_verify.py)
   that runs both builds on every call: the patched build must accept and reject exactly as the
   unpatched one, with the same output apart from the build digest and times;
2. upstream's standard-form test (verifier/tests/served_form_test.py, with the patch's dense
   cases) passes with the patched build, and with the unpatched build fails exactly the cases that
   only the patch meets;
3. the provers' parameter-file reader, built from the patched source, admits a dense parameter
   file with three table digests and refuses the forms that cp_verify refuses;
4. the fold constants in the four patched implementations agree with each other and with an exact
   derivation;
5. weight_sums, built from the patched source, writes the sums that upstream's exact oracle
   prescribes, at width 5,120 and at width 2,048;
6. a dense parameter file at TinyLlama-1.1B's shape (tinyllama/tinyllama_plan.py, placeholder
   scales) passes the patched build's planner in the standard parameter form, and the unpatched
   build refuses it for its three table digests (policy-tables).

No GPU, weights or paid compute. Nothing here compiles or runs the CUDA prover; its nvcc build is
recorded separately (the probe's README).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import flashprove as fp
from run_e2e import write_json
from run_llm_e2e import require

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research/probes/flashprove/tinyllama"
PINS = json.loads((PROBE / "patches.json").read_text())
UPSTREAM_DIGEST = fp.PIN["verifier_source_digest"]


def run(args, log, cwd=None, env=None, timeout=1800):
    """Run a command, keep its output in `log`, and return (exit code, output)."""
    done = subprocess.run(
        [str(a) for a in args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    log.write_text(done.stdout)
    return done.returncode, done.stdout


def checked_patches(output):
    """Copy each pinned patch into the run folder after checking its size and SHA-256 and the
    upstream revision it is for. Runs before anything from the archive is built or run."""
    require(PINS["upstream_revision"] == fp.PIN["revision"], "the patches are for another revision")
    copies = []
    for name in PINS["apply_order"]:
        pin = PINS["patches"][name]
        data = (PROBE / name).read_bytes()
        require(
            len(data) == pin["bytes"] and hashlib.sha256(data).hexdigest() == pin["sha256"],
            f"{name} changed",
        )
        copy = output / name  # the checked bytes are the applied bytes
        copy.write_bytes(data)
        copies.append(copy)
    return copies


def apply_patches(tree, output, copies):
    """Apply the checked copies in their order; each must apply cleanly."""
    # Only what git needs: no inherited GIT_* variable (GIT_DIR, GIT_CONFIG_PARAMETERS, ..) and no
    # system or user configuration. An enclosing checkout would make git apply skip the source.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(output),
        "LC_ALL": "C",
        "GIT_CEILING_DIRECTORIES": str(tree.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    for copy in copies:
        for flags in (["--check"], []):
            code, _ = run(
                ["git", "apply", *flags, copy], output / f"apply-{copy.name}.log", tree, env, 60
            )
            require(code == 0, f"{copy.name} does not apply to the pinned source")


def check_timed_formats(tree):
    """dual_verify.py must list every wall-clock time (%.3f) that the verifier's sources print. A
    listed format whose text is wrong fails loudly instead: its times stay unmasked, the two builds'
    outputs differ, and dual_verify.py exits 99."""
    sys.path.insert(0, str(PROBE))
    import dual_verify

    text = "".join(
        p.read_text(errors="replace")
        for p in sorted((tree / "verifier/src").iterdir())
        if p.suffix in (".cpp", ".h", ".inc")
    )
    listed = dual_verify.TIMED_FORMATS
    require(
        text.count("%.3f") == sum(f.count("%.3f") for f in listed),
        "cp_verify prints a time that dual_verify.py does not list",
    )
    return len(listed)


def rn32_sqrt(q):
    """RN32(sqrt(q)) for an exact q > 0 whose root is a normal binary32 number, by exact
    comparisons alone, as a binary32 bit pattern."""
    k = q.numerator.bit_length() - q.denominator.bit_length()
    k = k if q >= Fraction(2) ** k else k - 1  # 2^k <= q < 2^(k + 1)
    step = Fraction(2) ** (k // 2 - 23)
    m = q / (step * step)
    r = math.isqrt(m.numerator // m.denominator)
    half = Fraction(2 * r + 1, 2)
    require(m != half * half, "a rounding tie")
    n = r if m < half * half else r + 1  # the root in units of `step`, 2^23 <= n <= 2^24
    exponent = k // 2 + (n >> 24)
    n = n >> (n >> 24)
    return (exponent + 127) << 23 | (n - (1 << 23))


def exact_constants(d, dh):
    """c_0 = RN32(sqrt(d) / 2^t), t the largest integer with 4^t <= d, and RN32(c / sqrt(dh))."""
    t = (d.bit_length() - 1) // 2
    return rn32_sqrt(Fraction(d, 4**t)), rn32_sqrt(Fraction(d, 4**t * dh))


HEX = r"0x[0-9A-Fa-f]{8}"


def constants_in_sources(tree):
    """{implementation: {(d, dh): (c_0, c_0 / sqrt(dh))}} as read from the patched sources."""

    def table(path, pattern):
        text = (tree / path).read_text()
        return {
            (int(d), int(h)): (int(c, 16), int(q, 16)) for d, h, c, q in re.findall(pattern, text)
        }

    found = {
        "prover": table(
            "prover/weights_commitment/fold_weights.cuh",
            rf"if \(d == (\d+) && dh == (\d+)\) return \{{({HEX})u, ({HEX})u\}}",
        ),
        "weight_sums.cpp": table(
            "weight_sums/weight_sums.cpp", rf"\{{(\d+), (\d+), ({HEX})u, ({HEX})u\}}"
        ),
        "weight_sums.py": table(
            "weight_sums/weight_sums.py", rf"\((\d+), (\d+)\): \(({HEX}), ({HEX})\)"
        ),
    }
    model = (tree / "integrations/vllm/integer_model/integer_model/block_weights.py").read_text()
    words = dict(re.findall(rf"^(_C0|_C0_QUERY) = ({HEX})", model, re.M))
    pairs = re.findall(rf"\((\d+), (\d+)\): \((_C0|{HEX}), (_C0_QUERY|{HEX})\)", model)
    found["integer_model"] = {
        (int(d), int(h)): tuple(int(words.get(w, w), 16) for w in (c, q)) for d, h, c, q in pairs
    }
    return found


def check_constants(tree):
    pinned = {
        tuple(int(n) for n in shape.split(",")): tuple(int(w, 16) for w in words)
        for shape, words in PINS["fold_constants"].items()
    }
    for shape, words in pinned.items():
        require(
            exact_constants(*shape) == words, f"the pinned constants of {shape} are not the rule's"
        )
    found = constants_in_sources(tree)
    for where, table in found.items():
        require(table == pinned, f"{where} has other fold constants than the pins: {table}")
    return {where: sorted(f"{d},{h}" for d, h in table) for where, table in found.items()}


def compare_builds(tree, output, verifiers):
    """Upstream's stored proofs and altered copies, each verifier call made by both builds."""
    shim = output / "dual_verify"
    source = (PROBE / "dual_verify.py").read_text()
    shim.write_text(f"#!{sys.executable} -I\n" + source)
    shim.chmod(0o700)
    log = output / "dual.jsonl"
    os.environ.update(
        DUAL_UPSTREAM=str(verifiers["upstream"]),
        DUAL_PATCHED=str(verifiers["patched"]),
        DUAL_UPSTREAM_DIGEST=UPSTREAM_DIGEST,
        DUAL_PATCHED_DIGEST=PINS["verifier_source_digest"],
        DUAL_LOG=str(log),
    )

    def differing():
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return sum(not row["same"] for row in rows)

    checks = {}
    for name, case in fp.PIN["examples"].items():
        folder = tree / "examples" / name
        code, last = fp.direct(shim, folder, case["commitment_id"], output / f"direct-{name}.log")
        require(
            code == 0 and last == "YES",
            f"stored example {name}: exit {code}, last line {last!r}, {differing()} differing calls"
            " so far (exit 99: the builds' outputs differ; see dual.jsonl)",
        )
        ok, row = fp.check_proof(
            tree,
            shim,
            folder,
            case["commitment_id"],
            case["range"],
            case["next_token"],
            output / f"check-{name}.log",
            build=PINS["verifier_source_digest"],
        )
        require(ok, f"stored example {name} was not accepted by the patched build")
        checks[name] = row
    tables = fp.rejection_tables(tree, shim, output)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    differ = [c for c in calls if not c["same"]]
    require(not differ, f"{len(differ)} of {len(calls)} calls differ between the builds")
    # Each example: one direct call and check_proof.py's two; each table: its unaltered set and
    # one call per altered set.
    expected = 3 * len(fp.PIN["examples"]) + sum(n + 1 for *_, n in fp.PIN["rejection_tables"])
    require(len(calls) == expected, f"{len(calls)} calls compared, {expected} expected")
    return dict(
        examples=checks, rejection_tables=tables, calls_compared=len(calls), calls_differing=0
    )


# A case's line in served_form_test.py's output that ends "not as expected": its name.
NOT_AS_EXPECTED = re.compile(r"^([a-z0-9-]+\.[A-Za-z0-9-]+): .*: not as expected$", re.M)


def served_form(tree, output, verifier, name):
    """Upstream's standard-form test, patched, on one build: (exit code, cases not as expected)."""
    args = [
        sys.executable,
        "-I",
        "-B",
        tree / "verifier/tests/served_form_test.py",
        verifier,
        tree / "prover/tests/configurations/pinned",
        output / name,
    ]
    code, text = run(args, output / f"{name}.log", timeout=1800)
    missed = NOT_AS_EXPECTED.findall(text)
    return code, text.rstrip("\n").splitlines()[-1] if text.strip() else "", missed


def compiler():
    return ["g++", "-fno-ipa-modref"] if platform.system() == "Linux" else ["clang++"]


def check_judge(tree, output):
    """The provers' parameter-file reader (prover/common/parameter_file.h, patched) on the dense
    kit that served_form_test.py made, and on forms that cp_verify refuses."""
    judge = output / "parameter_file_judge"
    args = [*compiler(), "-std=c++17", "-O2", "-DBLK_FIELD=2", "-DBLK_HASH=1"]
    args += ["-I", tree / "prover", "-I", tree / "blocks", "-o", judge]
    code, _ = run(
        [*args, tree / "prover/tests/parameter_file_judge.cpp"], output / "judge-build.log"
    )
    require(code == 0, "the parameter-file judge does not build")
    work = output / "served-form-patched"
    expected = {
        tree / "prover/tests/configurations/pinned/policy.txt": "admitted",
        work / "routed-experts.dense/policy.txt": "admitted",
        work / "routed-experts.dense-three-digests/policy.txt": "admitted",
        work / "routed-experts.dense-with-digests/policy.txt": "policy-tables",
        work / "routed-experts.dense-sig-for-rope/policy.txt": "policy-tables",
        work / "table-digests.no-tsig/policy.txt": "policy-tables",
    }
    paths = list(expected)
    done = subprocess.run(
        [judge], input="".join(f"{p}\n" for p in paths), capture_output=True, text=True, timeout=60
    )
    verdicts = done.stdout.splitlines()
    (output / "judge.log").write_text(done.stdout + done.stderr)
    require(done.returncode == 0 and len(verdicts) == len(paths), "the judge did not answer")
    rows = {}
    for path, verdict in zip(paths, verdicts, strict=True):
        name = verdict.split(":")[0]
        require(name == expected[path], f"judge on {path}: {verdict}")
        rows[str(path.relative_to(output if work in path.parents else tree))] = name
    return rows


def check_weight_sums(tree, output):
    """weight_sums against upstream's exact oracle (weight_sums/tests/oracle.py, patched)."""
    program = output / "weight_sums"
    args = [*compiler(), "-std=c++17", "-O2", "-ffp-contract=off", "-pthread", "-o", program]
    code, _ = run([*args, tree / "weight_sums/weight_sums.cpp"], output / "weight_sums-build.log")
    require(code == 0, "weight_sums does not build")
    rows = []
    for shape, seed in PINS["weight_sums_oracle_cases"]:
        folder = output / f"oracle-{shape.replace(',', 'x')}-{seed}"
        env = os.environ | {"WEIGHT_SUMS_ORACLE_SHAPE": shape}
        oracle = [
            sys.executable,
            "-I",
            "-B",
            tree / "weight_sums/tests/oracle.py",
            seed,
            11,
            folder,
        ]
        code, _ = run(oracle, output / f"{folder.name}.log", env=env)
        require(code == 0, f"oracle.py failed at {shape}, seed {seed}")
        shards = sorted(folder.glob("oracle-*.gguf"))
        code, _ = run(
            [program, "--threads", 2, "--out", folder / "sums.txt", *shards],
            output / f"{folder.name}-weight_sums.log",
        )
        same = (
            code == 0
            and (folder / "sums.txt").read_bytes() == (folder / "expected.txt").read_bytes()
        )
        require(same, f"weight_sums differs from the oracle at {shape}, seed {seed}")
        for shard in shards:  # synthetic and large; the sums files stay
            shard.unlink()
        rows.append(dict(shape=shape, seed=seed, same=True))
    return rows


THREE_TABLES = (
    "plan: this program's three public tables match the parameter file's digests "
    "(lines tsilu0 .. texp7, no tsig)"
)
NOT_STANDARD = re.compile(r"^plan: not the standard parameter form", re.M)
NO_POLICY = re.compile(r"^plan: the policy's kind is ", re.M)
UNMET = re.compile(
    r"^plan: not the standard parameter form \((.*)\): no run with these public inputs can be a "
    r"verification \(at best status \d+\)$",
    re.M,
)
DIGEST_LINE = re.compile(r"^(tsilu|tsig|trope|texp)[0-7] ")
PROOF_BYTES = re.compile(r"^length: proof: at most (\d+) bytes", re.M)
BITS = re.compile(r"^accounting: as one number, by the compiled bound: ([0-9.]+) bits", re.M)
# The last lines of a planning call that reached the planner's end (planEndH in upstream's
# verifier/src/cp_verify.cpp): "plan: profile ..", the one-time files' two lines, then only
# "plan: .." lines (among them the standard-form line when a condition is unmet), then these five.
# Other planning paths print "PLAN ONLY" without them, so the standard-form line's absence counts
# only after all of them.
ONE_TIME = (
    "accounting: one-time files: the plan of a key: ",
    "accounting: one-time files: the plan of an entry: ",
)
ENDING = (
    re.compile(r"statement: \d+ words, y at word \d+"),
    re.compile(r"h_par=([0-9a-f]{64})"),
    re.compile(r"configuration=([0-9a-f]{64})"),
    re.compile(r"dw=8"),
    re.compile(r"PLAN ONLY: no proof was checked"),
)


def stdout_of(args, log):
    """(exit code, standard output) of a planning call; standard error goes to its own log."""
    done = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=600)
    log.write_text(done.stdout)
    log.with_suffix(".stderr").write_text(done.stderr)
    return done.returncode, done.stdout


def planner_ending(text, name, unmet=(), three_tables=True):
    """(h_par, configuration) from a planning call's last lines, after checking their order and
    form, that the standard form misses exactly the conditions `unmet` (none: it is met), and
    whether the three tables match their digests."""
    lines = text.splitlines()
    starts = [i for i, x in enumerate(lines) if x.startswith("plan: profile ")]
    require(len(starts) == 1, f"{name}: not one line 'plan: profile'")
    tail = lines[starts[0] + 1 :]
    require(
        len(tail) >= len(ONE_TIME) + len(ENDING), f"{name}: the planner's last lines are missing"
    )
    one_time, tail = tail[: len(ONE_TIME)], tail[len(ONE_TIME) :]
    require(all(map(str.startswith, one_time, ONE_TIME)), f"{name}: no one-time files' lines")
    middle, last = tail[: -len(ENDING)], tail[-len(ENDING) :]
    require(all(x.startswith("plan: ") for x in middle), f"{name}: an unexpected planner line")
    found = [pattern.fullmatch(x) for pattern, x in zip(ENDING, last, strict=True)]
    require(all(found), f"{name}: the planner's last lines are not as expected")
    if unmet:
        missed = UNMET.findall(text)
        once = len(NOT_STANDARD.findall(text)) == 1
        require(once and missed == [", ".join(unmet)], f"{name}: unmet {missed}, not {list(unmet)}")
    else:
        require(not NOT_STANDARD.search(text), f"{name}: not in the standard parameter form")
    require(not NO_POLICY.search(text), f"{name}: every call would end NO: policy")
    require((THREE_TABLES in middle) == three_tables, f"{name}: the three tables' digest line")
    return found[1].group(1), found[2].group(1)


def planning_summary(accounting, request, control):
    """What the planning calls on the TinyLlama-shaped files must show, from their outputs."""
    ids = planner_ending(accounting, "-plan accounting")
    require(planner_ending(request, "-plan request") == ids, "the planning calls name other files")
    proof, bits = PROOF_BYTES.findall(accounting), BITS.findall(accounting)
    require(len(proof) == 1 and len(bits) == 1, "the accounting has no proof length or bits")
    refusals = re.findall(r"^NO: (\S+)$", control, re.M)
    require(refusals == ["policy-tables"], f"the unpatched build says {refusals}")
    require(control.splitlines()[-1:] == ["NO: policy-tables"], "the unpatched build's last line")
    return dict(
        proof_bytes_at_most=int(proof[0]),
        bits_compiled_bound=float(bits[0]),
        h_par=ids[0],
        configuration=ids[1],
    )


def check_tinyllama_plan(tree, output, verifiers):
    """A dense parameter file at TinyLlama-1.1B's shape (tinyllama/tinyllama_plan.py) through the
    planner of both builds: the patched one admits it in the standard parameter form, the
    unpatched one refuses its three table digests. Placeholder scales; no proof is made."""
    folder = output / "tinyllama-plan"
    made = [sys.executable, "-I", "-B", PROBE / "tinyllama_plan.py", tree, folder]
    code, _ = run(made, output / "tinyllama-plan.log")
    require(code == 0, "tinyllama_plan.py failed")
    patched, upstream = verifiers["patched"], verifiers["upstream"]
    code, arrays = stdout_of(
        [patched, "-plan", "arrays", folder, 0], output / "tinyllama-arrays.log"
    )
    require(code == 4, f"-plan arrays: status {code}")
    (folder / "arrays.txt").write_text(arrays)
    policy = (folder / "policy.txt").read_text().splitlines()
    row_log = next(x for x in policy if x.startswith("group link ")).split(" ")[4]
    pack = [sys.executable, "-I", "-B", tree / "prover/tools/pack_layout.py"]
    code, layout = stdout_of(
        [*pack, "--arrays", folder / "arrays.txt", row_log, 4], output / "tinyllama-layout.log"
    )
    require(code == 0, "pack_layout.py failed")
    (folder / "layout.txt").write_text(layout)
    calls = {}
    tokens = 1 << int(next(x for x in policy if x.startswith("lt ")).split(" ")[1])
    (folder / "request.bin").write_bytes(
        struct.pack(f"<{tokens + 2}I", *([7] * tokens + [tokens - 1, tokens]))
    )
    (folder / "y.bin").write_bytes(struct.pack("<I", 5))
    for name, verifier in (("patched", patched), ("upstream", upstream)):
        args = [verifier, "-plan", "request", folder / "policy.txt", folder / "KL.bin", 128]
        args += ["-layout", folder / "layout.txt", "-prompt", folder / "request.bin"]
        args += ["-reply", folder / "y.bin", "-q"]
        calls[name] = stdout_of(args, output / f"tinyllama-request-{name}.log")
    require(calls["patched"][0] == 4 and calls["upstream"][0] == 1, "planning exit codes")
    code, accounting = stdout_of(
        [patched, "-plan", "accounting", folder, 128], output / "tinyllama-accounting.log"
    )
    require(code == 4, f"-plan accounting: status {code}")
    summary = planning_summary(accounting, calls["patched"][1], calls["upstream"][1])
    pin = PINS["tinyllama_plan"]
    values = dict(x.split(" ") for x in policy if len(x.split(" ")) == 2)
    shape = {k: int(values[k]) for k in pin["shape"]}
    require(shape == pin["shape"], f"the generated file's shape is {shape}")
    ids = (summary["h_par"], summary["configuration"])
    require(
        ids == (pin["h_par"], pin["configuration"]), "the generated files are not the pinned ones"
    )
    # The same file without its digest lines: the unpatched build must miss exactly the three
    # conditions that the patch changes at NE 0, and the patched build only the digests.
    bare = output / "tinyllama-plan-no-digests"
    bare.mkdir()
    for name in ("KL.bin", "layout.txt"):
        shutil.copyfile(folder / name, bare / name)
    (bare / "policy.txt").write_text("".join(x + "\n" for x in policy if not DIGEST_LINE.match(x)))
    unmet = {}
    for key, verifier in (("patched", patched), ("unpatched", upstream)):
        log = output / f"tinyllama-no-digests-{key}.log"
        code, text = stdout_of([verifier, "-plan", "accounting", bare, 128], log)
        want = pin["unmet_without_digests"][key]
        require(code == 4, f"{key} build without digests: status {code}")
        planner_ending(text, f"{key} build without digests", unmet=want, three_tables=False)
        unmet[key] = want
    return dict(shape=shape, **summary, unpatched="policy-tables", unmet_without_digests=unmet)


def verify(output):
    started = time.monotonic()
    copies = checked_patches(output)
    tree = fp.source_tree(output)
    verifiers, builds = {}, {}
    verifiers["upstream"], builds["upstream"] = fp.build_verifier(
        tree, output, UPSTREAM_DIGEST, "bin-upstream"
    )
    apply_patches(tree, output, copies)
    timed = check_timed_formats(tree)
    verifiers["patched"], builds["patched"] = fp.build_verifier(
        tree, output, PINS["verifier_source_digest"], "bin-patched"
    )
    result = dict(builds=builds, timed_formats=timed)
    result["upstream_inputs"] = compare_builds(tree, output, verifiers)

    code, last, missed = served_form(tree, output, verifiers["patched"], "served-form-patched")
    require(
        code == 0 and last == "served_form_test.py: all as expected" and not missed, "standard form"
    )
    control = served_form(tree, output, verifiers["upstream"], "served-form-upstream")
    only = PINS["served_form_cases_only_the_patch_meets"]
    footer = f"served_form_test.py: {len(only)} not as expected"
    require(
        control[0] == 1 and control[1] == footer and sorted(control[2]) == sorted(only),
        f"negative control: {control[1]}, {control[2]}",
    )
    result["standard_form"] = dict(patched=last, unpatched_misses=sorted(control[2]))

    result["provers_parameter_file_reader"] = check_judge(tree, output)
    result["fold_constants"] = check_constants(tree)
    result["weight_sums_against_oracle"] = check_weight_sums(tree, output)
    result["tinyllama_plan"] = check_tinyllama_plan(tree, output, verifiers)
    result["seconds"] = time.monotonic() - started
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="flashprove-tinyllama-", dir=ROOT / "runs"))
    print("Outputs: " + str(output), flush=True)
    try:
        require(shutil.which("git"), "git is needed to apply the patches")
        result = verify(output)
        result |= dict(
            passed=True,
            schema_id="hallmark.flashprove-tinyllama-verify/1",
            source=fp.PIN,
            patches=PINS,
            host=dict(system=platform.system(), machine=platform.machine()),
            prover_compiled_by_this_run=False,
            synthetic_inputs_only=True,
            qualified_backend=False,
            hiding_claim=False,
        )
        write_json(output / "result.json", result)
        print(
            json.dumps(
                {k: result[k] for k in ("passed", "seconds")}
                | dict(calls_compared=result["upstream_inputs"]["calls_compared"])
            )
        )
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
