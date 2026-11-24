"""Rebuild the TinyLlama-1.1B kit for pinned FlashProve and run the decisive CPU check (0094).

`make verify-flashprove-tinyllama-answer` runs on a CPU, with no GPU and no spend. It downloads the
pinned upstream archive, TinyLlama-1.1B-Chat-v1.0's files at the revision of Hallmark's SASH path
(about 2.2 GB) and WikiText-2's training split, makes two Python environments from the hash-locked
requirement files of requirements/ (its own platform's; only macOS arm64 and Linux x86_64 have
them), applies the three pinned patches and scales.patch, and then requires, in order:

1. the export: export_gguf.py writes the GGUF file with the pinned SHA-256;
2. the kit, rebuilt by upstream's own route (scales/README.md, "A real model", steps 1 to 5) with
   the patched tools: weight_sums, make_calibration_set.py --tinyllama, calibrate.py, then its
   fit again at kit.json's eg (fit_eg.py, which at eg 8 must give calibrate.py's words.json byte
   for byte), words_to_policy.py from check 6's dense base file, -plan arrays, pack_layout.py.
   Every kit file but maxima.json must equal the committed kit (tinyllama/kit/) byte for byte;
   maxima.json, whose last digits depend on the device and libraries, is compared and recorded
   only;
3. the patched verifier's planner admits the kit in the standard parameter form, for the whole kit
   and for SASH's request, with the pinned h_par and configuration; the unpatched one refuses it
   (policy-tables); the provers' parameter-file reader admits it; and the part of cp_commit's and
   cp_prove's set-up that needs only the kit, run on a CPU with upstream's code
   (tinyllama/prover_setup.cpp: the weight table's parameters for W's rows, the plan of attention
   of each layer group and each group's layer layout), admits it, while the same kit with kFM one
   higher is refused by the plan of attention and with one step less of the table's message
   entries by the table's own check; the kit's line table must be the one that
   tinyllama_plan.table_numbers() gives for the rows that upstream's layoutW() counts;
4. upstream's scale-tool tests pass on the patched tree, run as SUITE.md runs its two scales rows
   (the other two tests have no row); status 77 means passed in part: this target builds no
   make_model and sets no SMALL_MODEL, so those cases skip;
5. the float reference: upstream's float pass on the GGUF agrees with Hugging Face transformers on
   the safetensors, and float greedy decoding gives SASH's answer;
6. the decisive check: upstream's integer model (patched), on the GGUF and the committed kit, with
   every file pinned and the weights checked against wsums.txt, gives SASH's eight answer tokens
   after SASH's 25 prompt tokens, in one window of 64 tokens with no failed range condition, and
   its logits follow the float logits (slope and correlation bounds in kit.json); two mutation
   controls (MF + 1, and one layer's kv + 1) must each fail at least one of these conditions;
7. cp_commit's worst-case check, computed on a CPU: worst_case_cpu.py (Hallmark's port of
   upstream's worst_case.cuh) on the kit and the GGUF file, and gate_float_bound.py on the float
   weights. The sites that do not fit must be kit.json's worst_case.short (none: at eg 7 every site
   fits), and the two must agree on the gate: the same layers over the limit, and in each layer the
   integer key bound must be the float one times the norm's ball radius over 2,048, within
   kit.json's tolerance.

No commitment, proof or GPU run is made. Equal tokens show that the integer model agrees with
SASH's answer on this request; they do not show that FlashProve's and SASH's integer functions are
equal.
Set FLASHPROVE_TINYLLAMA_INPUTS to a folder holding the pinned model files and parquet file to skip
the downloads; each file is checked against its pin before it is read.
"""

import argparse
import hashlib
import http.client
import json
import os
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
from run_e2e import write_json
from run_llm_e2e import fetch, require

ROOT, PROBE = ft.ROOT, ft.PROBE
sys.path.insert(0, str(PROBE))
import tinyllama_plan  # noqa: E402

KIT_DIR = PROBE / "kit"
KIT = json.loads((PROBE / "kit.json").read_text())
CASE_PATH = ROOT / KIT["model"]["case"]
CASE = json.loads(CASE_PATH.read_text())
KIT_FILES = ("policy.txt", "KL.bin", "layout.txt")  # the configuration directory
LAYER_WORDS = ("kq", "kk", "kv", "ko", "kg", "ku", "kd", "M1", "K1", "M2", "K2", "sh", "rp")
SETUP_FILES = ("wsums.txt", "fold.txt")  # beside the weights
FIT_FILES = ("words.json", "maxima.json")
CPU_TORCH = "https://download.pytorch.org/whl/cpu"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, log, cwd=None, env=None, timeout=3600):
    return ft.run(args, log, cwd=cwd, env=env, timeout=timeout)


def pinned_copies(output):
    """The three pinned patches (checked by flashprove_tinyllama) and scales.patch, checked here."""
    require(KIT["upstream_revision"] == fp.PIN["revision"], "the kit is for another revision")
    require(
        sha(PROBE / "patches.json") == KIT["patches_json_sha256"], "the kit is for other patches"
    )
    copies = ft.checked_patches(output)
    pin = KIT["scales_patch"]
    data = (PROBE / pin["name"]).read_bytes()
    require(
        len(data) == pin["bytes"] and hashlib.sha256(data).hexdigest() == pin["sha256"],
        "scales.patch changed",
    )
    copy = output / pin["name"]
    copy.write_bytes(data)
    return [*copies, copy]


def committed_kit():
    """The committed kit's files, each checked against kit.json."""
    for name, digest in KIT["files"].items():
        require(sha(KIT_DIR / name) == digest, f"kit/{name} does not match kit.json")
    return {name: (KIT_DIR / name).read_bytes() for name in KIT["files"]}


def platform_key():
    """The lock files' platform name for this host; other hosts are refused."""
    key = {("Darwin", "arm64"): "macos-arm64", ("Linux", "x86_64"): "linux-x86_64"}.get(
        (platform.system(), platform.machine())
    )
    require(key is not None, f"no lock file for {platform.system()} {platform.machine()}")
    return key


def environment(output, name):
    """A new virtual environment, made by uv from the pinned, hash-locked requirement file of
    this platform (requirements/<name>-<platform>.txt, its SHA-256 in kit.json), installed with
    --require-hashes. Returns its interpreter and what it holds (uv pip freeze)."""
    spec = KIT["environments"][name]
    lock = spec["locks"][platform_key()]
    path = PROBE / "requirements" / lock["file"]
    require(sha(path) == lock["sha256"], f"{lock['file']} does not match kit.json")
    venv = output / f"venv-{name}"
    code, _ = run(["uv", "venv", "-q", "-p", spec["python"], venv], output / f"venv-{name}.log")
    require(code == 0, f"uv cannot make the {name} environment")
    python = venv / "bin/python"
    args = ["uv", "pip", "install", "-q", "-p", python, "--require-hashes", "--no-deps", "-r", path]
    if platform.system() == "Linux":  # torch's CPU build; PyPI's Linux wheels pull CUDA libraries
        args += ["--index-url", CPU_TORCH, "--extra-index-url", "https://pypi.org/simple"]
        args += ["--index-strategy", "unsafe-best-match"]
    code, _ = run(args, output / f"venv-{name}-install.log")
    require(code == 0, f"cannot install the {name} lock file")
    code, text = run(
        [python, "-I", "-c", VERSIONS, *spec["packages"]], output / f"venv-{name}-versions.log"
    )
    require(code == 0, f"the {name} environment does not import")
    versions = json.loads(text)
    for package, wanted in spec["packages"].items():
        require(
            versions[package].split("+")[0] == wanted, f"{name}: {package} is {versions[package]}"
        )
    code, frozen = run(["uv", "pip", "freeze", "-p", python], output / f"venv-{name}-freeze.log")
    require(code == 0, f"uv pip freeze failed ({name})")
    installed = sorted(
        re.sub(r"[-_.]+", "-", x.split("==")[0]).lower() + "==" + x.split("==")[1]
        for x in (y.strip() for y in frozen.splitlines())
        if "==" in x
    )
    require(installed == locked(path), f"the {name} environment is not exactly its lock file")
    versions["lock_file"] = lock["file"]
    versions["installed"] = installed
    return python, versions


def locked(path):
    """The name==version lines of a lock file, normalised as uv pip freeze prints them."""
    lines = Path(path).read_text().splitlines()
    pins = [x.split(" ")[0] for x in lines if re.match(r"[A-Za-z]", x)]
    return sorted(
        re.sub(r"[-_.]+", "-", x.split("==")[0]).lower() + "==" + x.split("==")[1] for x in pins
    )


VERSIONS = (
    "import importlib.metadata as m, json, sys; "
    "print(json.dumps({p: m.version(p) for p in sys.argv[1:]}"
    " | {'python': sys.version.split()[0]}))"
)


def inputs(output):
    """The model files of SASH's case and the parquet file: downloaded, or taken from
    FLASHPROVE_TINYLLAMA_INPUTS, and checked against their pins before any program reads them."""
    given = os.environ.get("FLASHPROVE_TINYLLAMA_INPUTS")
    model = CASE["model"]
    base = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/"
    wanted = {name: (base + name, model["files"][name]) for name in KIT["model"]["files"]}
    text = KIT["calibration_text"]
    parquet = "wikitext-2-raw-v1-train.parquet"
    wanted[parquet] = (text["url"], text)
    folder = output / "inputs"
    folder.mkdir()
    paths = {}
    for name, (url, pin) in wanted.items():
        if given:
            path = Path(given) / name
            require(path.is_file() and not path.is_symlink(), f"{path} is not a regular file")
            require(
                path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
                f"{path} is not the pinned file",
            )
        else:
            path = folder / name
            fetch_retrying(url, path, pin)
        paths[name] = path
    model_dir = folder / "model"
    model_dir.mkdir()
    for name in KIT["model"]["files"]:
        (model_dir / name).symlink_to(paths[name].resolve())
    return model_dir, paths[parquet]


def fetch_retrying(url, path, pin, attempts=3, wait=5):
    """fetch(), tried again after a failed or short download: its one connection has a timeout of
    120 s and no retry of its own. The file must still have the pinned size and SHA-256."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch(url, path, pin["sha256"], pin["bytes"])
        except (OSError, http.client.HTTPException, ValueError):
            path.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(wait * attempt)


def minimal_env(output):
    """The environment of a child that reads no PYTHON* variable of the caller's."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(output)}


def same(path, expected, what):
    require(Path(path).read_bytes() == expected, f"{what} differs from the committed kit")


def build_tool(tree, output, source, name, extra=()):
    program = output / name
    args = [*ft.compiler(), "-std=c++17", "-O2", *extra, "-o", program, tree / source]
    code, _ = run(args, output / f"{name}-build.log")
    require(code == 0, f"{name} does not build")
    return program


def rebuild_kit(tree, output, pythons, model_dir, parquet, gguf_dir, expected, verifiers):
    """Steps 1 and 2 of the docstring. Returns what was found."""
    scales, _ = pythons["scales"]
    found = {}
    # 1. The export.
    gguf = gguf_dir / KIT["gguf"]["name"]
    code, _ = run([scales, "-I", PROBE / "export_gguf.py", model_dir, gguf], output / "export.log")
    require(code == 0, "export_gguf.py failed")
    pin = KIT["gguf"]
    require(
        gguf.stat().st_size == pin["bytes"] and sha(gguf) == pin["sha256"],
        "the GGUF file is not the pinned one",
    )
    found["gguf"] = dict(name=gguf.name, bytes=pin["bytes"], sha256=pin["sha256"])
    # 2a. Sums file and fold.txt (scales/README.md, "A real model", step 2).
    weight_sums = build_tool(
        tree,
        output,
        "weight_sums/weight_sums.cpp",
        "weight_sums",
        ["-ffp-contract=off", "-pthread"],
    )
    code, _ = run(
        [weight_sums, "--eres", KIT["e_res"], "--out", gguf_dir / "wsums.txt", gguf],
        output / "wsums.log",
    )
    require(code == 0, "weight_sums failed on the GGUF file")
    (gguf_dir / "fold.txt").write_bytes(b"eres %d\nqkrows asis\n" % KIT["e_res"])
    for name in SETUP_FILES:
        same(gguf_dir / name, expected[name], name)
    # 2b. The calibration set.
    cal = output / "calibration_set.npy"
    tokenizer = model_dir / "tokenizer.json"
    args = [
        scales,
        "-s",
        tree / "scales/make_calibration_set.py",
        "--tinyllama",
        parquet,
        tokenizer,
        cal,
    ]
    code, text = run(args, output / "calibration-set.log", env=minimal_env(output))
    require(
        code == 0 and "the drawn rows are as expected" in text,
        "the calibration set is not the pinned one",
    )
    pin = KIT["calibration_set"]
    require(sha(cal) == pin["sha256"] and cal.stat().st_size == pin["bytes"], "calibration set")
    require(pin["drawn_rows_sha256"] in text, "the drawn rows' SHA-256 is not upstream's")
    found["calibration_set"] = dict(pin, drawn_rows_match_upstream=True)
    # 2c. The fit (step 3).
    fit = output / "fit"
    args = [
        scales,
        "-s",
        tree / "scales/calibrate.py",
        "--gguf",
        gguf_dir,
        "--wsums",
        gguf_dir / "wsums.txt",
    ]
    args += ["--tokens", cal, "--out", fit, "--eres", KIT["e_res"]]
    started = time.monotonic()
    code, _ = run(args, output / "calibrate.log", env=minimal_env(output))
    require(code == 0, "calibrate.py failed")
    found["calibrate_seconds"] = time.monotonic() - started
    # calibrate.py fits at eg 8. fit_eg.py makes the same call at kit.json's eg; at eg 8 it must
    # give calibrate.py's words.json byte for byte, which shows that it is that fit.
    for eg, words in ((8, fit / "words-eg8.json"), (KIT["eg"], fit / "words-eg.json")):
        args = [scales, "-s", "-B", PROBE / "fit_eg.py", tree, gguf_dir, gguf_dir / "wsums.txt"]
        args += [fit / "maxima.json", KIT["e_res"], eg, words]
        code, _ = run(args, output / f"fit-eg{eg}.log", env=minimal_env(output))
        require(code == 0, f"fit_eg.py failed at eg {eg}")
    eg8 = (fit / "words-eg8.json").read_bytes() == (fit / "words.json").read_bytes()
    require(eg8, "fit_eg.py at eg 8 does not give calibrate.py's words.json")
    same(fit / "words-eg.json", expected["words.json"], "words.json")
    found["maxima_equal_to_committed"] = (fit / "maxima.json").read_bytes() == expected[
        "maxima.json"
    ]
    # 2d. The converter (step 4), from check 6's dense base file.
    base = output / "base"
    code, _ = run(
        [sys.executable, "-I", "-B", PROBE / "tinyllama_plan.py", tree, base], output / "base.log"
    )
    require(
        code == 0 and sha(base / "policy.txt") == KIT["base_policy_sha256"],
        "the base file is not check 6's",
    )
    config = output / "config"
    config.mkdir()
    args = [
        scales,
        "-s",
        tree / "scales/words_to_policy.py",
        base / "policy.txt",
        fit / "words-eg.json",
    ]
    args += [gguf_dir / "wsums.txt", config / "policy.txt", config / "KL.bin"]
    code, text = run(args, output / "words_to_policy.log", env=minimal_env(output))
    require(code == 0 and text.rstrip().endswith("w2p end 0"), "words_to_policy.py failed")
    # 2e. The layout (step 5).
    code, arrays = ft.stdout_of(
        [verifiers["patched"], "-plan", "arrays", config, 128], output / "arrays.log"
    )
    require(code == 4, f"-plan arrays: status {code}")
    (output / "arrays.txt").write_text(arrays)
    policy = (config / "policy.txt").read_text().splitlines()
    row_log = next(x for x in policy if x.startswith("group link ")).split(" ")[4]
    pack = [
        sys.executable,
        "-I",
        "-B",
        tree / "prover/tools/pack_layout.py",
        "--arrays",
        output / "arrays.txt",
        row_log,
        4,
    ]
    code, layout = ft.stdout_of(pack, output / "layout.log")
    require(code == 0, "pack_layout.py failed")
    (config / "layout.txt").write_text(layout)
    for name in KIT_FILES:
        same(config / name, expected[name], name)
    found["kit_reproduced"] = True
    return gguf, found


def request_files(folder):
    """SASH's request as FlashProve takes it: prompt and answer ids, filler 0 to T, then r and e."""
    req = KIT["request"]
    ids = CASE["prompt_ids"] + CASE["answer_ids"]
    require(req["r"] == len(CASE["prompt_ids"]) - 1 and req["e"] == len(ids) - 1, "r and e")
    window = ids + [req["filler"]] * (req["T"] - len(ids))
    (folder / "request.bin").write_bytes(
        struct.pack(f"<{req['T'] + 2}I", *window, req["r"], req["e"])
    )
    (folder / "y.bin").write_bytes(
        struct.pack("<I", 0)
    )  # e < T: the token after the window is not covered


def check_planning(tree, output, verifiers):
    """Step 3: both builds' planners and the provers' reader on the committed kit."""
    folder = output / "kit"
    folder.mkdir()
    for name in KIT_FILES:
        shutil.copyfile(KIT_DIR / name, folder / name)
    request_files(folder)
    found = {}
    code, accounting = ft.stdout_of(
        [verifiers["patched"], "-plan", "accounting", folder, 128], output / "plan-accounting.log"
    )
    require(code == 4, f"patched -plan accounting: status {code}")
    ids = ft.planner_ending(accounting, "-plan accounting")
    require(
        ids == (KIT["h_par"], KIT["configuration"]),
        "the kit's h_par or configuration is not the pinned one",
    )
    args = [
        "-plan",
        "request",
        folder / "policy.txt",
        folder / "KL.bin",
        128,
        "-layout",
        folder / "layout.txt",
    ]
    args += ["-prompt", folder / "request.bin", "-reply", folder / "y.bin", "-q"]
    code, request = ft.stdout_of([verifiers["patched"], *args], output / "plan-request.log")
    require(
        code == 4 and ft.planner_ending(request, "-plan request") == ids, "patched -plan request"
    )
    # planner_ending() requires the whole ending, the standard form and the three-tables line.
    found["patched"] = "standard parameter form, three tables, PLAN ONLY (accounting and request)"
    for kind, args_of in (("accounting", ["-plan", "accounting", folder, 128]), ("request", args)):
        code, text = ft.stdout_of(
            [verifiers["upstream"], *args_of], output / f"plan-{kind}-upstream.log"
        )
        last = text.rstrip().splitlines()[-1] if text.strip() else ""
        require(code == 1 and last == "NO: policy-tables", f"unpatched -plan {kind}: {code} {last}")
        found[f"unpatched_{kind}"] = last
    proof = ft.PROOF_BYTES.findall(accounting)
    bits = ft.BITS.findall(accounting)
    require(len(proof) == 1 and len(bits) == 1, "the planner's proof length or bits line")
    found |= dict(
        h_par=ids[0],
        configuration=ids[1],
        proof_bytes_at_most=int(proof[0]),
        bits_compiled_bound=float(bits[0]),
    )
    judge = build_tool(
        tree,
        output,
        "prover/tests/parameter_file_judge.cpp",
        "parameter_file_judge",
        ["-DBLK_FIELD=2", "-DBLK_HASH=1", "-I", tree / "prover", "-I", tree / "blocks"],
    )
    done = subprocess.run(
        [judge], input=f"{folder / 'policy.txt'}\n", capture_output=True, text=True, timeout=60
    )
    (output / "judge.log").write_text(done.stdout + done.stderr)
    require(
        done.returncode == 0 and done.stdout.strip() == "admitted",
        f"provers' reader: {done.stdout.strip()}",
    )
    found["provers_parameter_file_reader"] = "admitted"
    setup = build_tool(
        tree,
        output,
        PROBE / "prover_setup.cpp",
        "prover_setup",
        ["-DBLK_FIELD=2", "-DBLK_HASH=1", "-I", tree / "prover", "-I", tree / "blocks"],
    )
    code, text = ft.stdout_of([setup, folder], output / "prover-setup.log")
    last = text.rstrip().splitlines()[-1] if text.strip() else ""
    require(code == 0 and last.startswith("prover set-up: admitted"), f"cp_prove's set-up: {last}")
    hiding = next(x for x in text.splitlines() if x.startswith("hiding conditions"))
    policy = (folder / "policy.txt").read_text().splitlines()
    # The weight table: upstream's layoutW() and HfParams::from_policy() admit the kit's line
    # table, W has the rows that tinyllama_plan.py sized it for, and the line is that rule's.
    tables = [x for x in text.splitlines() if x.startswith("weight table: admitted (W: ")]
    require(len(tables) == 1, "cp_commit's set-up of the weight table")
    table, rows = table_of(policy)
    require(
        tables[0].startswith(f"weight table: admitted (W: {rows} rows of 2^{table[5]} columns "),
        f"layoutW() gives other rows than {rows}: {tables[0]}",
    )
    require(
        tinyllama_plan.table_numbers(rows, table[0], table[3], table[5]) == table,
        "the kit's line table is not the rule's",
    )

    def control(name, change, wanted):
        """The kit with one line changed must be refused by upstream's set-up code."""
        folder_c = output / f"kit-{name}"
        folder_c.mkdir()
        shutil.copyfile(folder / "KL.bin", folder_c / "KL.bin")
        (folder_c / "policy.txt").write_text("".join(change(x) + "\n" for x in policy))
        code, text = ft.stdout_of([setup, folder_c], output / f"prover-setup-{name}.log")
        refused = text.rstrip().splitlines()[-1] if text.strip() else ""
        require(code == 2 and wanted in refused, f"the control {name}: {code} {refused}")
        return refused

    kfm = next(int(x.split(" ")[1]) for x in policy if x.startswith("kFM "))
    refused = control(
        "kFM-plus-1",
        lambda x: f"kFM {kfm + 1}" if x.startswith("kFM ") else x,
        "attention: the shift sh = ",
    )
    # One step of fi0 less leaves no room for a row's pads.
    step = max(1 << (table[5] - table[0]), 1 << (table[3] - 1).bit_length())
    short = " ".join(map(str, [*table[:2], table[2] - step, *table[3:]]))
    too_short = control(
        "table-one-step-short",
        lambda x: "table " + short if x.startswith("table ") else x,
        "policy-pad-place",
    )
    found["prover_setup"] = dict(
        kit=last,
        hiding=hiding,
        weight_table=tables[0],
        control_kFM_plus_1=refused,
        control_table_one_step_short=too_short,
    )
    return found


def table_of(policy):
    """The numbers of the kit's line table, and W's rows by tinyllama_plan.weight_rows()."""
    table = tinyllama_plan.frame_numbers(policy, "table")
    words = {
        w[0]: int(w[1]) for w in (x.split(" ") for x in policy) if len(w) == 2 and w[1].isdigit()
    }
    return table, tinyllama_plan.weight_rows(words | {"table": table})


# SUITE.md's rows for two of the tests: status 0, a pattern the last line must match, and patterns
# that some line must match.
SUITE_ROWS = {
    "test_policy_tables.py": (r"^OK", [r"^Ran 23 tests? in"]),
    "table_ties.py": (r"^OK: 26 checks$", []),
}


def suite_passes(test, code, text):
    """Whether a test's status and output pass its SUITE.md row."""
    lines = text.rstrip("\n").splitlines() or [""]
    last, others = SUITE_ROWS[test]
    found = all(any(re.search(x, line) for line in lines) for x in others)
    return code == 0 and re.search(last, lines[-1]) is not None and found


def check_scale_tests(tree, output, scales):
    """Step 4: upstream's scale-tool tests on the patched tree (see the module's docstring)."""
    rows = {}
    for test in (
        "test_policy_tables.py",
        "table_ties.py",
        "test_calibrate.py",
        "test_gguf_weights.py",
    ):
        code, text = run(
            [scales, "-B", tree / "scales/tests" / test],
            output / f"scales-{test}.log",
            cwd=tree,
            env=minimal_env(output),
        )
        last = text.rstrip().splitlines()[-1] if text.strip() else ""
        # Two tests have SUITE.md rows (the patched row counts 23 tests). The other two have none:
        # 0, or 77 for "passed in part" (this target builds no make_model and sets no SMALL_MODEL,
        # so those cases skip).
        if test in SUITE_ROWS:
            passed = suite_passes(test, code, text)
        else:
            passed = (code == 0 and last.startswith("OK")) or (
                code == 77 and last.startswith("in part: ")
            )
        require(passed and "FAILED" not in text, f"scales/tests/{test}: {code} {last}")
        rows[test] = dict(status=code, last_line=last)
    return rows


def failures(answer):
    """The conditions of the decisive check that an answer_check.py result misses (none: pass)."""
    bounds = KIT["bounds"]
    if answer["range_failure"] is not None:
        return [f"a range condition fails: {answer['range_failure']}"]
    missed = []
    if not answer["matches"]:
        missed.append(f"the choices are {answer['choices']}, not SASH's {CASE['answer_ids']}")
    if not answer["gate_key_within_limit"]:
        reached = answer["peaks"].get("key")
        missed.append(
            "the gate key is not recorded" if reached is None else f"the gate key reaches {reached}"
        )
    req, ids = KIT["request"], CASE["prompt_ids"] + CASE["answer_ids"]
    if (answer["T"], answer["r"], answer["e"]) != (req["T"], req["r"], req["e"]):
        missed.append("the window or its range is not kit.json's request")
    if answer["window"] != ids + [req["filler"]] * (req["T"] - len(ids)):
        missed.append("the window is not SASH's ids with kit.json's filler")
    if [r["position"] for r in answer["positions"]] != list(range(answer["r"], answer["e"])):
        missed.append("the checked positions are not r .. e - 1")
    low, high = bounds["logit_slope"]
    for row in answer["positions"]:
        if not low <= row["slope"] <= high:
            missed.append(f"the logits' scale at position {row['position']}: {row['slope']:.4f}")
        if not row["correlation"] >= bounds["logit_correlation"]:  # NaN fails too
            missed.append(
                f"the logits' correlation at position {row['position']}: {row['correlation']:.5f}"
            )
    return missed


def mutated(policy, kl):
    """The kit with one-bit changes of the model itself: the final norm's MF (it doubles every
    logit), and one layer's value shift kv. (Not the line s: without a seed the forward pass does
    not read it, so a change of s would test only this target's divisor.)"""
    lines = policy.decode("ascii").split("\n")
    at = next(i for i, x in enumerate(lines) if x.startswith("MF "))
    mf_plus = lines[:at] + [f"MF {int(lines[at].split()[1]) + 1}"] + lines[at + 1 :]
    words = list(struct.unpack(f"<{len(kl) // 4}I", kl))
    words[10 * len(LAYER_WORDS) + LAYER_WORDS.index("kv")] += 1
    return {
        "final-norm-MF-plus-1": ("\n".join(mf_plus).encode("ascii"), kl),
        "layer-10-kv-plus-1": (policy, struct.pack(f"<{len(words)}I", *words)),
    }


def check_answer(tree, output, pythons, model_dir, gguf):
    """Steps 5 and 6."""
    scales, _ = pythons["scales"]
    im, _ = pythons["integer_model"]
    case = output / "response-case.json"
    case.write_bytes(CASE_PATH.read_bytes())
    logits = output / "float-logits.npy"
    float_json = output / "float-reference.json"
    args = [
        scales,
        "-I",
        PROBE / "float_reference.py",
        tree / "scales",
        gguf.parent,
        model_dir,
        case,
        float_json,
        logits,
    ]
    offline = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(output),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    code, _ = run(args, output / "float-reference.log", env=offline)
    require(code == 0, "float_reference.py failed")
    floats = json.loads(float_json.read_text())
    bounds = KIT["bounds"]
    require(
        floats["final_norm_relative_l2"] <= bounds["final_norm_relative_l2"],
        "the float pass differs from transformers",
    )
    require(
        floats["float_greedy_is_the_answer"],
        f"float greedy decoding gives {floats['float_greedy']}",
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(output),
        "PYTHONPATH": str(tree / "integrations/vllm/integer_model"),
    }

    def integer_run(name, policy, kl):
        """answer_check.py on the GGUF file with this policy.txt and KL.bin, every file pinned."""
        model = output / f"integer-model-{name}"
        model.mkdir()
        (model / "policy.txt").write_bytes(policy)
        (model / "KL.bin").write_bytes(kl)
        settings = dict(
            blocks="gguf", setup_folder=str(gguf.parent), shards=[str(gguf)], verify_sums=True
        )
        (model / "integer_model.json").write_text(
            json.dumps(dict(weights="blocks", weight_settings=settings))
        )
        digests = {"integer_model.json": sha(model / "integer_model.json")}
        digests |= {"policy.txt": hashlib.sha256(policy).hexdigest()}
        digests |= {"KL.bin": hashlib.sha256(kl).hexdigest()}
        digests |= {n: KIT["files"][n] for n in SETUP_FILES}
        pinned = output / f"integer-model-{name}-digests.json"  # outside the model directory
        pinned.write_text(json.dumps(digests))
        found = output / f"answer-{name}.json"
        args = [im, "-s", PROBE / "answer_check.py", model, pinned, case, logits, found]
        code, _ = run(args, output / f"answer-check-{name}.log", env=env)
        require(code == 0, f"answer_check.py failed ({name})")
        return json.loads(found.read_text())

    kit = committed_kit()
    require(
        hashlib.sha256(kit["policy.txt"]).hexdigest() == KIT["files"]["policy.txt"]
        and hashlib.sha256(kit["KL.bin"]).hexdigest() == KIT["files"]["KL.bin"],
        "the kit's policy.txt or KL.bin is not the pinned one",
    )
    answer = integer_run("kit", kit["policy.txt"], kit["KL.bin"])
    missed = failures(answer)
    require(not missed, "the decisive check fails: " + "; ".join(missed))
    # Controls: one-bit changes of the kit must fail at least one condition, or the bounds have
    # no power. The answer check's own figures are recorded for each.
    controls = {}
    for name, (policy, kl) in mutated(kit["policy.txt"], kit["KL.bin"]).items():
        found = integer_run(name, policy, kl)
        slopes = [row["slope"] for row in found.get("positions", [])]
        controls[name] = dict(
            failed=failures(found),
            choices=found.get("choices"),
            slope_range=[min(slopes), max(slopes)] if slopes else None,
            least_correlation=min(
                (r["correlation"] for r in found.get("positions", [])), default=None
            ),
        )
        require(controls[name]["failed"], f"the control {name} passes every condition")
    answer["controls"] = controls
    return floats, answer


def worst_case_findings(port, floats):
    """Step 7's conditions on worst_case_cpu.py's result and gate_float_bound.py's: the ones that
    fail (none: as kit.json records)."""
    expected = KIT["worst_case"]
    missed = []
    if port["short"] != expected["short"]:
        missed.append(f"the sites that do not fit are {port['short']}, not {expected['short']}")
    gates = [layer for site, layer in port["short"] if site == "gate"]
    if floats["layers_over_the_limit"] != gates:
        missed.append(
            f"the float bound has the gates of {floats['layers_over_the_limit']} over the limit"
        )
    if len(port["layers"]) != len(floats["layers"]):
        missed.append("the two computations have different numbers of layers")
    for mine, real in zip(port["layers"], floats["layers"], strict=False):
        ratio = mine["gate_key_bound"] / real["key"] / (mine["ball_2"] / 2048)
        if not abs(ratio - 1) <= expected["ratio_tolerance"]:  # NaN fails too
            missed.append(
                f"layer {mine['layer']}: the gate key bounds differ by a factor {ratio:.5f}"
            )
    return missed


def check_worst_case(tree, output, pythons, model_dir):
    """Step 7: cp_commit's worst-case check on a CPU (see the module's docstring)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(output),
        "PYTHONPATH": str(tree / "integrations/vllm/integer_model"),
    }
    port_json, float_json = output / "worst-case-cpu.json", output / "gate-float-bound.json"
    args = [pythons["integer_model"][0], "-s", "-B", PROBE / "worst_case_cpu.py"]
    args += [output / "integer-model-kit", output / "integer-model-kit-digests.json", port_json]
    code, _ = run(args, output / "worst-case-cpu.log", env=env)
    require(code == 0, "worst_case_cpu.py failed")
    args = [pythons["scales"][0], "-I", "-B", PROBE / "gate_float_bound.py", model_dir, KIT["eg"]]
    args += [float_json]
    code, _ = run(args, output / "gate-float-bound.log", env=minimal_env(output))
    require(code == 0, "gate_float_bound.py failed")
    port, floats = json.loads(port_json.read_text()), json.loads(float_json.read_text())
    missed = worst_case_findings(port, floats)
    require(not missed, "the CPU worst-case check: " + "; ".join(missed))
    return dict(
        short=port["short"],
        row_sites_fit=port["row_sites_fit"],
        public_sites_fit=port["public_sites_fit"],
        stream_sites_fit_by_frobenius=port["stream_sites_fit_by_frobenius"],
        worst_by_site=port["worst_by_site"],
        gate_key_bounds=[x["gate_key_bound"] for x in port["layers"]],
        gate_rows_beyond_the_limit=[x["gate_rows_beyond_the_limit"] for x in port["layers"]],
        float_gate_keys=[round(x["key"], 1) for x in floats["layers"]],
        float_rows_beyond_the_limit=[x["rows_beyond_the_limit"] for x in floats["layers"]],
        rows_recomputed_exactly=port["rows_recomputed_exactly"],
        # No hash of the port's file: it holds the run's seconds, so its hash cannot be recomputed.
        float_sha256=sha(float_json),
    )


def verify(output):
    started = time.monotonic()
    copies = pinned_copies(output)
    expected = committed_kit()
    tree = fp.source_tree(output)
    verifiers, builds = {}, {}
    verifiers["upstream"], builds["upstream"] = fp.build_verifier(
        tree, output, ft.UPSTREAM_DIGEST, "bin-upstream"
    )
    ft.apply_patches(tree, output, copies)
    verifiers["patched"], builds["patched"] = fp.build_verifier(
        tree, output, ft.PINS["verifier_source_digest"], "bin-patched"
    )
    pythons = {name: environment(output, name) for name in KIT["environments"]}
    model_dir, parquet = inputs(output)
    gguf_dir = output / "weights"
    gguf_dir.mkdir()
    gguf, kit = rebuild_kit(
        tree, output, pythons, model_dir, parquet, gguf_dir, expected, verifiers
    )
    result = dict(builds=builds, environments={k: v[1] for k, v in pythons.items()}, kit=kit)
    result["planning"] = check_planning(tree, output, verifiers)
    result["scale_tool_tests"] = check_scale_tests(tree, output, pythons["scales"][0])
    result["float_reference"], result["answer"] = check_answer(
        tree, output, pythons, model_dir, gguf
    )
    result["worst_case"] = check_worst_case(tree, output, pythons, model_dir)
    result["seconds"] = time.monotonic() - started
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="flashprove-tinyllama-answer-", dir=ROOT / "runs"))
    print("Outputs: " + str(output), flush=True)
    try:
        require(shutil.which("git") and shutil.which("uv"), "git and uv are needed")
        result = verify(output)
        result |= dict(
            passed=True,
            schema_id="hallmark.flashprove-tinyllama-answer/1",
            source=fp.PIN,
            kit_pins=KIT,
            host=dict(system=platform.system(), machine=platform.machine()),
            proof_made=False,
            gpu_used=False,
            hiding_claim=False,
        )
        write_json(output / "result.json", result)
        print(
            json.dumps(
                dict(
                    passed=True,
                    choices=result["answer"]["choices"],
                    worst_case_short=result["worst_case"]["short"],
                    seconds=round(result["seconds"]),
                )
            )
        )
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
