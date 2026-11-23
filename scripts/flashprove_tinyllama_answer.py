"""Rebuild the TinyLlama-1.1B kit for pinned FlashProve and run the decisive CPU check (0094).

`make verify-flashprove-tinyllama-answer` runs on a CPU, with no GPU and no spend. It downloads the
pinned upstream archive, TinyLlama-1.1B-Chat-v1.0's files at the revision of Hallmark's SASH path
(about 2.2 GB) and WikiText-2's training split, applies the three pinned patches and scales.patch,
and then requires, in order:

1. the export: export_gguf.py writes the GGUF file with the pinned SHA-256;
2. the kit, rebuilt by upstream's own route (scales/README.md, "A real model", steps 1 to 5) with
   the patched tools: weight_sums, make_calibration_set.py --tinyllama, calibrate.py and
   words_to_policy.py from check 6's dense base file, -plan arrays and pack_layout.py. Every file
   must equal the committed kit (tinyllama/kit/) byte for byte; maxima.json is compared and
   recorded;
3. the patched verifier's planner admits the kit in the standard parameter form, for the whole kit
   and for SASH's request, with the pinned h_par and configuration; the unpatched one refuses it
   (policy-tables); the provers' parameter-file reader admits it;
4. upstream's scale-tool tests pass on the patched tree;
5. the float reference: upstream's float pass on the GGUF agrees with Hugging Face transformers on
   the safetensors, and float greedy decoding gives SASH's answer;
6. the decisive check: upstream's integer model (patched), on the GGUF and the committed kit, with
   every file pinned and the weights checked against wsums.txt, gives SASH's eight answer tokens
   after SASH's 25 prompt tokens, in one window of 64 tokens with no failed range condition.

No commitment, proof or GPU run is made. Equal tokens show that the integer model is TinyLlama on
this request; they do not show that FlashProve's and SASH's integer functions are equal.
Set FLASHPROVE_TINYLLAMA_INPUTS to a folder holding the pinned model files and parquet file to skip
the downloads; each file is checked against its pin before it is read.
"""

import argparse
import hashlib
import json
import os
import platform
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
KIT_DIR = PROBE / "kit"
KIT = json.loads((PROBE / "kit.json").read_text())
CASE_PATH = ROOT / KIT["model"]["case"]
CASE = json.loads(CASE_PATH.read_text())
KIT_FILES = ("policy.txt", "KL.bin", "layout.txt")  # the configuration directory
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


def environment(output, name):
    """A new virtual environment with the pinned packages, made by uv."""
    spec = KIT["environments"][name]
    venv = output / f"venv-{name}"
    code, _ = run(["uv", "venv", "-q", "-p", spec["python"], venv], output / f"venv-{name}.log")
    require(code == 0, f"uv cannot make the {name} environment")
    python = venv / "bin/python"
    packages = [f"{k}=={v}" for k, v in spec["packages"].items()]
    linux = platform.system() == "Linux"
    if linux:  # the CPU build of torch; PyPI's Linux wheels pull CUDA libraries
        torch = [p for p in packages if p.startswith("torch==")]
        packages = [p for p in packages if not p.startswith("torch==")]
        code, _ = run(
            ["uv", "pip", "install", "-q", "-p", python, "--index-url", CPU_TORCH, *torch],
            output / f"venv-{name}-torch.log",
        )
        require(code == 0, f"cannot install torch into the {name} environment")
    code, _ = run(
        ["uv", "pip", "install", "-q", "-p", python, *packages], output / f"venv-{name}-install.log"
    )
    require(code == 0, f"cannot install the {name} packages")
    code, text = run(
        [python, "-I", "-c", VERSIONS, *spec["packages"]], output / f"venv-{name}-versions.log"
    )
    require(code == 0, f"the {name} environment does not import")
    versions = json.loads(text)
    for package, wanted in spec["packages"].items():
        require(
            versions[package].split("+")[0] == wanted, f"{name}: {package} is {versions[package]}"
        )
    return python, versions


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
            fetch(url, path, pin["sha256"], pin["bytes"])
        paths[name] = path
    model_dir = folder / "model"
    model_dir.mkdir()
    for name in KIT["model"]["files"]:
        (model_dir / name).symlink_to(paths[name].resolve())
    return model_dir, paths[parquet]


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
    code, text = run(args, output / "calibration-set.log")
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
    code, _ = run(args, output / "calibrate.log")
    require(code == 0, "calibrate.py failed")
    found["calibrate_seconds"] = time.monotonic() - started
    same(fit / "words.json", expected["words.json"], "words.json")
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
        fit / "words.json",
    ]
    args += [gguf_dir / "wsums.txt", config / "policy.txt", config / "KL.bin"]
    code, text = run(args, output / "words_to_policy.log")
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
    return found


def check_scale_tests(tree, output, scales):
    """Step 4: upstream's scale-tool tests on the patched tree, as SUITE.md runs them."""
    rows = {}
    for test in (
        "test_policy_tables.py",
        "table_ties.py",
        "test_calibrate.py",
        "test_gguf_weights.py",
    ):
        code, text = run(
            [scales, "-B", tree / "scales/tests" / test], output / f"scales-{test}.log", cwd=tree
        )
        last = text.rstrip().splitlines()[-1] if text.strip() else ""
        # 0: all ran and passed. 77: upstream's status for "passed in part", when cases need what
        # this host lacks (make_model needs SHA-NI; SMALL_MODEL is its output).
        whole = code == 0 and (last == "OK" or last.startswith("OK: "))
        partial = code == 77 and last.startswith("in part: ")
        require((whole or partial) and "FAILED" not in text, f"scales/tests/{test}: {code} {last}")
        rows[test] = dict(status=code, last_line=last)
    return rows


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
    model = output / "integer-model"
    model.mkdir()
    for name in ("policy.txt", "KL.bin"):
        shutil.copyfile(KIT_DIR / name, model / name)
    settings = dict(
        blocks="gguf", setup_folder=str(gguf.parent), shards=[str(gguf)], verify_sums=True
    )
    (model / "integer_model.json").write_text(
        json.dumps(dict(weights="blocks", weight_settings=settings))
    )
    digests = {name: sha(model / name) for name in ("integer_model.json", "policy.txt", "KL.bin")}
    digests |= {name: KIT["files"][name] for name in SETUP_FILES}
    pinned = (
        output / "integer-model-digests.json"
    )  # outside the model directory, as load_from_folder asks
    pinned.write_text(json.dumps(digests))
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(output),
        "PYTHONPATH": str(tree / "integrations/vllm/integer_model"),
    }
    answer_json = output / "answer.json"
    args = [im, "-s", PROBE / "answer_check.py", model, pinned, case, logits, answer_json]
    code, _ = run(args, output / "answer-check.log", env=env)
    require(code == 0, "answer_check.py failed")
    answer = json.loads(answer_json.read_text())
    require(answer["range_failure"] is None, f"a range condition fails: {answer['range_failure']}")
    require(
        answer["matches"],
        f"the integer model gives {answer['choices']}, not SASH's {CASE['answer_ids']}",
    )
    require(answer["gate_key_within_limit"], f"the gate key reaches {answer['peaks']['key']}")
    low, high = bounds["logit_slope"]
    for row in answer["positions"]:
        require(
            low <= row["slope"] <= high,
            f"the logits' scale at position {row['position']}: {row['slope']}",
        )
        require(
            row["correlation"] >= bounds["logit_correlation"],
            f"the logits at position {row['position']}",
        )
    return floats, answer


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
                    seconds=round(result["seconds"]),
                )
            )
        )
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
