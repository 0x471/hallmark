"""Offline checks of the TinyLlama kit, its pins and the caller-side code of the answer check.

No file is fetched and no model runs: `make verify-flashprove-tinyllama-answer` does that.
"""

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flashprove_tinyllama as harness
import flashprove_tinyllama_answer as answer

ROOT = Path(__file__).resolve().parents[1]

PROBE = harness.PROBE
KIT = json.loads((PROBE / "kit.json").read_text())
CASE = json.loads((ROOT / KIT["model"]["case"]).read_text())
LAYER_WORDS = ("kq", "kk", "kv", "ko", "kg", "ku", "kd", "M1", "K1", "M2", "K2", "sh", "rp")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(name):
    return (PROBE / "kit" / name).read_text().splitlines()


def values(name):
    return dict(x.split(" ", 1) for x in lines(name) if " " in x)


def test_the_kit_files_and_patches_match_their_pins():
    assert sorted(p.name for p in (PROBE / "kit").iterdir()) == sorted(KIT["files"])
    for name, digest in KIT["files"].items():
        assert sha(PROBE / "kit" / name) == digest, name
    patch = PROBE / KIT["scales_patch"]["name"]
    assert patch.stat().st_size == KIT["scales_patch"]["bytes"]
    assert sha(patch) == KIT["scales_patch"]["sha256"]
    assert sha(PROBE / "patches.json") == KIT["patches_json_sha256"]
    assert KIT["upstream_revision"] == harness.PINS["upstream_revision"]
    assert answer.committed_kit().keys() == KIT["files"].keys()


def test_scales_patch_touches_only_the_scale_tools_and_marks_every_file():
    text = (PROBE / KIT["scales_patch"]["name"]).read_text()
    files = re.findall(r"^diff --git a/(\S+) b/\1$", text, re.M)
    assert files and all(f.startswith("scales/") or f == "SUITE.md" for f in files), files
    assert text.isascii()
    for path in files:  # Apache-2.0 section 4(b)
        hunk = text.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("\ndiff --git ", 1)[0]
        assert "Modified by the Hallmark project" in hunk, path


def test_the_model_pins_are_the_sash_paths():
    model = CASE["model"]
    assert model["id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert set(KIT["model"]["files"]) <= set(model["files"])
    # TinyLlama's tokenizer is byte for byte the Llama 2 tokenizer that upstream's rule names.
    assert model["files"]["tokenizer.json"]["sha256"] == (
        "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e"
    )


def test_the_parameter_file_is_tinyllama_dense_and_fitted():
    v = values("policy.txt")
    shape = {k: int(v[k]) for k in ("lt", "L", "d", "H", "G", "dh", "f", "NE", "V")}
    assert shape == dict(lt=6, L=22, d=2048, H=32, G=4, dh=64, f=5632, NE=0, V=32000)
    assert (v["theta0_lo"], v["theta0_hi"]) == ("10000", "0")
    digests = [
        x.split(" ")[0] for x in lines("policy.txt") if re.match(r"t(silu|sig|rope|exp)\d ", x)
    ]
    assert digests == [f"{t}{i}" for t in ("tsilu", "trope", "texp") for i in range(8)]
    words = json.loads((PROBE / "kit/words.json").read_text())["W"]
    assert int(v["s"]) == words["E_logit"]
    assert (int(v["MF"]), int(v["KF"])) == (words["MF"], words["KF"])
    for name in ("eps1", "eps2", "epsF"):
        assert int(v[f"{name}_lo"]) + (int(v[f"{name}_hi"]) << 32) == words[name]
    # eps = d * 1e-5 * 4^e_res, rounded, by the converter's exact formula
    assert words["eps1"] == (2 * 2048 * 4 ** KIT["e_res"] + 100000) // 200000
    assert words["e_res"] == KIT["e_res"] and words["theta"] == 10000


def test_kl_bin_holds_the_fitted_words_of_every_layer():
    data = (PROBE / "kit/KL.bin").read_bytes()
    rows = json.loads((PROBE / "kit/words.json").read_text())["KL"]
    assert len(rows) == 22 and len(data) == 4 * len(LAYER_WORDS) * 22  # no expert words at NE 0
    words = struct.unpack(f"<{len(data) // 4}I", data)
    expected = [row[k] for row in rows for k in LAYER_WORDS]
    assert list(words) == expected
    assert all(row["rp"] == 1 for row in rows)  # the rotary encoding in every layer
    # Norm constants at width 2,048 (t = 5): M = 32 + Es + t with Es <= 14 - t.
    assert all(32 + 5 <= row[m] <= 32 + 14 for row in rows for m in ("M1", "M2"))


def test_the_setup_files():
    assert lines("fold.txt") == [f"eres {KIT['e_res']}", "qkrows asis"]
    sums = lines("wsums.txt")
    assert sums[-1] == "END 156"
    names = [x.split(" ")[0] for x in sums[:-1]]
    kinds = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
    wanted = {f"blk.{n}.{k}.weight" for n in range(22) for k in kinds}
    assert set(names) == wanted | {"token_embd.weight", "output.weight"} and len(names) == 156
    exponents = {x.split(" ")[0]: int(x.split(" ")[2]) for x in sums[:-1]}
    assert (
        exponents["token_embd.weight"] == KIT["e_res"]
    )  # the embedding shares the stream's exponent


def test_the_request_is_sashs_prompt_and_answer_in_one_window(tmp_path):
    answer.request_files(tmp_path)
    req = KIT["request"]
    data = (tmp_path / "request.bin").read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    prompt, reply = CASE["prompt_ids"], CASE["answer_ids"]
    assert len(words) == req["T"] + 2
    assert list(words[: req["T"]]) == prompt + reply + [0] * (req["T"] - len(prompt) - len(reply))
    assert (
        words[req["T"] :] == (len(prompt) - 1, len(prompt) + len(reply) - 1) == (req["r"], req["e"])
    )
    assert req["e"] < req["T"]  # so y is not covered, and the token file holds 0
    assert (tmp_path / "y.bin").read_bytes() == struct.pack("<I", 0)
    # Checked tokens r + 1 .. e: exactly SASH's eight answer tokens, the last of them EOS.
    assert words[req["r"] + 1 : req["e"] + 1] == tuple(reply) and reply[-1] == 2


def test_the_environments_pin_upstreams_versions():
    scales = KIT["environments"]["scales"]["packages"]
    im = KIT["environments"]["integer_model"]["packages"]
    # Upstream requirements.in (scale tools) and integrations/vllm/requirements.txt (integer model).
    assert (scales["torch"], scales["numpy"], scales["gguf"]) == ("2.9.1", "2.3.5", "0.19.0")
    assert (scales["pyarrow"], scales["tokenizers"]) == ("23.0.1", "0.22.2")
    assert (im["torch"], im["numpy"], im["gguf"]) == ("2.13.0", "2.4.6", "0.19.0")


def a_passing_answer():
    rows = [
        dict(
            position=24 + i, token=tok, margin=1.0, float_margin=1.0, slope=1.0, correlation=0.9995
        )
        for i, tok in enumerate(CASE["answer_ids"])
    ]
    ids = CASE["prompt_ids"] + CASE["answer_ids"]
    return dict(
        T=64,
        window=ids + [0] * (64 - len(ids)),
        r=24,
        e=32,
        range_failure=None,
        choices=CASE["answer_ids"],
        matches=True,
        gate_key_within_limit=True,
        peaks=dict(key=7108),
        positions=rows,
    )


def test_failures_passes_only_an_answer_that_meets_every_condition():
    assert answer.failures(a_passing_answer()) == []
    cases = {
        "range": dict(range_failure=dict(position=40, condition=6)),
        "choices": dict(matches=False, choices=[1] * 8),
        "gate key": dict(gate_key_within_limit=False, peaks=dict(key=40000)),
        "positions": dict(r=23),
        "window": dict(window=[0] * 64),
        "T": dict(T=128),
    }
    for what, change in cases.items():
        assert answer.failures(a_passing_answer() | change), what
    for field, value in (
        ("slope", 0.5),
        ("slope", 2.0),
        ("correlation", 0.995),
        ("correlation", float("nan")),
    ):
        bad = a_passing_answer()
        bad["positions"][3][field] = value
        assert answer.failures(bad), (field, value)


def test_the_mutation_controls_change_one_value_by_one():
    policy = (PROBE / "kit/policy.txt").read_bytes()
    kl = (PROBE / "kit/KL.bin").read_bytes()
    controls = answer.mutated(policy, kl)
    assert sorted(controls) == sorted(KIT["mutation_controls"])
    mf_policy, mf_kl = controls["final-norm-MF-plus-1"]
    before, after = policy.decode().split("\n"), mf_policy.decode().split("\n")
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert mf_kl == kl and len(changed) == 1 and changed[0][0].startswith("MF ")
    assert int(changed[0][1].split()[1]) == int(changed[0][0].split()[1]) + 1
    kv_policy, kv_kl = controls["layer-10-kv-plus-1"]
    old, new = (struct.unpack(f"<{len(kl) // 4}I", x) for x in (kl, kv_kl))
    diff = [i for i in range(len(old)) if old[i] != new[i]]
    assert kv_policy == policy and diff == [10 * 13 + 2] and new[diff[0]] == old[diff[0]] + 1


def test_same_refuses_a_changed_file(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    answer.same(path, b"abc", "f")
    with pytest.raises(ValueError):
        answer.same(path, b"abd", "f")


def test_the_lock_files_are_pinned_and_hold_the_pinned_versions():
    folder = PROBE / "requirements"
    named = set()
    for env, spec in KIT["environments"].items():
        for platform_key, lock in spec["locks"].items():
            path = folder / lock["file"]
            named.add(lock["file"])
            assert sha(path) == lock["sha256"]
            text = path.read_text()
            pins = dict(
                x.split(" ")[0].split("==") for x in text.splitlines() if re.match(r"[a-z]", x)
            )
            hashed = re.findall(r"^[a-z][^ ]*==[^ ]+ \\\n +--hash=sha256:[0-9a-f]{64}", text, re.M)
            assert len(hashed) == len(pins), lock["file"]  # every pin is followed by its hashes
            assert len(answer.locked(path)) == len(
                pins
            )  # what the target compares with uv pip freeze
            for package, version in spec["packages"].items():
                assert pins[package].split("+")[0] == version, (env, platform_key, package)
            assert not any(p.startswith(("nvidia", "triton")) for p in pins), lock["file"]
            if platform_key.startswith("linux"):
                assert pins["torch"].endswith("+cpu")
    assert {p.name for p in folder.glob("*.txt")} == named
