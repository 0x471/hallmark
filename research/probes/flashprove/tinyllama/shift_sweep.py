"""A one-off sweep of the decisive check's power (decision 0094): every layer's kv, ko and kd shift
changed by one bit, 66 kits, each run through answer_check.py and judged by the target's own
failures().

Usage: uv run --locked python research/probes/flashprove/tinyllama/shift_sweep.py RUN_DIR OUT_JSON

RUN_DIR is the output folder of a passing `make verify-flashprove-tinyllama-answer` (runs/...): its
integer model environment, patched tree, GGUF file and setup files, float logits and its pinned
kit's run are reused, every file pinned as the target pins them. A kit that no condition catches
is reported, not refused: the sweep measures the bounds, it is not part of the target.
"""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import flashprove_tinyllama_answer as target

SHIFTS = ("kv", "ko", "kd")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_dir, out_json):
    run = Path(run_dir).resolve()
    out = Path(out_json).resolve()
    work = out.parent / (out.stem + "-kits")
    work.mkdir(parents=True, exist_ok=False)
    passed = json.loads((run / "result.json").read_text())

    # The pins, not the notes: kit.json's prose on the calibration text's licence may change.
    def pins(kit):
        return {k: v for k, v in kit.items() if k != "calibration_text"}

    if not passed.get("passed") or pins(passed["kit_pins"]) != pins(target.KIT):
        raise SystemExit("RUN_DIR is not a passing run of this kit")
    kit = target.committed_kit()
    policy, kl = kit["policy.txt"], kit["KL.bin"]
    words = list(struct.unpack(f"<{len(kl) // 4}I", kl))
    settings = json.loads((run / "integer-model-kit/integer_model.json").read_text())
    python = run / "venv-integer_model/bin/python"
    tree = next((run / "src").iterdir())
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work),
        "PYTHONPATH": str(tree / "integrations/vllm/integer_model"),
    }
    rows = []
    for layer in range(len(words) // len(target.LAYER_WORDS)):
        for shift in SHIFTS:
            name = f"L{layer}-{shift}+1"
            model = work / name
            model.mkdir()
            changed = list(words)
            changed[layer * len(target.LAYER_WORDS) + target.LAYER_WORDS.index(shift)] += 1
            (model / "policy.txt").write_bytes(policy)
            (model / "KL.bin").write_bytes(struct.pack(f"<{len(changed)}I", *changed))
            (model / "integer_model.json").write_text(json.dumps(settings))
            digests = {n: sha(model / n) for n in ("integer_model.json", "policy.txt", "KL.bin")}
            digests |= {n: target.KIT["files"][n] for n in target.SETUP_FILES}
            pinned = work / f"{name}-digests.json"
            pinned.write_text(json.dumps(digests))
            found = work / f"{name}.json"
            args = [python, "-s", target.PROBE / "answer_check.py", model, pinned, target.CASE_PATH]
            args += [run / "float-logits.npy", found]
            subprocess.run(args, env=env, check=True, capture_output=True)
            answer = json.loads(found.read_text())
            missed = target.failures(answer)
            positions = answer.get("positions", [])  # absent after a range failure
            slopes = [p["slope"] for p in positions]
            least = min((p["correlation"] for p in positions), default=None)
            # Caught by the correlation floor alone, and at least 0.99: the earlier floor passes it.
            floor_only = bool(missed) and all(
                m.startswith("the logits' correlation") for m in missed
            )
            rows.append(
                dict(
                    mutation=name,
                    caught=bool(missed),
                    missed=missed,
                    passes_a_floor_of_0_99=not missed or (floor_only and least >= 0.99),
                    tokens_same=answer.get("matches", False),
                    range_failure=answer["range_failure"],
                    min_correlation=least,
                    slope_range=[min(slopes), max(slopes)] if slopes else None,
                )
            )
            print(name, "caught" if missed else "NOT CAUGHT", flush=True)
    kept = [r for r in rows if r["tokens_same"]]
    summary = dict(
        program="research/probes/flashprove/tinyllama/shift_sweep.py",
        run_result_sha256=sha(run / "result.json"),
        kits=len(rows),
        caught=sum(r["caught"] for r in rows),
        tokens_changed=sum(r["range_failure"] is None and not r["tokens_same"] for r in rows),
        range_failures=sum(r["range_failure"] is not None for r in rows),
        highest_min_correlation_with_tokens_kept=max(r["min_correlation"] for r in kept)
        if kept
        else None,
        would_pass_a_floor_of_0_99=sum(r["passes_a_floor_of_0_99"] for r in rows),
        rows=rows,
    )
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"caught {summary['caught']} of {summary['kits']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
