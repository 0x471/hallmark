"""The text of the dense TinyLlama request against its token ids, on a CPU.

usage: python scripts/flashprove_tinyllama_request.py

FlashProve proves a statement about token ids. This check ties the ids of SASH's pinned case to
its text, as SASH's own preparation does: the pinned tokenizer with its chat template must turn
the case's prompt into the case's prompt ids, the answer ids must decode to the case's response,
and the answer's last id must be the tokenizer's end token. The tokenizer's files are those that
the case pins (SHA-256 and size), fetched from the model's public repository at the pinned
revision, and they are read in the kit's hash-locked `scales` environment. No GPU, no proof.
"""

import json
import tempfile
from pathlib import Path

import flashprove_tinyllama_answer as answer
from run_e2e import write_json
from run_llm_e2e import require

ROOT, PROBE, CASE = answer.ROOT, answer.PROBE, answer.CASE
FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def judged(said):
    """What the tokenizer said, held to the case; returns the record."""
    require(said["prompt_ids"] == CASE["prompt_ids"], "the tokenizer or the prompt changed")
    require(said["response"] == CASE["response"], "the answer's ids decode to another text")
    require(
        said["eos_token_id"] == CASE["answer_ids"][-1], "the answer does not end at the end token"
    )
    return dict(
        prompt_tokens=len(said["prompt_ids"]),
        answer_tokens=len(CASE["answer_ids"]),
        end_token=said["eos_token_id"],
        chat_template_applied=True,
        tokenizer_files={name: CASE["model"]["files"][name]["sha256"] for name in FILES},
    )


def check_text(output):
    """Fetches the pinned tokenizer files, reads them in the scales environment and judges."""
    model = CASE["model"]
    base = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/"
    folder = output / "tokenizer"
    folder.mkdir()
    for name in FILES:
        answer.fetch_retrying(base + name, folder / name, model["files"][name])
    python, versions = answer.environment(output, "scales")
    code, text = answer.run(
        [python, "-I", PROBE / "request_text.py", folder, answer.CASE_PATH],
        output / "request-text.log",
        env=answer.minimal_env(output) | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        timeout=300,
    )
    require(code == 0, "the tokenizer could not be read")
    record = judged(json.loads(text.strip().splitlines()[-1]))
    record["environment"] = {k: versions[k] for k in ("transformers", "tokenizers", "lock_file")}
    return record


def main():
    (ROOT / "runs").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="flashprove-tinyllama-request-", dir=ROOT / "runs"))
    print("Outputs: " + str(output), flush=True)
    record = check_text(output) | dict(
        passed=True, schema_id="hallmark.flashprove-tinyllama-request/1"
    )
    write_json(output / "result.json", record)
    print(json.dumps({k: record[k] for k in ("passed", "prompt_tokens", "answer_tokens")}))


if __name__ == "__main__":
    main()
