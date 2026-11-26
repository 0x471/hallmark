"""The request's text against its ids: the judge alone; no tokenizer is read here."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flashprove_tinyllama_request as request_text

CASE = request_text.CASE


def said(**changes):
    return dict(prompt_ids=CASE["prompt_ids"], response=CASE["response"], eos_token_id=2) | changes


def test_the_cases_own_text_passes_and_names_the_pinned_files():
    record = request_text.judged(said())
    assert (record["prompt_tokens"], record["answer_tokens"], record["end_token"]) == (25, 8, 2)
    assert set(record["tokenizer_files"]) == set(request_text.FILES)
    assert all(
        record["tokenizer_files"][name] == CASE["model"]["files"][name]["sha256"]
        for name in request_text.FILES
    )


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (dict(prompt_ids=CASE["prompt_ids"][:-1]), "tokenizer or the prompt"),
        (dict(prompt_ids=[*CASE["prompt_ids"][:-1], 0]), "tokenizer or the prompt"),
        (dict(response=CASE["response"] + " "), "decode to another text"),
        (dict(eos_token_id=0), "end token"),
    ],
)
def test_another_tokenizer_s_answer_is_refused(change, why):
    with pytest.raises(ValueError, match=why):
        request_text.judged(said(**change))
