"""The pinned tokenizer and its chat template on a case's text, as SASH's preparation applies them
(research/probes/sash/prepare_native.py). Prints what they give; the caller compares.

usage: python request_text.py MODEL_DIR CASE_JSON   (in the kit's `scales` environment)
"""

import json
import sys

from transformers import AutoTokenizer

case = json.loads(open(sys.argv[2]).read())
tokenizer = AutoTokenizer.from_pretrained(
    sys.argv[1], local_files_only=True, trust_remote_code=False
)
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": case["prompt"]}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)["input_ids"][0]
print(
    json.dumps(
        dict(
            prompt_ids=tokens.tolist(),
            response=tokenizer.decode(case["answer_ids"], skip_special_tokens=True),
            eos_token_id=tokenizer.eos_token_id,
        )
    )
)
