# FlashProve proofs

[FlashProve](https://github.com/anthropics/FlashProve) is Anthropic's research
prototype for zero-knowledge proofs of LLM inference
([announcement](https://www.anthropic.com/research/provable-inference)). A provider
commits to a Mixture-of-Experts model's weights once. For each request, its GPU
then proves that every checked token was the committed model's greedy choice.
Anyone can check the proof on a CPU in seconds, without the weights.

Hallmark drives FlashProve as a pinned external backend: upstream commit
`9e2bdd8` (v0.16.0, Apache-2.0), unchanged, called through command-line adapters
([decision 0093](decisions/0093-flashprove-backend.md)). FlashProve proves MoE
models only, so Hallmark's dense TinyLlama response stays on
[SASH](LLM-PROOF.md). [SASH and FlashProve compared](PROVER-COMPARISON.md) sets
the two side by side. Proposed, not adopted:
[decision 0094](decisions/0094-flashprove-tinyllama.md) adds small pinned
patches so that the same FlashProve can take dense TinyLlama-1.1B. Four dense proofs
of one request, SASH's, have been made with them on a rented H100, at the contributor's
own cost and with no owner-approved budget; only the patched verifier accepts them
([details](../research/probes/flashprove/tinyllama/README.md#the-first-dense-proof)).

| Command | What it does | Needs |
|---|---|---|
| `make verify-flashprove` | Builds `cp_verify`, checks the two stored upstream proofs, runs 99 rejection cases | Mac or Linux, a C++17 compiler, about 5 GB RAM, no GPU, no spend |
| `make verify-flashprove-tinyllama` | Proposed (decision 0094): builds the unpatched and the patched `cp_verify` and checks the TinyLlama patches against upstream's proofs and tests ([details](../research/probes/flashprove/tinyllama/README.md#checks)) | Mac or Linux, a C++17 compiler, git, about 5 GB RAM and 1 GB of disk, no GPU, no spend |
| `make verify-flashprove-tinyllama-answer` | Proposed (decision 0094): rebuilds TinyLlama-1.1B's GGUF file and fitted kit from pinned inputs (every required file byte for byte), in environments from hash-locked requirement files, plans them with both verifiers, runs the part of `cp_prove`'s set-up that needs only the kit on a CPU, and requires that upstream's integer model gives SASH's eight answer tokens on SASH's prompt; and that `cp_commit`'s worst-case check, computed on a CPU, finds every site within its limit ([details](../research/probes/flashprove/tinyllama/README.md#the-tinyllama-kit-and-the-answer-check)) | macOS arm64 or Linux x86_64, a C++17 compiler, git, `uv`, about 8 GB of disk; downloads the pinned model (2.2 GB), the upstream archive and the locked Python packages; no GPU, no spend |
| `python scripts/flashprove_tinyllama_proof.py --case DIR --commitment-id ID` | Proposed (decision 0094): checks a dense TinyLlama proof from [`gpu_job.sh`](../research/probes/flashprove/tinyllama/gpu_job.sh) with caller-held values under the patched `cp_verify`, requires the unpatched one's refusal and, of eight altered copies, each one's own refusal | The public files of a GPU run (the first proof's: [download and commands](../research/probes/flashprove/tinyllama/README.md#to-repeat-the-callers-check)); Mac or Linux, a C++17 compiler, git, about 3 GB RAM; no GPU, no spend |
| `make verify-flashprove-tinyllama-request` | Proposed (decision 0094): requires that the pinned tokenizer and chat template give the request's token ids for the case's prompt, that the answer ids decode to the case's response and that the answer ends at the end token | macOS arm64 or Linux x86_64, `uv`; no GPU, no spend |
| `make verify-flashprove-tinyllama-proof FLASHPROVE_TINYLLAMA_CASE=DIR` | Proposed (decision 0094): the same check of a dense proof as the script above, under the enrolled commitment ID. The free local verify target of #186 | As the script above |
| `make e2e-flashprove-tinyllama GPU_CREDIT_BALANCE_USD=… FLASHPROVE_TINYLLAMA_BUILD=… FLASHPROVE_TINYLLAMA_GGUF=…` | Proposed (decision 0094): rents one sm_90 host on RunPod, commits to and proves dense TinyLlama there under a caller-set cap of USD 3.00, deletes the host and rechecks the proof here. The cap is not owner-approved, RunPod is not a provider of `research/AUTHORITY.json`, and the provers must be built beforehand ([details](../research/probes/flashprove/tinyllama/README.md#the-capped-run)) | `RUNPOD_API_KEY`, prepaid credit, built programs, the exported model; about USD 0.40 a run; planned worst case USD 3.00, more if the host cannot be deleted |
| `make e2e-flashprove GPU_CREDIT_BALANCE_USD=…` | Builds the prover, commits to and proves the small synthetic kit on one HF H200, rechecks the proof here | `HF_TOKEN`, prepaid HF credit; worst case USD 3.27 |
| `make compare-provers FLASHPROVE_RUN=…` | Re-times `cp_verify` on three proofs and gathers the recorded SASH and FlashProve figures | A passed `e2e-flashprove` run; no spend |

## Verify stored proofs

```sh
make verify-flashprove
```

The command downloads the pinned 158 MB archive and checks its SHA-256 and size.
It then builds `cp_verify` and checks two proofs that ship with upstream:

- `accepted-proof`: a small model, 3 checked tokens and the next token;
- `full-size`: a model of Llama 4 Maverick's shape, 1,001 tokens in one proof.

Both use synthetic weights. Every expected value comes from Hallmark's
[pin file](../research/probes/flashprove/source.json), not from the proof
directory: commitment ID, range and next token. Then upstream's three tables of
altered copies run, and each of the 99 copies must be rejected by the check
the table names.

On Linux the build is upstream `verifier/build.sh`. On macOS, Hallmark compiles
the same source with Apple clang, using the same macros and source digest, and
labels the result `apple-clang`. Results go to a fresh `runs/flashprove-verify-*`
directory.

On an Apple M4 Max, raw `cp_verify` takes 0.62 s on the small proof and 7.88 s
with a 4.23 GB peak on the full-size one (medians of five calls).

## Prove on a GPU

```sh
make e2e-flashprove GPU_CREDIT_BALANCE_USD=31
```

Give the balance you checked yourself. The command refuses to start if the
balance is below the worst case, or if another HF job is active. It runs the
free local suite first, then two jobs, one attempt each:

1. **Build** (cpu-xl, at most 3,600 s): upstream `verifier/build.sh` and
   `prover/build.sh` with CUDA 13.2.2 (nvcc V13.2.86), as an ordinary account.
2. **Prove** (H200, at most 900 s): upstream `run_target_example.sh
   target_small_f2_roomy`. It commits to synthetic weights, proves one 64-token
   request (15 prompt tokens, 48 of the model's own, plus the next token) and
   runs Linux `cp_verify`.

Only public files come back. The secret key directory stays on the worker.
Hallmark then checks that the configuration and request match the pinned kit
byte for byte. It takes the commitment ID that `cp_commit` printed, verifies the
proof on this machine and requires six altered copies to be rejected.

## What a pass means

A pass means `cp_verify` accepted a proof of this statement: under the integer
model committed under this ID, each checked token is the greedy choice. Here the
models are synthetic. A pass says nothing about a served model, about Llama 4
Maverick or about weight hiding. The `bits=` figures are the verifier's own
accounting, not a security level. Upstream's Lean soundness theorem covers the
Lean verifier, not `cp_verify` and not the CUDA prover.
