# 0094: Dense TinyLlama on pinned FlashProve through Hallmark's own patches

Date: 2026-11-21. Status: **proposed**; needs owner approval
([#186](https://github.com/AbdelStark/hallmark/issues/186): option 3 "needs owner approval"). No
proof, GPU run or paid compute so far.

**Context.** #186 asks how Hallmark's LLM case, dense TinyLlama-1.1B, can be proved with FlashProve,
which decision 0093 pins at `9e2bdd8` and which reads real weights only at Llama 4 Maverick's widths. #187 asks for
"comparable LLM statements" and a like-for-like section in
[PROVER-COMPARISON](../PROVER-COMPARISON.md): the same model under both provers. The aim is to see whether FlashProve can take over the TinyLlama case from SASH: that needs FlashProve's numbers on TinyLlama itself, not on Maverick, which is far heavier than this demo needs. No fair comparison of the two provers exists yet; this work is meant to make one.

- **Option 1, a small public MoE model: rejected.** The unchanged prover reads real weights only at
  width 5,120 and head width 128 (`prover/weights_blocks/gguf_source.cuh`), in Llama 4's MoE layout:
  with experts, every odd layer is an MoE layer with one routed expert per token
  (`verifier/docs/parameter-file.md`). The models known here in that layout are Meta's Llama 4, which
  #186 rules out (gated weights). A different model would not be like-for-like either.
- **Option 2, TinyLlama as an MoE on unchanged upstream: rejected, though possible in a padded form.**
  As written it fails: NE 1 is refused, since NE is 0 or a power of two from 2
  (`verifier/docs/parameter-file.md`). TinyLlama zero-padded to width 5,120 with four zero experts is
  exact in float, and passes the unchanged verifier's planning check and the provers' parameter rules
  (a local CPU probe, not in the repository; no proof). It is rejected because:
  - every #187 number would describe the padded model: by its shapes about 11.4 billion committed
    parameters, mostly zeros, against TinyLlama's 1.1 billion, and about seven times the
    multiplications per token (computed from the shapes, not measured);
  - unchanged `cp_commit` always runs the worst-case check on real weights, and whether TinyLlama's
    scales pass it is not measured; if they fail, the scales must be roomier, at an accuracy cost
    nobody has measured;
  - it needs new scale tools anyway, since upstream's refuse rotary base 10,000, and a padding exporter
    on top.

  What it keeps is upstream's unchanged verifier, so its `YES` would be upstream's own. If the owner
  prefers that, it is the fallback. Lean covers neither route (`lean/GAPS.md`: one model size).
- **Option 3, a patched FlashProve: proposed**, as Hallmark's own small patches on the pinned source.
  They are applied with `git apply`, as the SASH patches are, and each is SHA-256-pinned. It is not a
  dependency on a fork.

**Decision (proposed).** Adopt the three patches in
[`research/probes/flashprove/tinyllama/`](../../research/probes/flashprove/tinyllama/README.md):

- **`verifier.patch`** lets a dense model (NE 0) reach the standard parameter form:
  - it takes the digests of the three tables that a dense model forms (no `tsig`), only at NE 0;
  - it does not ask `routed-experts` or `both-clamps` at NE 0;
  - no check of a proof's bytes changes; only admission does.
- **`prover.patch`**:
  - adds the fold constants for width 2,048 and head width 64 (`0x3FB504F3`, `0x3E3504F3`, by
    upstream's own rule), and makes `cp_commit` check the model file's width, head width and expert
    layers against the parameter file, as `cp_prove` already does;
  - adds the same three-digest form to the provers' parameter-file reader;
  - leaves out the worst-case check under `-public-weights` unless asked for. The check is a hiding
    safeguard on which soundness does not rest, and nothing is hidden under a public seed.
- **`tools.patch`** gives `weight_sums` and upstream's CPU integer model the same constants and dense
  form.

Decision 0093 and `make verify-flashprove` stay as they are. The patched verifier is a second build
with its own source digest, `6f7ac148a57bd03f24bd029ba41c7001772cc83578e80b7bd2ef25a13dee3a52`.
SASH stays Hallmark's LLM proof path, and its code is unchanged. A switch to FlashProve needs it to
match SASH first and gets its own decision (#186).

**What a patched `YES` means.** Upstream's proof checks passed on a dense model's statement under
parameters that upstream itself does not admit to `YES`. NE 0 is among the forms its documents name
as never accepted, and the pinned repository ships no NE 0 FlashProve kit or proof. Without the gate's clamp (`gcl`, refused at
NE 0), the gate's key is range-checked rather than clamped; the shipped provers make no proof for a
key above 32,639. With a public seed nothing is hidden, and
without the worst-case check a request may end `failed` where a value leaves its range. The dense form
is chosen by NE alone: as upstream's, a `YES` is about the enrolled integer commitment and the caller's
request and range, not about which checkpoint was converted, the tokenizer, or SASH's integer function.

**Checks so far.** `make verify-flashprove-tinyllama` (CPU) checks the pinned patches and archive,
builds both verifiers and requires:

- **Upstream inputs.** On upstream's stored proofs and 99 altered copies, every one of 108 calls gives
  the same result and output from both builds, apart from the build digest and cp_verify's timed
  fields.
- **Standard-form test.** Upstream's test, with the dense cases, passes on the patched build. On the
  unpatched build it fails exactly the five cases that only the patch meets.
- **Provers' reader.** It admits the dense file with three digests and refuses the forms the verifier
  refuses.
- **Constants.** The four implementations' constants agree with an exact derivation.
- **`weight_sums`.** It matches upstream's exact oracle at both widths.
- **nvcc.** Outside the target, upstream's `prover/build.sh` built `serving_cp_commit` and
  `serving_cp_prove`, and the test builds of both, from the patched source with nvcc 13.2.86 on
  Linux, without a GPU. No program has run.
- **TinyLlama's shape, planning only.** A dense parameter file at TinyLlama-1.1B's shape, with
  placeholder scales, passes the patched planner in the standard parameter form; the unpatched build
  refuses it for its three table digests. Without those digests, the unpatched build finds exactly
  `routed-experts`, `both-clamps` and `table-digests` unmet there, and the patched build only
  `table-digests`: at this shape the patch changes admission and nothing else that the planner sees.

The current revision passed it on 2026-11-22 on Linux (upstream `verifier/build.sh`, g++ 13.3) and
on macOS (Apple clang 17, built as `make verify-flashprove` builds it;
[evidence](../../research/evidence/flashprove-tinyllama-2026-11-22.json)). Upstream's `integer_model` CPU suite passes on the patched tree (947, 0 failed). Apart from upstream's
stored proofs and their altered copies, which both builds check, these are planning, parsing and
table checks: no TinyLlama proof exists or is checked. At TinyLlama's 2,048/64 shape the planner runs
on placeholder scales only, and nothing yet runs through `integer_model`'s GGUF reader. Further one-off
checks and their limits:
[README](../../research/probes/flashprove/tinyllama/README.md#checks).

**Not done; each needs its own step.**

1. **`scales/` for TinyLlama**: a dense, `llama`-file converter (today `llama4` MoE only), digest
   lines at NE 0 (`policy_tables.py` refuses them), theta 10,000, and the shifts for t = 5. Then a
   planning check with fitted scales and a dense GGUF through `integer_model`.
2. **The kit**: a dense parameter file and scale file at T = 64, with a calibration text whose
   licence is reviewed.
3. **A pinned GGUF export** that states `llama.attention.key_length 64`. llama.cpp's converter is a new
   source to review.
4. **The decisive free check.** Upstream's integer model, run on the CPU on Hallmark's 25 prompt
   tokens, must give SASH's eight answer tokens. No GPU money before this passes. Equal tokens are a
   smoke test, not a proof that the two integer functions are equal; #187's like-for-like section must
   name each backend's model, conversion, request, window, range and statement.
5. **GPU jobs, only with owner approval and a cap**: commit and prove (`cp_prove` x3, as #187 asks).
   The prover patch already compiles with nvcc 13.2.86 on a host without a GPU.

**Not claimed.** No dense proof exists. There is no claim of soundness of the patched C++ or CUDA code,
no Lean coverage, no hiding, and no like-for-like numbers yet.

**Prior work.** `marcellobardus/FlashProve-TinyLlama1.1B` at `1a94f01` (Apache-2.0) reports two
single-token proofs of a dense model at its own shape, by its own logs, which Hallmark has not
reproduced. Hallmark uses none of its code,
proofs, kits, seeds or commitments.
