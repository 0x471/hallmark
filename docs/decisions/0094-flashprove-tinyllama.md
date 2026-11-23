# 0094: TinyLlama on pinned FlashProve, padded into its MoE format

Date: 2026-11-21, revised 2026-11-23. Status: **proposed**. No proof, GPU run or paid compute so
far.

**Context.** #186 asks how Hallmark's LLM case, dense TinyLlama-1.1B, can be proved with FlashProve,
which decision 0093 pins at `9e2bdd8` and drives unchanged ("No upstream source file is changed").
FlashProve reads real weights only at Llama 4 Maverick's widths and proves MoE models only. #187 asks
for "comparable LLM statements" and a like-for-like section in
[PROVER-COMPARISON](../PROVER-COMPARISON.md): the same model under both provers. #186 lists three
paths and makes the third, a fork or extension of FlashProve, the last resort that needs owner
approval.

- **Option 1, a small public MoE model: rejected.** The unchanged prover reads real weights only at
  width 5,120 and head width 128 (`prover/weights_blocks/gguf_source.cuh`), in Llama 4's MoE layout:
  with experts, every odd layer is an MoE layer with one routed expert per token
  (`verifier/docs/parameter-file.md`). The models known here in that layout are Meta's Llama 4, which
  #186 rules out (gated weights). A different model would not be like-for-like either.
- **Option 2, TinyLlama as an MoE on unchanged upstream: proposed, in a padded form.** As #186 words
  it, one expert, it fails: NE is 0 or a power of two from 2 (`verifier/docs/parameter-file.md`). With
  four experts it works: TinyLlama zero-padded to width 5,120 and head width 128; on the odd layers
  its feed-forward block becomes the shared expert, beside four zero routed experts and a zero
  router.
- **Option 3, Hallmark's own patches for dense layers: tested, not adopted.** Four small pinned
  patches let the pinned source take TinyLlama at its own shape, and on a CPU upstream's integer model
  with a kit fitted through them gives SASH's answer. But `cp_commit`'s mandatory worst-case check
  refuses that kit (below), and the only ways past it are to leave an upstream check out or a much
  larger patch. The patches and their checks stay in
  [`research/probes/flashprove/tinyllama/`](../../research/probes/flashprove/tinyllama/README.md) as
  the record of this test.

**Decision (proposed).** Prove TinyLlama-1.1B-Chat, at the revision of Hallmark's SASH path, padded
into FlashProve's MoE format, with FlashProve `9e2bdd8` unchanged, as decision 0093 drives it:

- **No upstream source file changes.** The verifier, `cp_commit` and `cp_prove` are upstream's, so a
  `YES` is upstream's own. That is decision 0093's rule and #186's option 2.
- **Every upstream check runs.** None of Hallmark's patches to SASH drops a check without putting
  another in its place: they fix faults or add guards
  ([`research/probes/sash/`](../../research/probes/sash/README.md)). Among the checks is `cp_commit`'s
  worst-case check: the standard parameter form carries the gate's clamp `gcl 1`, whose limit (a key
  of 4,194,175) is far above the gate row that stops option 3 (in real units the same row, a key of
  roughly 47,000).
- **Hallmark's own offline tools** make the padded file and its kit: a padding exporter, and a kit
  fitter for the padded file, since upstream's scale fitter is Maverick-specific (`llama4` files, no
  rotation on every fourth layer, Llama 4's float pass) and refuses TinyLlama's rotary base 10,000.
  They change no upstream file.

What this costs:

- every #187 number describes the padded model: by its shapes about 11.4 billion committed
  parameters, mostly zeros, against TinyLlama's 1.1 billion, and about seven times the
  multiplications per token (computed from the shapes, not measured);
- the CPU checks of a padded kit (the integer model, the worst-case check) hold about 23 GB of
  integer weights, more than the hosts used so far;
- the padding is part of the statement: a client must know that NE 4, the zero experts and the zero
  router are TinyLlama, padded.

Established so far, in a local CPU probe that is not in the repository: on a small TinyLlama-like
model with random weights the padded form gives the same logits (largest difference 3.1e-15) and
never picks a pad token; and a padded parameter file at TinyLlama's shape, with placeholder scales,
passes the unchanged verifier's planner in the standard parameter form and the provers' parameter
reader (NE 1 is refused with `shape-NE`, a wrong rotary base with `rotary-table`). Not yet: the
padded export, a fitted padded kit, its worst-case check, its integer-model answer, any commitment or
proof.

SASH stays Hallmark's LLM proof path, and its code is unchanged. A switch to FlashProve needs it to
match SASH first and gets its own decision (#186). Decision 0093 and `make verify-flashprove` stay as
they are. Lean covers neither route (`lean/GAPS.md`: one model size).

**Why not option 3.** Its probe ([README](../../research/probes/flashprove/tinyllama/README.md))
holds four patches on the pinned source: `verifier.patch` lets NE 0 reach the standard parameter form
(three table digests, no `routed-experts` or `both-clamps` at NE 0; no check of a proof's bytes
changes), `prover.patch` adds the fold constants for width 2,048 and head width 64 and keeps the
worst-case check mandatory, `tools.patch` gives `weight_sums` and the integer model the same, and
`scales.patch` lets upstream's scale tools fit a dense `llama` file. On a CPU:

- `make verify-flashprove-tinyllama`: upstream's stored proofs and 99 altered copies give the same
  108 results from the unpatched and the patched verifier (all MoE proofs), and a TinyLlama-shaped
  file passes the patched planner, on Linux and macOS at `1208545`
  ([evidence](../../research/evidence/flashprove-tinyllama-2026-11-23.json)). The prover patch
  compiles with nvcc 13.2.86; no program ran.
- `make verify-flashprove-tinyllama-answer`: the kit rebuilt byte for byte from pinned inputs, and
  upstream's integer model with it gives SASH's eight answer tokens on SASH's prompt over a 64-token
  window, with two one-bit mutations of the kit caught, on Linux and macOS at `7004750`
  ([evidence](../../research/evidence/flashprove-tinyllama-answer-2026-11-23.json)).
- **`cp_commit`'s worst-case check, computed on a CPU** by Hallmark's port of upstream's
  `worst_case.cuh` (`worst_case_cpu.py`, the deciding rows' rounding emulated exactly) and checked
  from the float weights alone (`gate_float_bound.py`), the two agreeing in every layer: every site
  fits but the gate of layer 21, where one row of 5,632 can take a key of 47,481 against the limit
  of 32,639 (a real input of 177.4; no other layer's gate key above 14,631; on SASH's prompt the key
  reaches 7,108). No change of the kit's scales fixes this: in real units the bound is the row's
  own. Upstream's remedy, `gcl 1`, reads the clamp from the sigmoid table, which only a model with
  experts forms, and the verifier refuses it at NE 0.

So option 3 could commit this kit only by leaving the worst-case check out (an opt-in under
`-public-weights`: upstream gives weight hiding as the check's reason, and soundness does not rest on
it, `scales/worst-case-bounds.md`; but no Hallmark patch has dropped a check without a
replacement), or by
making `gcl` work at NE 0, which changes what the verifier checks in a proof. Either way it leaves
0093's "unchanged upstream" rule, as #186's last resort. The port has not been compared with a GPU
run, and it does not cover the GPU's guards on its matrix products.

**What carries over.** From the probe: the GGUF export of the pinned model (`export_gguf.py`, BF16
bits unchanged, llama.cpp's query and key permutation), the calibration rule with TinyLlama's
tokenizer, the hash-locked environments, the integer-model answer check and the CPU worst-case port
(to be extended to layers with experts).

**Not done; each needs its own step.**

1. **The calibration text's licence**: upstream's rule calibrates on WikiText-2 (CC BY-SA; its
   dataset card's tags name 3.0 and GFDL, its text 4.0), as the probe's kit does. No text or token
   id from it is committed, but a kit's scales are computed from it. The card's tags and its text
   name different versions, and `research/AUTHORITY.json` excludes ambiguous licences, so the owner
   decides; a no means calibrating on another text.
2. **The padded route**: the padding exporter, the kit fitter, a fitted padded kit, its CPU
   worst-case check and integer-model answer check on SASH's request, on a host with room for them.
3. **GPU jobs, only with owner approval and a cap**: commit and prove (`cp_prove` x3, as #187 asks);
   `cp_commit`'s own verdict on the worst-case check runs only there. Then altered copies of the first
   proof, as decision 0093 did for its own.
4. **#187's proof-system lines**: a kit for each number of checked tokens in #187's sweep, sized for
   speed rather than as placeholders.

**Not claimed.** No TinyLlama proof exists, padded or dense. There is no claim of soundness of the
C++ or CUDA code, no Lean coverage, no hiding (the weights are public), and no like-for-like numbers
yet. Equal tokens are a smoke test, not a proof that FlashProve's and SASH's integer functions are
equal; #187's like-for-like section must name each backend's model, conversion, request, window,
range and statement.

**Prior work.** `marcellobardus/FlashProve-TinyLlama1.1B` at `1a94f01` (Apache-2.0) reports two
single-token proofs of a dense model at its own shape, by its own logs, which Hallmark has not
reproduced; its plan leaves `cp_commit`'s worst-case check out under `-public-weights`. Hallmark uses
none of its code, proofs, kits, seeds or commitments.
