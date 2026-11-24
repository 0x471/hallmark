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
  - it needs the same new scale tools as option 3 (upstream's refuse rotary base 10,000), and a
    padding exporter on top.

  What option 2 keeps is upstream's unchanged verifier, so its `YES` would be upstream's own, and
  its standard form carries the gate's clamp `gcl 1`, which option 3's kit lacks (below). If the
  owner prefers that, it is the fallback. Lean covers neither route (`lean/GAPS.md`: one model
  size).
- **Option 3, a patched FlashProve: proposed**, as Hallmark's own small patches on the pinned source.
  They are applied with `git apply`, as the SASH patches are, and each is SHA-256-pinned. It is not a
  dependency on a fork. With upstream's mandatory worst-case check `cp_commit` would not commit
  the TinyLlama kit (below), so `prover.patch` also adds an opt-in `-without-worst-case`, for
  public weights only.

**Decision (proposed).** Adopt the four patches in
[`research/probes/flashprove/tinyllama/`](../../research/probes/flashprove/tinyllama/README.md):

- **`verifier.patch`** lets a dense model (NE 0) reach the standard parameter form:
  - it takes the digests of the three tables that a dense model forms (no `tsig`), only at NE 0;
  - it does not ask `routed-experts` or `both-clamps` at NE 0;
  - no check of a proof's bytes changes; only admission does.
- **`prover.patch`**:
  - adds the fold constants for width 2,048 and head width 64 (`0x3FB504F3`, `0x3E3504F3`). Their
    rule, RN32(sqrt(d) / 2^t), is Hallmark's generalisation of upstream's RN32(sqrt(d) / 64): the
    same from width 4,096 to 16,383, twice it at 2,048, so that the fit needs no change. It also makes `cp_commit` check the model file's width, head width and expert
    layers against the parameter file, as `cp_prove` already does;
  - adds the same three-digest form to the provers' parameter-file reader;
  - keeps `cp_commit`'s worst-case check upstream's default and adds the opt-in
    `-without-worst-case`: `cp_commit` takes it only with `-public-weights`, in the first form, and
    with neither `-worst-case` nor `-worst-case-table`, and then leaves the check out, so `isk.txt`
    gets no line `ranges`; `cp_prove` serves such a commitment only with its own
    `-without-worst-case`, which it takes only with `-public-weights` (so under "seed public").
    Without the option both programs behave as upstream's. (An earlier revision left the check out
    by default under `-public-weights`, which changed upstream's default for every such commitment;
    it was dropped.)
- **`tools.patch`** gives `weight_sums` and upstream's CPU integer model the same constants and dense
  form.
- **`scales.patch`** (pinned in `kit.json`) lets upstream's scale tools fit a dense `llama` model:
  `gguf_weights.py` reads it (dense only), `policy_tables.py` writes its three tables' digests,
  `words_to_policy.py` takes the rotary base 10,000, and `make_calibration_set.py --tinyllama` makes
  upstream's calibration set with TinyLlama's tokenizer and start token.

With these, `kit/` holds a TinyLlama-1.1B kit (T = 64, e_res 11) fitted by upstream's own route from a
GGUF file that `export_gguf.py` writes from the published safetensors (BF16 bits unchanged).
It is calibrated on WikiText-2, as upstream's rule does, at the revision upstream pins. Its licence
was reviewed by the research agent, to which `research/AUTHORITY.json` delegates source selection
and licence review, not by the owner: the text is 2016 Wikipedia, under CC BY-SA 3.0 and GFDL (the
dataset card's tags); the card's text names CC BY-SA 4.0, which CC BY-SA 3.0 section 4(b) allows
for adaptations, so the versions agree on what they permit and the licence is not ambiguous. No
text or token id from it is committed; the kit holds scale statistics computed from it, and
`kit.json` records the source and the review.

Decision 0093 and `make verify-flashprove` stay as they are. The patched verifier is a second build
with its own source digest, `6f7ac148a57bd03f24bd029ba41c7001772cc83578e80b7bd2ef25a13dee3a52`.
SASH stays Hallmark's LLM proof path, and its code is unchanged. A switch to FlashProve needs it to
match SASH first and gets its own decision (#186).

**What a patched `YES` means.** Upstream's proof checks passed on a dense model's statement under
parameters that upstream itself does not admit to `YES`. NE 0 is among the forms its documents name
as never accepted, and the pinned repository ships no NE 0 FlashProve kit or proof. Without the gate's clamp (`gcl`, refused at
NE 0), the gate's key is range-checked rather than clamped; the shipped provers make no proof for a
key above 32,639. With a public seed nothing is hidden. The dense form
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

`make verify-flashprove-tinyllama-answer` (CPU, downloads the pinned model) rebuilds the GGUF file
and the kit from pinned inputs (every file byte for byte but `maxima.json`, which is device-dependent
and recorded only), in environments from hash-locked requirement files, checks the kit with both planners and the provers'
reader, runs upstream's scale-tool tests on the patched tree and a float reference against
transformers, and then requires **the decisive check**: upstream's integer model, with the kit and
every file pinned, gives SASH's eight answer tokens after SASH's 25 prompt tokens, over a whole
64-token window with no failed range condition and the gate's key at most 32,639, with its logits
following the float logits; two one-bit mutations of the kit must each fail that check.
It passed on 2026-11-23 on Linux (x86_64, every input downloaded) and on macOS (arm64); both rebuilt
every required file byte for byte, and the integer model's choices, peaks, margins and clamped cells are identical on the two
([evidence](../../research/evidence/flashprove-tinyllama-answer-2026-11-23.json)).

The current revision of the patches passed `make verify-flashprove-tinyllama` on 2026-11-23 on Linux (upstream `verifier/build.sh`, g++ 13.3) and
on macOS (Apple clang 17, built as `make verify-flashprove` builds it;
[evidence](../../research/evidence/flashprove-tinyllama-2026-11-23.json)). Upstream's `integer_model` CPU suite passed on the patched tree on 2026-11-22 (947, 0 failed), on hunks that have not changed since. Apart from upstream's
stored proofs and their altered copies, which both builds check, these are planning, parsing and
table checks: no TinyLlama proof exists or is checked. The answer check runs the integer model, not a
prover, on one request. Further one-off checks and their limits:
[README](../../research/probes/flashprove/tinyllama/README.md#checks).

**`cp_commit`'s worst-case check, on a CPU.** Upstream computes it on a GPU only. Step 7 of the
answer target computes it with Hallmark's port of upstream's `worst_case.cuh`
(`worst_case_cpu.py`, the decisive sites' rounding emulated exactly) and checks the gate from the
float weights alone (`gate_float_bound.py`); the two agree in every layer. Every site fits but one:
the gate of layer 21, where one row of 5,632 can take a key of 47,481 against the limit of 32,639
(a real input of 177.4; no other layer's gate key above 14,631). No change of the kit's
scales fixes this: in real units the bound is the row's own. Upstream's remedy, `gcl 1`, clamps the
key with the sigmoid table, which only a model with experts forms, and the verifier refuses it at
NE 0. So the TinyLlama commitment is made with `-public-weights -without-worst-case`:

- upstream gives weight hiding as the check's reason (`scales/worst-case-bounds.md`), and under
  `-public-weights` nothing is hidden; soundness does not rest on the check: every proof shows
  again that its values lie in the model's ranges;
- what it costs is availability: a request that drives that gate row past its range ends `failed`
  in `cp_prove`, with no proof. On SASH's prompt the key reaches at most 7,108;
- the alternatives were larger: `gcl` at NE 0 needs the sigmoid table for a dense model and so
  changes what the verifier checks in a proof; padded option 2, whose standard form carries
  `gcl 1` (limit 4,194,175), measures a padded model of about 11.4 billion parameters.

`make verify-flashprove-tinyllama` checks the option's rules on the host from the patched sources:
`cp_commit`'s in a serving and a test build, and on Linux `cp_prove`'s, each refusal with a control
that differs only by the option.

The port has not been compared with a GPU run, and it does not cover the GPU's guards on its matrix
products ([README](../../research/probes/flashprove/tinyllama/README.md#cp_commits-worst-case-check-on-a-cpu)).

**Not done; each needs its own step.**

Done since the first draft: the scale tools for TinyLlama, the kit, a pinned GGUF export (our own
`export_gguf.py` on the `gguf` package that upstream pins, in place of llama.cpp's converter) and the
decisive free check above. Equal tokens are a smoke test, not a proof that the two integer functions
are equal; #187's like-for-like section must name each backend's model, conversion, request, window,
range and statement.

1. **GPU jobs, only with owner approval and a cap** (#186): commit with `-public-weights
   -without-worst-case` and prove (`cp_prove` x3, as #187 asks). The prover
   patch compiles with nvcc 13.2.86 on a host without a GPU. `cp_commit`'s own verdict on the
   worst-case check and `cp_prove`'s rules beyond its parameter-file reader run only there.
2. **A dense proof's altered copies.** The 108 equal calls are all on MoE proofs, and no proof has
   gone through the verifier's NE 0 branches. Before a dense `YES` is relied on, run an
   altered-copies table (as upstream's `examples/rejected-proofs`) against the first dense proof.
3. **#187's proof-system lines.** The kit keeps check 6's lines: placeholders set from the verifier's
   rules, not sized for speed, which `cp_prove` has judged only by its parameter-file reader; a kit
   for each number of checked tokens in #187's sweep.

**Not claimed.** No dense proof exists. There is no claim of soundness of the patched C++ or CUDA code,
no Lean coverage, no hiding, and no like-for-like numbers yet.

**Prior work.** `marcellobardus/FlashProve-TinyLlama1.1B` at `1a94f01` (Apache-2.0) reports two
single-token proofs of a dense model at its own shape, by its own logs, which Hallmark has not
reproduced. Hallmark uses none of its code,
proofs, kits, seeds or commitments.
