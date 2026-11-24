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
  its standard form carries the gate's clamp `gcl 1`, which option 3 cannot have at NE 0 and
  replaces with `eg 7` (below). If the owner prefers that, it is the fallback. Lean covers neither
  route (`lean/GAPS.md`: one model size).
- **Option 3, a patched FlashProve: proposed**, as Hallmark's own small patches on the pinned source.
  They are applied with `git apply`, as the SASH patches are, and each is SHA-256-pinned. It is not a
  dependency on a fork. At the SiLU key's default exponent (eg 8) `cp_commit`'s mandatory
  worst-case check would refuse the TinyLlama kit at one gate row; the kit therefore has the line
  `eg 7`, which `verifier.patch` admits at NE 0, and with it every site fits (below).

**Decision (proposed).** Adopt the four patches in
[`research/probes/flashprove/tinyllama/`](../../research/probes/flashprove/tinyllama/README.md):

- **`verifier.patch`** lets a dense model (NE 0) reach the standard parameter form:
  - it takes the digests of the three tables that a dense model forms (no `tsig`), only at NE 0;
  - it does not ask `routed-experts` or `both-clamps` at NE 0;
  - at NE 0 the line `eg 7` (the SiLU table's key in steps of 2^-7) meets `no-range-words`; no
    other `eg` and no `smax` does, and above NE 0 nothing changes;
  - no check of a proof's bytes changes; only admission does.
- **`prover.patch`**:
  - adds the fold constants for width 2,048 and head width 64 (`0x3FB504F3`, `0x3E3504F3`). Their
    rule, RN32(sqrt(d) / 2^t), is Hallmark's generalisation of upstream's RN32(sqrt(d) / 64): the
    same from width 4,096 to 16,383, twice it at 2,048, so that the fit needs no change. It also makes `cp_commit` check the model file's width, head width and expert
    layers against the parameter file, as `cp_prove` already does;
  - adds the same three-digest form to the provers' parameter-file reader;
  - leaves `cp_commit`'s worst-case check as upstream's: mandatory. With the kit's `eg 7` the check,
    computed on a CPU, passes (below).
- **`tools.patch`** gives `weight_sums` and upstream's CPU integer model the same constants and dense
  form.
- **`scales.patch`** (pinned in `kit.json`) lets upstream's scale tools fit a dense `llama` model:
  `gguf_weights.py` reads it (dense only), `policy_tables.py` writes its three tables' digests,
  `words_to_policy.py` takes the rotary base 10,000, and `make_calibration_set.py --tinyllama` makes
  upstream's calibration set with TinyLlama's tokenizer and start token.

With these, `kit/` holds a TinyLlama-1.1B kit (T = 64, e_res 11, `eg 7`) fitted by upstream's own
route, with `fit_eg.py` for the fit at `eg 7`, from a GGUF file that `export_gguf.py` writes from the
published safetensors (BF16 bits unchanged).

Decision 0093 and `make verify-flashprove` stay as they are. The patched verifier is a second build
with its own source digest, `33e9d53ef4640823782db51318e0024847ef9600ea8d15256e13d5fbc23fbbd1`.
SASH stays Hallmark's LLM proof path, and its code is unchanged. A switch to FlashProve needs it to
match SASH first and gets its own decision (#186).

**What a patched `YES` means.** Upstream's proof checks passed on a dense model's statement under
parameters that upstream itself does not admit to `YES`. NE 0 is among the forms its documents name
as never accepted, and the pinned repository ships no NE 0 FlashProve kit or proof. Without the gate's clamp (`gcl`, refused at
NE 0), the gate's key is range-checked rather than clamped; the shipped provers make no proof for a
key above 32,639. At NE 0 the key may be one bit coarser (`eg 7`) than upstream's standard form
allows. With a public seed nothing is hidden. The dense form
is chosen by NE alone: as upstream's, a `YES` is about the enrolled integer commitment and the caller's
request and range, not about which checkpoint was converted, the tokenizer, or SASH's integer function.

**Checks so far.** `make verify-flashprove-tinyllama` (CPU) checks the pinned patches and archive,
builds both verifiers and requires:

- **Upstream inputs.** On upstream's stored proofs and 99 altered copies, every one of 108 calls gives
  the same result and output from both builds, apart from the build digest and cp_verify's timed
  fields.
- **Standard-form test.** Upstream's test, with the dense cases (among them `eg 7` met at NE 0,
  `eg 6` and `smax` not), passes on the patched build. On the unpatched build it fails exactly the
  eight cases that only the patch meets; upstream's own `eg 7` case, with experts, still misses
  `no-range-words`.
- **Provers' reader.** It admits the dense file with three digests and refuses the forms the verifier
  refuses.
- **Constants.** The four implementations' constants agree with an exact derivation.
- **`weight_sums`.** It matches upstream's exact oracle at both widths.
- **nvcc.** Outside the target, upstream's `prover/build.sh` built `serving_cp_commit` and
  `serving_cp_prove`, and the test builds of both, from the patched source with nvcc 13.2.86 on
  Linux, without a GPU. No program has run.
- **TinyLlama's shape, planning only.** A dense parameter file at TinyLlama-1.1B's shape with
  `eg 7`, with placeholder scales, passes the patched planner in the standard parameter form; the
  unpatched build refuses it for its three table digests. Without those digests, the unpatched
  build finds exactly `routed-experts`, `both-clamps`, `table-digests` and `no-range-words` unmet
  there, and the patched build only `table-digests`: at this shape the patch changes admission and
  nothing else that the planner sees.

`make verify-flashprove-tinyllama-answer` (CPU, downloads the pinned model) rebuilds the GGUF file
and the kit from pinned inputs (every file byte for byte but `maxima.json`, which is device-dependent
and recorded only), in environments from hash-locked requirement files, checks the kit with both planners and the provers'
reader, runs upstream's scale-tool tests on the patched tree and a float reference against
transformers, and then requires **the decisive check**: upstream's integer model, with the kit and
every file pinned, gives SASH's eight answer tokens after SASH's 25 prompt tokens, over a whole
64-token window with no failed range condition and the gate's key at most 32,639, with its logits
following the float logits; two one-bit mutations of the kit must each fail that check; and
`cp_commit`'s worst-case check, computed on a CPU, must find every site within its limit. It
passed on 2026-11-24 on Linux (x86_64, every input downloaded) and on macOS (arm64), with the
integer model's results identical on the two
([evidence](../../research/evidence/flashprove-tinyllama-answer-2026-11-24.json)).

The current revision of the patches passed `make verify-flashprove-tinyllama` on 2026-11-24 on
Linux (upstream `verifier/build.sh`, g++ 13.3) and on macOS (Apple clang 17, built as
`make verify-flashprove` builds it), and upstream's `integer_model` CPU suite passed on the patched
tree on Linux (947 passed, 0 failed)
([evidence](../../research/evidence/flashprove-tinyllama-2026-11-24.json)). Apart from upstream's
stored proofs and their altered copies, which both builds check, these are planning, parsing and
table checks: no TinyLlama proof exists or is checked. The answer check runs the integer model, not a
prover, on one request. Further one-off checks and their limits:
[README](../../research/probes/flashprove/tinyllama/README.md#checks).

**`cp_commit`'s worst-case check, on a CPU.** Upstream computes it on a GPU only. Step 7 of the
answer target computes it with Hallmark's port of upstream's `worst_case.cuh`
(`worst_case_cpu.py`, the decisive sites' rounding emulated exactly) and checks the gate from the
float weights alone (`gate_float_bound.py`); the two agree in every layer. At `eg 8` every site
fits but one ([evidence](../../research/evidence/flashprove-tinyllama-answer-2026-11-24.json),
`retained_failure`): the gate of layer 21, where one row of 5,632 can take a key of 47,481 against the
limit of 32,639 (a real input of 177.4; no other layer's gate key above 14,631). No change of the kit's
scales fixes this: in real units the bound is the row's own. Upstream's remedy, `gcl 1`, clamps the
key with the sigmoid table, which only a model with experts forms, and the verifier refuses it at
NE 0. Upstream's other remedy for a short gate is a coarser key: each step of `eg` below 8 halves
the key and raises the gate's limit by one bit (`scales/worst_case.py`; `fit_scales.py --eg`), but
any `eg` takes a file out of the standard parameter form (`no-range-words`). So `verifier.patch`
admits `eg 7` at NE 0, and the kit is fitted at `eg 7` with upstream's own fit (`fit_eg.py` calls
`fit_scales.fit()` as `calibrate.py` does, which at `eg 8` gives `calibrate.py`'s file byte for
byte). At `eg 7` every site fits: the gate of layer 21 can take a key of 23,741 against 32,639,
and the decisive check still gives SASH's tokens. The cost is the SiLU table's input resolution,
2^-7 in place of 2^-8. No check is left out; the alternatives were an opt-in that leaves the check
out, `gcl` at NE 0 (the sigmoid table for a dense model, a change of what the verifier checks in a
proof), and padded option 2.

The port has not been compared with a GPU run, and it does not cover the GPU's guards on its matrix
products ([README](../../research/probes/flashprove/tinyllama/README.md#cp_commits-worst-case-check-on-a-cpu)).

**Not done; each needs its own step.**

1. **The calibration text's licence**: WikiText-2 (CC BY-SA; its dataset card's tags name 3.0 and
   GFDL, its text 4.0), as in upstream's rule. No text or token id from it is committed, but the
   kit's scales are computed from it. The card's tags and its text name different versions, and
   `research/AUTHORITY.json` excludes ambiguous licences, so the owner decides; a no means refitting
   on another text. The patches do not depend on it.
2. **GPU jobs, only with owner approval and a ceiling**: commit and prove (`cp_prove` x3, as #187
   asks), with a controller like `make e2e-flashprove`'s, not yet written. One attempt is one cpu-xl
   build of at most 3,600 s and one H200 job of at most 3,600 s for the commit and the three proofs.
   By 0093's cost formula (`planned_cost`, which counts HF's 600 s startup window per phase) at the
   HF rates recorded on 2026-10-20, cpu-xl USD 0.016667 and H200 USD 0.083333 a minute
   ([evidence](../../research/evidence/w13-compute-options-2026-10-20.json); they give 0093's
   USD 3.27), an attempt costs at most USD 7.02, and a second H200 job that reuses the build, as
   0093's did after its first prove job failed, at most USD 5.85. 0093's controller recomputes the
   bound at live prices before it starts. The times are ceilings, not estimates, since no prover
   program has run on this model. The ceiling, and with it the number of attempts, is the owner's.
   The prover patch compiles with nvcc 13.2.86 on a host without a GPU. `cp_commit`'s own verdict
   on the worst-case check (on a CPU, every site fits at `eg 7`) and `cp_prove`'s rules beyond its
   parameter-file reader run only there.
3. **A dense proof's altered copies.** The 108 equal calls are all on MoE proofs, and no proof has
   gone through the verifier's NE 0 branches. Before a dense `YES` is relied on, run an
   altered-copies table (as upstream's `examples/rejected-proofs`) against the first dense proof.
4. **#187's proof-system lines.** The kit keeps check 6's lines: placeholders set from the verifier's
   rules, not sized for speed, which `cp_prove` has judged only by its parameter-file reader; a kit
   for each number of checked tokens in #187's sweep.

**Not claimed.** No dense proof exists. There is no claim of soundness of the patched C++ or CUDA code,
no Lean coverage, no hiding, and no like-for-like numbers yet. Equal tokens are a smoke test, not a
proof that the two integer functions are equal; #187's like-for-like section must name each
backend's model, conversion, request, window, range and statement.

**Prior work.** `marcellobardus/FlashProve-TinyLlama1.1B` at `1a94f01` (Apache-2.0) reports two
single-token proofs of a dense model at its own shape, by its own logs, which Hallmark has not
reproduced. Hallmark uses none of its code,
proofs, kits, seeds or commitments.
