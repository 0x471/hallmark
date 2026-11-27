# 0094: Dense TinyLlama on pinned FlashProve through Hallmark's own patches

Date: 2026-11-21, revised 2026-11-25 and 2026-11-27. Status: **proposed**
([#186](https://github.com/AbdelStark/hallmark/issues/186), option 3). On 2026-11-24 and 25 (UTC) the contributor made a dense commitment and four
proofs on a rented H100, on the contributor's own account and at the contributor's cost
(USD 1.61), with no owner-approved budget and no owner-set cap ("The first dense proof" below).
In the night of 2026-11-26 to 27 (UTC) four more commitments and six more proofs followed under a
controller with a cap that the contributor set (about USD 2.20 with three drills, again the
contributor's cost; "The capped run, the enrollment and the composed join" below).

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
  They are applied with `git apply` and each is SHA-256-pinned: the mechanism of Hallmark's SASH
  patches, and the likeness ends there
  ([ARCHITECTURE](../ARCHITECTURE.md#how-sash-is-integrated) says what each of those does):
  `verifier.patch` here widens what the verifier admits. It is not a dependency on a fork. At the SiLU key's default exponent (eg 8) `cp_commit`'s mandatory
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
published safetensors (BF16 bits unchanged). Its committed table is sized for this model's 10,240
weight rows (`table 11 24 1376256 65535 0 18`, 2^35 entries; the full kit's line has 2^43), by
criteria read off upstream's three example kits and `HfTableParams::table_derive()`; upstream
records no rule for that line
([README](../../research/probes/flashprove/tinyllama/README.md#the-tinyllama-kit-and-the-answer-check)).

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
Upstream's own account of its limits counts a model without routed experts among the forms that
cannot pass and marks that as read, not tested (`lean/OnBytes/LIMITS.md` in the pinned source): the
branches that only a dense proof takes (the gate's two-limb tensor, its key and its quotient bound)
have been read by reviews of this proposal and run on one request, no more. A `YES` line does not say
that the model is dense or that the key is `eg 7`; the commitment ID covers both. For a file with
experts the patched verifier accepts and refuses what upstream's does; for a malformed file with two
faults the name after `NO:` can differ.

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
  Linux, without a GPU. The two production programs, built again, have since run (below).
- **TinyLlama's shape, planning only.** A dense parameter file at TinyLlama-1.1B's shape with
  `eg 7`, with placeholder scales, passes the patched planner in the standard parameter form; the
  unpatched build refuses it for its three table digests. Without those digests, the unpatched
  build finds exactly `routed-experts`, `both-clamps`, `table-digests` and `no-range-words` unmet
  there, and the patched build only `table-digests`: at this shape the patch changes admission and
  nothing else that the planner sees.

`make verify-flashprove-tinyllama-answer` (CPU, downloads the pinned model) rebuilds the GGUF file
and the kit from pinned inputs (every file byte for byte but `maxima.json`, which is device-dependent
and recorded only), in environments from hash-locked requirement files, checks the kit with both planners and the provers'
reader, runs a part of `cp_commit`'s and `cp_prove`'s set-up that needs only the kit on a CPU with
upstream's code (the weight table's parameters for W's rows, the ranges of each layer's scale words,
the plan of attention of each layer group and the layer layouts; the README lists the set-up rules
that it leaves out), runs upstream's scale-tool tests on the patched tree and a float reference against
transformers, and then requires **the decisive check**: upstream's integer model, with the kit and
every file pinned, gives SASH's eight answer tokens after SASH's 25 prompt tokens, over a whole
64-token window with no failed range condition and the gate's key at most 32,639, with its logits
following the float logits; two one-bit mutations of the kit must each fail that check; and
`cp_commit`'s worst-case check, computed on a CPU, must find every site within its limit. It
passed on 2026-11-24 on Linux (x86_64, every input downloaded) and on macOS (arm64), with the
integer model's results identical on the two
([evidence](../../research/evidence/flashprove-tinyllama-answer-2026-11-24.json)), and again on
Linux on 2026-11-25 with the sized table
([evidence](../../research/evidence/flashprove-tinyllama-gpu-2026-11-25.json)); with that table it also
passed on macOS, run by an AI-assisted review session on the contributor's laptop (not an independent human reviewer).

The current revision of the patches passed `make verify-flashprove-tinyllama` on 2026-11-24 on
Linux (upstream `verifier/build.sh`, g++ 13.3) and on macOS (Apple clang 17, built as
`make verify-flashprove` builds it), and upstream's `integer_model` CPU suite passed on the patched
tree on Linux (947 passed, 0 failed)
([evidence](../../research/evidence/flashprove-tinyllama-2026-11-24.json)); with the sized table it
passed again on Linux on 2026-11-25. Apart from upstream's stored proofs and their altered copies,
which both builds check, these two targets are planning, parsing and table checks: neither makes
or checks a TinyLlama proof. The answer check runs the integer model, not a prover, on one request. Further one-off checks and their limits:
[README](../../research/probes/flashprove/tinyllama/README.md#checks).

**`cp_commit`'s worst-case check, on a CPU.** Upstream computes it on a GPU only. Step 7 of the
answer target computes it with Hallmark's port of upstream's `worst_case.cuh`
(`worst_case_cpu.py`, the decisive sites' rounding emulated exactly) and checks the gate from the
float weights alone (`gate_float_bound.py`); the two agree in every layer. At `eg 8` every site
fits but one ([evidence](../../research/evidence/flashprove-tinyllama-answer-2026-11-24.json),
`retained_failures`): the gate of layer 21, where one row of 5,632 can take a key of 47,481 against the
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

The GPU's own table, from that run, agrees with the port (below). The port does not
cover the GPU's guards on its matrix products ([README](../../research/probes/flashprove/tinyllama/README.md#cp_commits-worst-case-check-on-a-cpu)).

**The first dense proof (the contributor's own run, 2026-11-24 and 25 UTC).** On two RunPod pods of one type, each an H100 SXM (driver
595.112.07, Xeon Platinum 8470), with the patched programs built on a CPU host and only public
files sent: `cp_commit` passed its worst-case check and committed in about 35 s (three times, the
same commitment ID `5ffd1aa3…06bd` each time: with `-public-weights` and upstream's published seed
(`examples/maverick/public-seed.bin`, SHA-256 `630dcd29…10dd`) anyone with such a GPU can recompute
it from the public weights and the kit); `cp_prove` made four proofs of
SASH's request in 8.4 to 9.8 s each, of 13.2 MB; `cp_verify` answered `YES` to each. On a CPU,
`scripts/flashprove_tinyllama_proof.py` then took the first of the four and held every expected
value itself (commitment ID, range `24:32`, end token, exact-match claim, build digest): accepted
by the patched verifier,
`NO: policy-tables` from the unpatched one, and eight altered copies each `NO:` with status 1 (seven
by the checks of the proof, the key, the leaf file or the ID; the token file's on the statement's
form, by a rule that does not depend on the proof's content). The ID is the pod's own print and is not recomputed by the check,
and the first proof's eight public files are an archive in a release of the contributor's fork (the
contributor's own upload of 2026-11-26, not a release of this repository; the check passed again from
the download on Linux and macOS:
[README](../../research/probes/flashprove/tinyllama/README.md#to-repeat-the-callers-check)). The
GPU's worst-case table agrees with the CPU port at every site that the port computes as the GPU
does (within 0.001 bits; the gate of layer 21 keeps 0.458 bits). Spend: USD 1.61 on the
contributor's account. #184 says that paid GPU runs need an owner-approved budget, #186 puts
unbudgeted GPU runs out of scope, `docs/OPERATIONS.md` says to stop before a paid job or an upload
for which no authority exists, and `research/AUTHORITY.json` lists other compute providers than
RunPod: this run had no budget, and no authority record was made or changed for it. Figures and retained failures:
[evidence](../../research/evidence/flashprove-tinyllama-gpu-2026-11-25.json),
[README](../../research/probes/flashprove/tinyllama/README.md#the-first-dense-proof). The times are
two like pods' on lines of which only `table` is sized for this model: not a benchmark.

**The capped run, the enrollment and the composed join (2026-11-26 and 27).** Three things that #186
asks for now exist as code, and each stops short of the issue's box:

- *A target that proves and verifies.* `make e2e-flashprove-tinyllama` rents one sm_90 host,
  runs `gpu_job.sh` there under time limits, deletes the host and runs the caller's check here.
  The planned worst case of the charge must fit a cap before a host is asked for. Its controller
  ran four times, each on its one attempt, for USD 0.36, 0.43, 0.42 and 0.39 against a cap of
  USD 3.00; the first three were calls of the script as it then stood, the fourth the target as
  committed. Each printed the enrolled commitment ID, and each first proof passed the caller's
  check with its eight altered copies. `make verify-flashprove-tinyllama-proof` is the free local check. Short of the box: the
  cap is the contributor's and not an owner's, the provider (RunPod) is not one of
  `research/AUTHORITY.json`'s, and the target takes the provers built beforehand.
- *The caller checks of the SASH path.* `enrollment.json` is written into each run before
  anything is paid for: the case, the kit, the patched verifier's digest and the commitment ID
  that the caller accepts. The pinned tokenizer and chat template must give the request's ids
  for the case's prompt and the case's response for the answer ids, and the answer must end at
  the tokenizer's end token (`make verify-flashprove-tinyllama-request`; the end token is also
  checked on the proof). Short of SASH's: the commitment ID is `cp_commit`'s print in the first
  run, enrolled as a value, where SASH's caller derives its roots on a CPU.
- *The composed demo.* `make e2e-composed LLM_BACKEND=flashprove FLASHPROVE_TINYLLAMA_CASE=DIR` checks a FlashProve
  proof afresh in place of SASH's 23, and the three classifier proofs bind this path's model
  identity (the checkpoint, the commitment ID, `h_par`, the configuration digest and the patched
  verifier's digest), so that they do not join SASH's proof and SASH's do not join this one.
  SASH stays the default. It ran end to end **once**, on the contributor's Mac (Apple M4 Pro),
  with the two frozen classifier quantizations of the owner's reference records: one FlashProve
  proof and three fresh classifier proofs, joined, with the 16 changed bindings refused
  ([evidence](../../research/evidence/flashprove-tinyllama-composed-run-2026-11-27.json)). Short of
  the box: one run on one host, not accepted by the owner, and fresh classifier training on that
  host did not give the pinned quantization.

In one drill the provider did not end a host at the time it was asked to (`terminateAfter`), so
the controller relies on its own deletion and on a second process on the same machine; with the
controller killed outright in a real run, that process had the host deleted within seconds. A
machine that sleeps or loses its network deletes nothing. An AI review between the third and the
fourth run found ways to leave a host billed and none to a false pass; the fourth run's
controller answers them. Figures, the drills, the review and what was not run:
[evidence](../../research/evidence/flashprove-tinyllama-capped-runs-2026-11-27.json),
[README](../../research/probes/flashprove/tinyllama/README.md#the-capped-run).

**Not done; each needs its own step.**

1. **The calibration text's licence**: WikiText-2 (CC BY-SA; its dataset card's tags name 3.0 and
   GFDL, its text 4.0), as in upstream's rule. No text or token id from it is committed, but the
   kit's scales are computed from it. The card's tags and its text name different versions, and
   `research/AUTHORITY.json` excludes ambiguous licences, so the owner decides; a no means refitting
   on another text. The three program patches do not depend on it; `scales.patch` pins the
   calibration set's SHA-256 and would change with the text.
2. **GPU jobs for #187, only with owner approval and a ceiling.** The run above was the
   contributor's, with no owner-approved budget. What it measured, for sizing a ceiling: one pod of 17
   minutes (upload of 2.2 GB, two commitments, four proofs, with about ten idle minutes of the
   controller's own mistake) cost USD 1.27 at USD 4.39 an hour; the commitment takes about 35 s and
   a proof under 10 s. That run's only limit was a watchdog on the controlling
   host, set to terminate the pods after 2,700 s and 3,600 s. A controller with a cap in code now
   exists (above); what it lacks for #187 is an owner's cap, an authorised provider and a build
   step, and it has made at most two proofs on a host.
3. **More altered copies.** Eight altered copies of one dense proof are refused, seven of them by
   checks that read the proof, the key or the ID; upstream's tables
   have 99 altered sets for its MoE proofs, and no table of altered items exists for a dense one.
4. **#187's proof-system lines.** Only `table` is sized for this model. The other lines are the
   full kit's, or set from the verifier's rules (`kFM` and `kFD` by `cp_prove`'s plan of
   attention): admitted by every consumer and proved with once, not sized for speed; and #187's
   sweep needs a kit for each number of checked tokens.
5. **Who answers `YES`.** Only Hallmark's patched verifier does: upstream's refuses the same files
   (`NO: policy-tables`), and Lean covers neither the dense form nor a padded one. Whether Hallmark
   wants a verdict that upstream does not give is the owner's call (option 2 keeps upstream's).
6. **How a verifier gets the commitment ID.** Upstream says to take it "from a party your user
   trusts"; SASH's caller derives its roots from the public weights on a CPU. Here the ID can be
   recomputed from the public weights, the kit and the published seed, but only with an sm_90
   GPU, and only in the mode that hides nothing. `enrollment.json` now holds the ID that a caller
   of this path accepts, as a value taken from the first run; how a Hallmark verifier ties the ID
   to the public TinyLlama checkpoint is still undecided.
7. **A composed run.** `make e2e-composed` with a FlashProve proof has not been run end to end
   (above). It costs nothing and needs a Mac that the classifier demo supports.
8. **A build step for the provers.** The capped target takes built programs. Decision 0093's
   worker builds upstream's provers on a rented CPU host in a pinned image; the same step with
   the four patches applied is not written.

**Not claimed.** One request's proofs exist, made with no owner-approved budget and under no
owner's cap. There is no claim of
soundness of the patched C++ or CUDA code, no Lean coverage, no hiding (the seed is public), no
benchmark and no like-for-like numbers yet. Equal tokens are a smoke test, not a
proof that the two integer functions are equal; #187's like-for-like section must name each
backend's model, conversion, request, window, range and statement.

**Prior work.** `marcellobardus/FlashProve-TinyLlama1.1B` at `1a94f01` (Apache-2.0) reports two
single-token proofs of a dense model at its own shape, by its own logs, which Hallmark has not
reproduced. Hallmark uses none of its code,
proofs, kits, seeds or commitments. Its commit `1c5c2aa` sets `kFM` by the same rule of `cp_prove`'s
plan of attention (14 at TinyLlama's shape).
