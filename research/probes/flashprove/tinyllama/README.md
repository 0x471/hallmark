# TinyLlama-1.1B on pinned FlashProve: dense-model patches

Status: **proposed, not owner-approved** ([#186](https://github.com/AbdelStark/hallmark/issues/186),
option 3; decision draft [0094](../../../../docs/decisions/0094-flashprove-tinyllama.md)). No proof has
been made with these patches. The prover patch compiles with upstream's nvcc 13.2.86 (building needs
no GPU), but neither prover program has run: that needs an H100-class GPU. Nothing here changes
`make verify-flashprove` or decision 0093's pins.

Upstream FlashProve at the pinned commit `9e2bdd8` ([`source.json`](../source.json)) reads real model
weights only at Llama 4 Maverick's widths, 5,120 and head width 128. Its own test kits are smaller
synthetic models (the pinned kit: width 512, head width 128). Hallmark's LLM case is dense
TinyLlama-1.1B (width 2,048, head width 64, no experts). These three patches are meant to let the same
pinned source prove and verify a dense model of TinyLlama's shape natively, so that #187 can compare
provers on the same model. That has not been shown end to end: see "Not checked". They are applied
with `git apply`, in the order of [`patches.json`](patches.json), as Hallmark applies its SASH
patches; each file's SHA-256 and size are pinned there.

## What changes

### `verifier.patch` (cp_verify; 3 files)

Upstream lets a dense model (`NE 0`) through every proof check, but never to `YES`. Three conditions of
the standard parameter form stop it:

1. `routed-experts`: NE must be above 0 (`verifier/src/vf_common.h`, `Served`).
2. `both-clamps`: the lines `rcl 1` and `gcl 1` must be there, and both are refused at NE 0
   (`policy-rcl`, `policy-gcl`).
3. `table-digests`: all four public tables' digests must be there, and the digest lines are refused at
   NE 0 (`policy-tables`, `cp_verify.cpp` `polLayout`), because a dense model forms no sigmoid table
   and the fourth digest would never be compared.

The patch:

- accepts the digests of the three tables a dense model forms (SiLU, RoPE, softmax), in upstream's
  order with the eight `tsig` lines left out, and only at NE 0. Four digests stay refused at NE 0, and
  three are refused above it. So every table formed has its digest compared, and every digest given
  is compared;
- treats `routed-experts` and `both-clamps` as met at NE 0. `rcl` and `gcl` stay refused at NE 0, as
  upstream refuses them;
- extends upstream's own standard-form test (`verifier/tests/served_form_test.py`) with the dense
  cases.

No check of a proof's bytes changes. What changes is admission: which parameter files can reach `YES`,
and, at NE 0, three table digests that are now compared where upstream refused the lines. Upstream's
proof checks already run on dense layers: Maverick's even layers are dense, and `NE 0` takes the existing
`p3 = NE > 0` branches. On a file malformed in two ways at once (say, no `tsig` and a wrong `hr` with
experts), the rejection's name can differ from upstream's; both reject.

What a patched `YES` means that upstream's does not:

- **Not upstream's endorsed form.** Upstream lists `NE 0` among the forms "on which no call ends with
  an acceptance" (`docs/format/11-2-feed-forward-segment.md`; `docs/format/FORMAT.md` 13.7,
  condition 12), and the pinned repository holds no NE 0 FlashProve kit or proof (its CPU
  integer-model tests do build NE 0 models). So the patch widens what the verifier stands behind to a
  form for which upstream ships no proof.
- **The gate's key is range-checked, not clamped.** Without `gcl` the model's gate key must lie in
  -32,768 .. 32,767 (`docs/model.md`, "9 Range conditions", condition 6). The shipped provers write it
  in two limbs and make no proof above 32,639 (`prover/prover/forward.cuh`, `gate_key`): such a
  request gets `failed`. The verifier's SiLU table covers the model's whole interval.
- **The dense form is chosen by NE alone.** Nothing in it ties a `YES` to width 2,048, to TinyLlama or
  to SASH. As upstream's, a `YES` is about the separately enrolled integer commitment and the caller's
  request and range. It says nothing of which checkpoint was converted, of the tokenizer, of the
  conversion, or of whether FlashProve's integer function equals SASH's.
- **Lean covers neither form.** Its statement is for one model size (`lean/GAPS.md`), so it covers
  neither this nor a padded TinyLlama.

### `prover.patch` (cp_commit, cp_prove; 9 files)

- **Fold constants (P1).** The conversion multiplies each norm's gains into the next matrix by c_0 =
  RN32(sqrt(d) / 2^t), where t is the largest integer with 4^t <= d, and by c_0 / sqrt(dh) for `attn_q`
  (`scales/README.md`, "The exponent of each cell", step 2). Upstream hard-codes them for 5,120/128.
  The patch adds a table of admitted shapes: (5,120, 128) with upstream's `0x3F8F1BBD`, `0x3DCA62C2`,
  and (2,048, 64) with `0x3FB504F3`, `0x3E3504F3` (sqrt 2 and sqrt 2 / 8). These are exact bit
  patterns, not computed at run time.
  - Files: `weights_commitment/fold_weights.cuh`, `fold_source.cuh`, `weights_blocks/gguf_source.cuh`,
    `hash_commitment/hf_commit_weights.cuh`.
  - Since the constants follow the GGUF file, `cp_commit` now also requires the file's width, head
    width and layers with experts to be the parameter file's, as `cp_prove`'s reader already does.
    Tensor sizes alone cannot tell dh 64 with H 32, G 4 from dh 128 with H 16, G 2.
- **Three digests (P3).** The provers' parameter-file reader (`common/parameter_file.h`) takes the
  same two digest forms as the verifier, with the same rule.
- **Worst-case check (P4).** Under `-public-weights` (the published seed in `isk.txt`'s line
  `seed public`), `cp_commit` leaves out the worst-case check unless `-worst-case` or
  `-worst-case-table` is given, as it already does under the test option `-test-seed`. `cp_prove` then
  starts without `isk.txt`'s line `ranges`, but only under `seed public` and `-public-weights`.
  - The check is a hiding safeguard. "Soundness does not rest on it"
    (`scales/worst-case-bounds.md`), and with public weights nothing is hidden. The cost: a request
    whose values leave a range ends `failed`.
  - Files: `weights_commitment/arguments.h`, `hash_commitment/hf_commit.cu`, `prover/isk.cuh`,
    `common/cp_files.h` (a comment).
  - What `cp_prove` checks at that point is only a label. `isk.txt` has no authentication, as upstream
    notes, and `read_isk()` requires `-public-weights` with a readable file whenever it says
    `seed public`. A secret commitment whose `isk.txt` was altered to say `seed public` and lacks
    `ranges` therefore takes this branch when run with a wrong public seed file. `cp_prove` then goes
    on starting, and may clear stale spool output, until `read_the_seed()` compares that file with
    `seed.bin` and refuses, before any request is served.
- **Not changed: `moe_at_layer`** (`prover/prover/layout.h`). It reads the prover's own profile,
  which is `mo ? 3 : 2` with `mo = NE > 0` (`prover/inputs.cuh`, `prover/layer_setup.cuh`). So at
  NE 0 no layer takes the MoE path, though the function itself does not test NE.

### `tools.patch` (CPU tools; 7 files)

- **`weight_sums`.**
  - `weight_sums.cpp`, `weight_sums.py` and the test oracle `weight_sums/tests/oracle.py` take the same
    table of shapes.
  - The oracle derives the two constants by exact rational comparison, and asserts that the derivation
    gives upstream's two words. The variable `WEIGHT_SUMS_ORACLE_SHAPE=2048,64` makes it write a model
    of that shape.
  - For the added pair both programs take width, head width and layer count only as unsigned integers.
    A width or layer count of another type is refused by both, with different exit statuses (C++ 22,
    Python 25).
  - With upstream's own pair, upstream's differences stay as they were: `weight_sums.py` accepts a
    width or layer count of another numeric type, where the C++ program refuses it
    (`weight_sums/tests/refusals.py`).
- **`integrations/vllm/integer_model`** (upstream's CPU reference of the integer model) reads a dense
  parameter file:
  - NE 0 with the three digests, and a message for NE 0 that asks for those 24 lines;
  - the expert-only weight widths (`wb_gs`, `wb_us`, `wb_gr`, `wb_ur`, `wb_dnr`, `wb_r`) and the
    router's digit form `hr` at 0, as `cp_verify` and the provers require (`header-absent-kind`,
    `header-absent-class`);
  - no sigmoid digest compared at NE 0;
  - its fold constants by shape.
  - Tests: in `tests/test_parameter_file.py`.

Every changed file carries a "Modified by the Hallmark project" notice (Apache-2.0, section 4(b)).
[`LICENSE.upstream`](LICENSE.upstream) and [`NOTICE.upstream`](NOTICE.upstream) are upstream's files
at the pinned commit, byte for byte.

## Checks

`make verify-flashprove-tinyllama` (CPU, synthetic test files only, no model weights, no paid compute; about
7 minutes on an M-series Mac and about 1 GB of disk while it runs) does this, in order:

1. checks the manifest's upstream revision, and each patch's size and SHA-256, copying the checked
   bytes into the run folder;
2. fetches the pinned archive and checks its size and SHA-256 before extracting it;
3. builds the unpatched `cp_verify`, after checking its source digest;
4. applies the checked copies (git with an explicit environment and no system or user configuration);
5. builds the patched `cp_verify`, after checking its source digest;
6. requires what the table lists.

It runs upstream's pinned scripts and builds with the caller's own permissions, as
`make verify-flashprove` does. It is not a sandbox.

| check | requirement |
|---|---|
| timed output | every wall-clock time (`%.3f`) in the verifier's sources is one of the 9 timed line formats in [`dual_verify.py`](dual_verify.py) (14 times) |
| upstream's stored proofs and altered copies; every `cp_verify` call made by both builds through `dual_verify.py` | exactly 108 calls. Each gives the same exit code and output from both builds, apart from the build digest and the times on those 9 formats. The two stored proofs end `YES`. Each table's unaltered control passes (the clipped-gap control ends `WITHIN LIMITS`), and all 99 altered sets are rejected by the expected check |
| `served_form_test.py` (patched) on the patched build | `all as expected` |
| the same test on the unpatched build (negative control) | fails exactly the 5 cases that only the patch meets, and says so in its last line |
| the provers' parameter-file reader, built from the patched source | admits the upstream kit, the dense kit and the dense kit with three digests; refuses four digests at NE 0, `tsig` for `trope`, and no `tsig` with experts (`policy-tables`) |
| fold constants in the 4 patched implementations | equal to each other, to the pins and to an exact derivation |
| `weight_sums` (patched) against upstream's exact oracle | the same sums at 5,120/128 (seed 1) and 2,048/64 (seeds 1 and 100) |
| a dense parameter file and `KL.bin` at TinyLlama-1.1B's shape, from [`tinyllama_plan.py`](tinyllama_plan.py) (width 2,048, 32 heads and 4 KV heads of 64, 5,632, 22 layers, vocabulary 32,000, rotary base 10,000, T = 64) | the patched build's planner (`-plan arrays`, `request`, `accounting`) admits it in the standard parameter form, with its three tables matching their digests; the unpatched build refuses it with `policy-tables` alone |

**Current revision** (patched verifier digest `6f7ac148…`, the patch hashes in `patches.json`):
**passed on Linux and on macOS** on 2026-11-22, every row above as required
([evidence](../../../evidence/flashprove-tinyllama-2026-11-22.json)):

- Ubuntu 24.04, x86_64, upstream `verifier/build.sh` with g++ 13.3. 108 of 108 calls compared equal,
  with exit codes 99 times 1, 7 times 0, and 2 times 4 (planning calls inside `check_proof.py`). A run
  the day before, without the TinyLlama row, passed with the same two verifier binaries.
- macOS arm64, Apple clang 17 (`apple-clang`, built as `make verify-flashprove` builds it): 108 of 108
  calls equal, 495 s.

Also at this revision, on the same Linux host:
- **nvcc 13.2.86** (no GPU): upstream's `verifier/build.sh`, then `prover/build.sh serving_cp_commit`,
  `serving_cp_prove`, `cp_commit` and `cp_prove`, built both production programs and both test
  builds from the patched source (exit 0; 26, 59, 20 and 46 minutes). Every compiler warning is in a
  file that no patch changes. No program has run;
- the `integer_model` CPU suite (`pytest -m "not vllm and not prover"`, Python 3.13, torch
  2.13.0+cpu) on the patched tree: 947 passed, 0 failed, 16 skipped (4 need a GPU, 12
  `transformers`);
- `weight_sums/tests/run_tests.sh`: 24 steps passed, among them both programs' refusal suites and
  every oracle step. The two `make_model` steps end with SIGILL, because that CPU has no SHA-NI;
- Hallmark's `make check`: one failure, `tests/test_worker.py::test_stream_copy_retains_rejected_output[modified]`,
  which fails on upstream `main` on that host too (4 of 5 runs); the branch touches neither it nor
  `src/`. Every other step passed, as did `make package-check` (uv 0.9.18), `check-guard` and
  `check-comparison`. On macOS the whole pytest suite passes (2,242 passed, 0 failed).

**Recorded runs, of an earlier revision.** Three runs on 2026-11-21 (macOS arm64, `apple-clang`, 328-396
s) passed the target as it then was:

- patched verifier digest `f363b8e0…`;
- earlier patch hashes;
- an earlier `dual_verify.py`, which masked every three-decimal number on any line containing
  "seconds" or ending in "<n>.<nnn> s";
- no check of the timed formats or of the negative control's last line.

They recorded 108 equal calls and every row above. They establish that result for that revision
only.

Other checks, run once on 2026-11-21 outside the make target, each on an earlier revision (scratch
copies). The verifier sources then differed from today's in comments only. The other files differed
as noted.

- **Eight more of upstream's CPU verifier rows** (`SUITE.md`), with the earlier two-build stand-in:
  2,294 calls, 0 differ, every row passing on the patched build.
  - Rows: `example_kits`, `kit_rounds`, `kit_rounds_second`, `plan_tau`, `commitment_id.examples`,
    `record_altered_sets.example`, `made_up_files.work`, `extreme_files`.
- **Upstream's unmodified `served_form_test.py` on the patched build.** Exactly one case differs,
  `routed-experts.dense`, which now misses only `table-digests`.
- **`weight_sums.cpp` and `weight_sums.py` against the oracle.** Seeds 1-4 and 100-102 at both shapes
  (all storage types), all equal, before the later type rules for the added pair.
  - `weight_sums.py` was re-run on seeds 1 and 100 at both shapes after the first of them.
  - Upstream's build refuses the 2,048/64 models (exit 25, `bound`).
  - Both programs refuse width 2,048 with head 128 or no `key_length` (absent means 128), width 4,096,
    and width 5,120 with head 64.
- **`integer_model`,** before the `hr` rule and the dense message:
  - `tests/test_parameter_file.py`: 449 passed. It compares the reader with the provers' reader built
    from the patched header.
  - `tests/test_against_recorded_values.py`: 81 passed.
  - Both under Python 3.13 and torch 2.13.0, with a `g++` shim for Apple clang.
- **`prover/tests/guards_stand.py`** (upstream's source pins of the provers' guards): `ALL AS EXPECTED`,
  before the `cp_commit` shape check.
- **Syntax only.** `prover/weights_commitment/arguments.h` compiles in serving and test builds, and
  `fold_weights.cuh` compiles host-side against a CUDA stub.

## Not checked

- **The prover patch has never run.** It compiles with nvcc (above), but no `cp_commit` or `cp_prove`
  has run at NE 0 here, and the pinned repository holds no NE 0 kit. More gaps of this kind can
  surface only in the GPU commit and proving jobs.
- **Tests that need x86-64 SHA instructions or Linux did not run on the Mac.**
  - Upstream's `weight_sums/tests/run_tests.sh`;
  - `integer_model`'s `tests/test_gguf_files.py` (it builds `make_model`);
  - the prover rows `parameter_file_one_bits_test` and `parameter_file_one_place_test`.
- **At TinyLlama's own shape only planning is checked.** The 2,048/64 checks are the constants,
  `weight_sums` on synthetic files, and the planner on `tinyllama_plan.py`'s file. A planning status
  checks no proof.
  - That file's scales are the full kit's, and the lines that the shape forces away from the full
    kit's values (`la`, `lB`, `lmS`, `kFD`, `iotaA_0`, `iotaH_0` to `iotaH_2`, `ranges-part`) were set
    from the verifier's own refusals. They are placeholders, not a fitted kit; `cp_prove` asks more of
    some of them (`lmS` among them) and has not planned this file.
  - Its sizes and bits from `-plan accounting` are those of placeholder lines and are no measurement.
  - No dense GGUF has been read by `integer_model`; that comes with the TinyLlama kit.
- **`scales/` is not patched yet.**
  - Its converter reads `llama4` MoE files only (`scales/README.md`), and `scales/policy_tables.py`
    refuses NE 0.
  - The shifts and E_logit assume t = 6, and theta 500,000.
  - A dense conversion must also set the per-layer rotary flags and keep Q and K rows in the order that
    the integer model reads.
  - So there is no dense kit for TinyLlama itself, and no check that FlashProve's integer TinyLlama
    gives SASH's answer. Even when it does, eight equal answer tokens are a smoke test: they do not
    show that the two provers' integer functions are equal.
- **The TinyLlama GGUF must state `llama.attention.key_length 64`.** Without the key, the provers and
  `weight_sums` read 128 (`docs/model.md`, 11.1) and refuse the model.

## Prior work

[`marcellobardus/FlashProve-TinyLlama1.1B`](https://github.com/marcellobardus/FlashProve-TinyLlama1.1B)
at `1a94f01` (Apache-2.0) reports two single-token proofs of a dense model at its own shape, by its own
logs, which Hallmark has not reproduced. Hallmark uses none of its code, proofs, kits, seeds or
commitments, and these patches were written from upstream's sources and documents.

Two differences from it:

- its verifier keeps a count per table, where this patch allows only the one gap where `tsig` would
  be;
- it changes `moe_at_layer`, which this patch finds unnecessary (above).
