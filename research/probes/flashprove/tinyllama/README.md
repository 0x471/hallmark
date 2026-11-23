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
patches; each file's SHA-256 and size are pinned there. A fourth patch, for upstream's scale tools,
and a TinyLlama kit fitted with them are in "The TinyLlama kit and the answer check" below: on a
CPU, upstream's integer model with the kit gives SASH's eight answer tokens.

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
  RN32(sqrt(d) / 2^t), where t is the largest integer with 4^t <= d, and by c_0 / sqrt(dh) for `attn_q`.
  This rule is Hallmark's generalisation, not upstream's: upstream defines c_0 = RN32(sqrt(d) / 64)
  (`docs/model.md`, 11.3), which is the same from width 4,096 to 16,383 but at 2,048 would be half
  of this one (`0x3F3504F3`). Hallmark's form makes c_0 cancel the factor c of sqrt(d) = c 2^t
  (`scales/README.md`, "The exponent of each cell", step 2), so the fit's shifts need no change.
  Upstream hard-codes its constants for 5,120/128.
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
8 to 9 minutes on an M-series Mac and about 1 GB of disk while it runs) does this, in order:

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
| a dense parameter file and `KL.bin` at TinyLlama-1.1B's shape, from [`tinyllama_plan.py`](tinyllama_plan.py) (width 2,048, 32 heads and 4 KV heads of 64, 5,632, 22 layers, vocabulary 32,000, rotary base 10,000, T = 64) | the patched build's planner (`-plan arrays`, `request`, `accounting`) admits it in the standard parameter form, with its three tables matching their digests. Both calls must print the planner's whole ending in upstream's order (the standard-form line can only be missing after it) and name the same files (`h_par`, `configuration`). The unpatched build refuses it with `policy-tables` alone, as its last line. Without its digest lines, the unpatched build finds exactly `routed-experts`, `both-clamps` and `table-digests` unmet, and the patched build only `table-digests`. The shape, `h_par` and `configuration` must equal the pins in `patches.json` |

**Current revision** (patched verifier digest `6f7ac148…`, the patch hashes in `patches.json`):
**passed on Linux and on macOS** on 2026-11-22, every row above as required
([evidence](../../../evidence/flashprove-tinyllama-2026-11-22.json)):

- Ubuntu 24.04, x86_64, upstream `verifier/build.sh` with g++ 13.3. 108 of 108 calls compared equal,
  with exit codes 99 times 1, 7 times 0, and 2 times 4 (planning calls inside `check_proof.py`), 1,670 s.
  Earlier runs, the day before without the TinyLlama row and earlier the same day with looser forms
  of it, passed with the same two verifier binaries.
- macOS arm64, Apple clang 17 (`apple-clang`, built as `make verify-flashprove` builds it): 108 of 108
  calls equal, 499 s.
- Both hosts' planning calls name the same TinyLlama-shaped files (equal `h_par` and
  `configuration`).

Also at this revision, on the same Linux host:
- **nvcc 13.2.86** (no GPU): upstream's `verifier/build.sh`, then `prover/build.sh serving_cp_commit`,
  `serving_cp_prove`, `cp_commit` and `cp_prove`, built both production programs and both test
  builds from the patched source (exit 0; 26, 59, 20 and 46 minutes). Every compiler warning is in a
  file that no patch changes. No program has run. Without a GPU, `build.sh` built its compiler
  sample but could not run it (status 77), so it checked the compiler by its version only. The build
  records name the sources' digest `0fa9c1ff…`, which upstream's `build.sh` computes from the
  sources and anyone can recompute from the pinned archive with the three patches applied;
- the `integer_model` CPU suite (`pytest -m "not vllm and not prover"`, Python 3.13, torch
  2.13.0+cpu) on the tree that the target patched from the pins: 947 passed, 0 failed, 16 skipped
  (4 need a GPU, 12 `transformers`);
- `weight_sums/tests/run_tests.sh`: 24 steps passed, among them both programs' refusal suites and
  every oracle step. The two `make_model` steps end with SIGILL, because that CPU has no SHA-NI;
- Hallmark's `make check` at the last code change: one failure,
  `tests/test_manifests.py::test_file_changed_during_read[abd]`, which fails on Hallmark's `main` on
  that host too (3 of 3 runs); in other runs it was
  `tests/test_worker.py::test_stream_copy_retains_rejected_output[modified]` (4 of 5 on `main`). The
  branch touches neither test nor `src/`. Every other step passed, as did `make package-check` (uv 0.9.18), `check-guard` and
  `check-comparison`. On macOS the whole pytest suite passes (2,266 passed, 0 failed).

**Recorded runs, of an earlier revision.** Three runs on 2026-11-21 (macOS arm64, `apple-clang`, 328-396
s) passed the target as it then was:

- patched verifier digest `f363b8e0…`;
- earlier patch hashes;
- an earlier `dual_verify.py`, which masked every three-decimal number on any line containing
  "seconds" or ending in "<n>.<nnn> s";
- no check of the timed formats or of the negative control's last line.

They recorded 108 equal calls, the standard-form test on the patched build, its negative control's
five misses (not its last line), the provers' reader, the fold constants and `weight_sums`. They had
no timed-output row and no TinyLlama row. They establish those results for that revision only.

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

## The TinyLlama kit and the answer check

[`scales.patch`](scales.patch), a fourth patch pinned in [`kit.json`](kit.json) (not `patches.json`, so
the three patches above and their records stay as they are), lets upstream's scale tools fit a dense
`llama` model. Its 8 files are under `scales/`, and `SUITE.md`, each marked "Hallmark dense patch":

- `gguf_weights.py`: `shape_words()` also reads the architecture `llama`, dense only: no experts,
  the rotary encoding on all coordinates of every layer, and the head dimension, rotary base,
  rotary dimension and norm epsilon required from the file. A `llama` file with experts, a router,
  rotary rescaling (`rope_freqs.weight`, or a `rope.scaling` or `rope.scale_linear` key), a rotary
  dimension other than the head dimension, biases or query/key norms is refused, since the float
  pass would compute another function;
- `policy_tables.py`: at NE 0 it computes the three tables a dense model forms (24 lines, no
  `tsig`), the form `verifier.patch` admits;
- `words_to_policy.py`: the rotary base may be 10,000 as well as 500,000 (the verifier's
  `rotary-table` answers `YES` with either);
- `make_calibration_set.py --tinyllama`: upstream's own sampling rule (WikiText-2, GPTQ-style) with
  TinyLlama's tokenizer, which is byte for byte the Llama 2 tokenizer that upstream's rule pins, and
  TinyLlama's start token 1 in place of Llama 4's 200,000;
- `fit_scales.py`: a comment only. Its shifts are right at width 2,048 as written, because the
  patched conversion multiplies by RN32(sqrt(d) / 2^t), not sqrt(d) / 64, so no factor 2^(t - 6) is
  left;
- `SUITE.md`: its `scales.policy_tables` row counts 23 tests, one more than upstream's;
- upstream's tests: the NE 0 rejection case of `test_policy_tables.py` becomes a test of the three
  tables, `test_gguf_weights.py` gets 16 dense cases, and its "another architecture" case now uses a
  name that stays unsupported.

[`export_gguf.py`](export_gguf.py) writes TinyLlama-1.1B-Chat-v1.0 (the revision and file pins of
Hallmark's SASH path, [`response-case.json`](../../sash/response-case.json)) as one GGUF file: the
BF16 matrices' bits unchanged, the norm gains in F32, the query and key rows permuted as llama.cpp's
converter does for `llama`, and `llama.attention.key_length 64`. `kit/` (pinned in [`kit.json`](kit.json)) holds the kit that
upstream's route (its `scales/README.md`, "A real model", steps 1 to 5) makes from that file, with
e_res 11 (the tools' default) and T = 64: the configuration (`policy.txt`, `KL.bin`, `layout.txt`),
the setup files beside the weights (`wsums.txt`, `fold.txt`) and the fit (`words.json`,
`maxima.json`). Its base parameter file is check 6's, from `tinyllama_plan.py`.

`make verify-flashprove-tinyllama-answer` (CPU only, no GPU, no spend; it downloads the pinned
model, about 2.2 GB, and needs about 8 GB of disk and `uv`) rebuilds all of it and requires, in order
(with `FLASHPROVE_TINYLLAMA_INPUTS` set to a folder of the pinned files it skips the downloads; each file
is checked against its pin and then read in place, so that folder must be one that only you can
change):

| step | requirement |
|---|---|
| export | the GGUF file's SHA-256 is the pin in `kit.json` |
| environments | two virtual environments made by uv from this platform's hash-locked file in `requirements/` (macOS arm64 or Linux x86_64; pinned in `kit.json`), installed with `--require-hashes`; the top-level versions must be upstream's, and the full installed set is recorded |
| setup files | `weight_sums` (patched) and `fold.txt` equal `kit/` byte for byte |
| calibration set | `--tinyllama` gives the pinned set, and its drawn rows have upstream's own `DRAWS_SHA256` (steps 1 to 3 of upstream's rule run on byte-identical inputs) |
| fit | `calibrate.py` gives `kit/words.json` byte for byte; `maxima.json` is compared and recorded |
| configuration | `words_to_policy.py` ends `w2p end 0`; it and `pack_layout.py` give `kit/policy.txt`, `KL.bin` and `layout.txt` byte for byte |
| planning | on the kit and on SASH's request (66 words: 25 prompt ids, 8 answer ids, 31 filler 0, r = 24, e = 32), the patched planner prints its whole ending in the standard parameter form with three tables and the pinned `h_par` and `configuration`; the unpatched build ends `NO: policy-tables`; the provers' parameter-file reader admits the kit |
| upstream's scale tests | `test_policy_tables.py`, `table_ties.py`, `test_calibrate.py` and `test_gguf_weights.py` pass on the patched tree, the first two with `SUITE.md`'s commands (the other two have no row). Status 77, passed in part, is accepted with an "in part" last line: the target builds no `make_model` and sets no `SMALL_MODEL`, so those cases skip on every host |
| float reference | upstream's float pass on the GGUF agrees with Hugging Face transformers on the safetensors at the final norm (relative L2 at most 1e-4), and transformers' greedy decoding gives SASH's answer |
| **the answer** | upstream's integer model (patched, torch 2.13.0), with every file pinned and the weights' integers checked against `wsums.txt`, run over the whole 64-token window: no range condition fails at any position; the greedy choices at positions 24 to 31 are SASH's `[1576, 7483, 310, 3444, 338, 3681, 29889, 2]`; the gate's key stays at most 32,639 (the integer model's condition for a key without the clamp `gcl`; it is meant to match the provers' refusal, but upstream has not compared its range tests with the released `cp_prove`); and at each checked position the integer logits over 2^s have a slope of 0.75 to 1.33 and a correlation of at least 0.998 against the float logits. The slope bound catches an error in the logits' overall scale (`s` or `MF` off by one bit gives a slope near 0.5 or 2); the correlation floor is set from the runs' 0.9991 so that one layer's shift off by one bit, which can leave the tokens and the slope as they are, falls below it |
| mutation controls | the same check on two one-bit changes of the kit must fail at least one condition: `s` + 1 in `policy.txt`, and layer 10's `kv` + 1 in `KL.bin` |

**Current kit** (`scales.patch`, the lock files and `kit/` as pinned in `kit.json`): **passed on
Linux and on macOS** on 2026-11-23, every row as required
([evidence](../../../evidence/flashprove-tinyllama-answer-2026-11-23.json)): Ubuntu 24.04 x86_64
(fresh clone, every input downloaded, 1,877 s) and macOS arm64 (inputs from a checked local
folder, 249 s).

- both environments hold exactly their lock files (43 and 19 distributions);
- every required kit file, the GGUF file and the calibration set rebuilt byte for byte on both
  hosts; `maxima.json` (recorded only) equals the committed one on macOS and differs in its last
  digits on Linux, as upstream's `scales/README.md` expects across devices, while `words.json` (the
  scales) is the same; the calibration set's drawn rows have upstream's `DRAWS_SHA256`;
- float reference: relative L2 3.9e-6 at the final norm; transformers' greedy reply is SASH's;
- the integer model: SASH's eight tokens, no failed range condition over the 64 positions, the
  gate's key at most 7,108, with every recorded peak and margin identical on the two hosts; logit
  slopes 0.988 to 1.004 and correlations of at least 0.9991; the smallest top-1 margin 0.86 logits
  (float: 1.06), at the full stop;
- the mutation controls: with `s` + 1 the slope is 0.49 to 0.50 at every checked position; with
  layer 10's `kv` + 1 the tokens and slopes stay, and the correlation falls to 0.994 to 0.997, below
  the floor;
- the planner's bounds for the kit: a proof of at most 14,061,196 bytes, 126.21 bits, the same as
  check 6, whose proof-system lines the kit keeps.

## Not checked

- **The prover patch has never run.** It compiles with nvcc (above), but no `cp_commit` or `cp_prove`
  has run at NE 0 here, and the pinned repository holds no NE 0 kit. More gaps of this kind can
  surface only in the GPU commit and proving jobs.
- **Tests that need x86-64 SHA instructions or Linux did not run on the Mac.**
  - Upstream's `weight_sums/tests/run_tests.sh`;
  - `integer_model`'s `tests/test_gguf_files.py` (it builds `make_model`);
  - the prover rows `parameter_file_one_bits_test` and `parameter_file_one_place_test`.
- **No TinyLlama commitment or proof.** The kit is fitted and the integer model gives SASH's answer
  on a CPU, but `cp_commit` and `cp_prove` have not run on it.
  - `cp_commit`'s worst-case check has not run (it needs a GPU), and `prover.patch` leaves it out
    under `-public-weights`. Whether TinyLlama's scales would pass it is not known.
  - Of the provers' rules, only their parameter-file reader has judged the kit (it admits it).
- **The kit's proof-system lines are check 6's**, not sized for speed: the full kit's values, with
  the lines that TinyLlama's shape forces (`ll`, which 22 layers force from 2 to 1; `la`, `lB`,
  `lmS`, `kFD`, `iotaA_0`, `iotaH_0` to `iotaH_2`, `ranges-part`) set from the verifier's own rules
  and refusals. The fit changes only the scales and the six lines that depend on them. They are
  placeholders that `cp_prove` has not judged beyond its parameter-file reader; it has further
  checks at run time (for example `forest-depth-A` in `prover/attn/front.cuh`). The planner's sizes
  and bits for the kit are bounds of these lines, not measurements, and #187 may want smaller ones.
- **One request.** The answer check covers SASH's one prompt in one window of T = 64. On it one value
  cell clamps (`v` in layer 2, 2,233 to 2,047, at position 32, the final EOS, after the checked
  positions); another prompt
  can clamp other cells or fail a range condition. #187's sweep over the number of checked tokens needs a
  kit for each T.
- **Equal tokens are a smoke test.** They show that this integer model agrees with SASH's answer on
  this request, not that FlashProve's and SASH's integer functions are equal. The mutation controls
  show that the bounds catch two kinds of one-bit error; other single errors may still pass.
- **The calibration text is WikiText-2**, as in upstream's rule: CC BY-SA (its dataset card's tags
  name CC BY-SA 3.0 and GFDL, its text CC BY-SA 4.0). No text and no token id from it is committed,
  but numbers computed from it are: `kit/maxima.json`, `words.json`, `KL.bin`, six lines of
  `policy.txt`, and the calibration set's SHA-256 in `kit.json`. Whether that suits Hallmark's source
  rules is the owner's call; if not, the kit must be refitted on another text.
- **The environments pin every distribution by hash, but not Python's patch version** (uv's choice
  of 3.13; the runs had 3.13.2 and 3.13.15), and only for macOS arm64 and Linux x86_64. They are
  installed from PyPI and, for torch on Linux, from PyTorch's CPU index.
- **The float reference is Hugging Face transformers 5.15.1**, run as a check of the export. It is
  not part of any statement.

## Prior work

[`marcellobardus/FlashProve-TinyLlama1.1B`](https://github.com/marcellobardus/FlashProve-TinyLlama1.1B)
at `1a94f01` (Apache-2.0) reports two single-token proofs of a dense model at its own shape, by its own
logs, which Hallmark has not reproduced. Hallmark uses none of its code, proofs, kits, seeds or
commitments, and these patches were written from upstream's sources and documents.

Two differences from it:

- its verifier keeps a count per table, where this patch allows only the one gap where `tsig` would
  be;
- it changes `moe_at_layer`, which this patch finds unnecessary (above).
