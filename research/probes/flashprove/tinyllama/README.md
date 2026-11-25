# TinyLlama-1.1B on pinned FlashProve: dense-model patches

Status: **proposed** ([#186](https://github.com/AbdelStark/hallmark/issues/186),
option 3; decision draft [0094](../../../../docs/decisions/0094-flashprove-tinyllama.md)). One dense
commitment and four proofs of SASH's request have been made with these patches on a rented H100, on
the contributor's own account and at the contributor's cost, with no owner-approved budget
("[The first dense proof](#the-first-dense-proof)"). Nothing here changes `make verify-flashprove`
or decision 0093's pins.

Upstream FlashProve at the pinned commit `9e2bdd8` ([`source.json`](../source.json)) reads real model
weights only at Llama 4 Maverick's widths, 5,120 and head width 128. Its own test kits are smaller
synthetic models (the pinned kit: width 512, head width 128). Hallmark's LLM case is dense
TinyLlama-1.1B (width 2,048, head width 64, no experts). These three patches are meant to let the same
pinned source prove and verify a dense model of TinyLlama's shape natively, so that #187 can compare
provers on the same model. One request has gone through commit, prove and verify; what that does
not show is in "Not checked". They are applied
with `git apply`, in the order of [`patches.json`](patches.json); each file's SHA-256 and size are
pinned there. That is the mechanism of Hallmark's SASH patches and no more
([ARCHITECTURE](../../../../docs/ARCHITECTURE.md#how-sash-is-integrated) says what each of those
does): `verifier.patch` here widens what the verifier admits. A fourth patch, for upstream's scale tools,
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
- treats `no-range-words` as met at NE 0 when the file has the line `eg 7` (the SiLU table's key in
  steps of 2^-7) and no `smax`. Without `gcl`, which needs experts, a gate's key must fit two limbs
  (at most 32,639), and `eg 7` doubles the gate inputs that this covers, at half the key's
  resolution; upstream's own `scales/worst_case.py` names a lower `eg` as the way to raise a gate's
  limit. No other `eg`, no `smax`, and nothing above NE 0 changes: upstream's `eg 7` case with
  experts still misses the condition;
- extends upstream's own standard-form test (`verifier/tests/served_form_test.py`) with the dense
  cases, `eg 7`, `eg 6` and `eg 7` with `smax` among them.

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

### `prover.patch` (cp_commit, cp_prove; 5 files)

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
- **No change to the worst-case check.** `cp_commit` runs upstream's mandatory worst-case check, and
  `cp_prove` requires `isk.txt`'s line `ranges`, as upstream does. Computed on a CPU, the check
  refuses a TinyLlama kit at the default `eg 8` at one gate row; the kit is therefore fitted at
  `eg 7`, which `verifier.patch` admits at NE 0, and then every site fits ("cp_commit's worst-case
  check, on a CPU" below). No check is left out.
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

`make verify-flashprove-tinyllama` (CPU, synthetic test files only, no model weights, no paid compute; 333 s to
513 s in the recorded runs on an M-series Mac, and about 1 GB of disk while it runs) does this, in order:

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
| the same test on the unpatched build (negative control) | fails exactly the 8 cases that only the patch meets (five dense cases and three `eg` cases at NE 0), and says so in its last line |
| the provers' parameter-file reader, built from the patched source | admits the upstream kit, the dense kit and the dense kit with three digests; refuses four digests at NE 0, `tsig` for `trope`, and no `tsig` with experts (`policy-tables`) |
| fold constants in the 4 patched implementations | equal to each other, to the pins and to an exact derivation |
| `weight_sums` (patched) against upstream's exact oracle | the same sums at 5,120/128 (seed 1) and 2,048/64 (seeds 1 and 100) |
| a dense parameter file and `KL.bin` at TinyLlama-1.1B's shape, from [`tinyllama_plan.py`](tinyllama_plan.py) (width 2,048, 32 heads and 4 KV heads of 64, 5,632, 22 layers, vocabulary 32,000, rotary base 10,000, T = 64) | the patched build's planner (`-plan arrays`, `request`, `accounting`) admits it in the standard parameter form, with its three tables matching their digests. Both calls must print the planner's whole ending in upstream's order (the standard-form line can only be missing after it) and name the same files (`h_par`, `configuration`). The unpatched build refuses it with `policy-tables` alone, as its last line. The file has `eg 7`, `kFM 14`, `kFD 15` and the sized line `table` (pinned in `patches.json`), as the kit has. Without its digest lines, the unpatched build finds exactly `routed-experts`, `both-clamps`, `table-digests` and `no-range-words` unmet, and the patched build only `table-digests`. The shape, `h_par` and `configuration` must equal the pins in `patches.json` |

**With the sized table** (the three patches unchanged; check 6's file has the new line `table`,
so its pinned `h_par` and `configuration` changed): **passed on Linux** on 2026-11-25 at `40186c6`,
every row above as required, on a clean clone under uv 0.9.18: 108 of 108 calls compared equal,
1,630 s; check 6's file plans to a proof of at most 13,685,772 bytes and 126.21 bits (the planner's
own accounting, not a security level)
([evidence](../../../evidence/flashprove-tinyllama-gpu-2026-11-25.json)). On macOS arm64 it passed at
the same commit (108 of 108, 333 s), run by an AI-assisted review session on the contributor's laptop (not an independent human reviewer).
Upstream's `verifier/build.sh` refuses to build below a directory that its group can write in: a
clone made under umask 002 fails this target at once (`chmod go-w` the clone).

**Before that** (patched verifier digest `33e9d53e…`, the patch hashes in `patches.json`):
**passed on Linux and on macOS** on 2026-11-24 at `bf3add1`, every row above as required
([evidence](../../../evidence/flashprove-tinyllama-2026-11-24.json)):

- Ubuntu 24.04, x86_64, upstream `verifier/build.sh` with g++ 13.3. 108 of 108 calls compared equal,
  with exit codes 99 times 1, 7 times 0, and 2 times 4 (planning calls inside `check_proof.py`),
  1,605 s. The standard-form test passed, and the unpatched build missed exactly the eight cases
  that only the patch meets.
- macOS arm64, Apple clang 17 (`apple-clang`, built as `make verify-flashprove` builds it): 108 of 108
  calls equal, with the same exit codes, 513 s.
- Both hosts' planning calls name the same TinyLlama-shaped files (equal `h_par` and
  `configuration`).

Also at this revision, on the same Linux host
([evidence](../../../evidence/flashprove-tinyllama-2026-11-24.json), `prover_compile` and
`other_checks`):
- **nvcc 13.2.86** (no GPU): upstream's `verifier/build.sh`, then `prover/build.sh serving_cp_commit`,
  `serving_cp_prove`, `cp_commit` and `cp_prove`, built both production programs and both test
  builds from the patched source (exit 0; 24, 67, 21 and 51 minutes). Every compiler warning is in a
  file that no patch changes. No program ran then (the production programs, built again, ran on
  2026-11-25: "The first dense proof"). Without a GPU, `build.sh` built its compiler
  sample but could not run it (status 77), so it checked the compiler by its version only. The build
  records name the sources' digest `7f304c42…`, which upstream's `build.sh` computes from the
  sources and anyone can recompute from the pinned archive with the three patches applied, and as
  the verifier beside `cp_commit` the target's own patched `cp_verify` (`a163a90a…`);
- on the target's patched tree, the `integer_model` CPU suite (`pytest -m "not vllm and not
  prover"`, Python 3.13, torch 2.13.0+cpu): 947 passed, 0 failed, 16 skipped (4 need a GPU, 12
  `transformers`); and `weight_sums/tests/run_tests.sh`: 24 steps passed, among them both
  programs' refusal suites and every oracle step. The two `make_model` steps end with SIGILL,
  because that CPU has no SHA-NI;
- Hallmark's `make check` and `make package-check` at `bf3add1`, as CI's
  core job runs them (a clean clone, uv 0.9.18, uv's Python 3.12.12, 3.13.15 and 3.14.2):
  `make package-check` passed for all three, and `make check` for 3.12 (pytest 2,277 passed). For
  3.13 and 3.14 `make check` stopped at one test that also fails on Hallmark's `main` on that host
  (in 5 of 7 full runs there), in a file the branch does not touch (nor `src/`):
  `tests/test_worker.py::test_stream_copy_retains_rejected_output[modified]`; there
  `tests/test_manifests.py::test_file_changed_during_read[abd]` fails too in 4 of 7 runs (two
  back-to-back writes can leave a file's `st_mtime_ns` unchanged).
  (`tests/test_e2e.py::test_timeout_allows_controller_to_reap_separate_worker` fails on any branch
  when the suite is started in the background of a non-interactive shell, which ignores SIGINT;
  it passes in the foreground.)
  On macOS (uv's Python 3.12.12, 3.13.15 and 3.14.7) both passed for all three (pytest 2,289
  passed, 0 failed, 102 skipped), and so does `make docs`. With Homebrew's Python 3.13.2 or 3.14.3 in place of uv's,
  `tests/test_native_llm.py::test_sandbox_nested_spawn_with_large_inherited_descriptor_limit` fails
  on macOS, on Hallmark's `main` too.

## The TinyLlama kit and the answer check

[`scales.patch`](scales.patch), a fourth patch pinned in [`kit.json`](kit.json) (not `patches.json`: only
the answer check applies it), lets upstream's scale tools fit a dense
`llama` model. Its 8 files are under `scales/`, and `SUITE.md`, each marked "Hallmark dense patch":

- `gguf_weights.py`: `shape_words()` also reads the architecture `llama`, dense only: no experts,
  the rotary encoding on all coordinates of every layer, and the head dimension, rotary base,
  rotary dimension and norm epsilon required from the file. A `llama` file with experts, a router,
  rotary rescaling (`rope_freqs.weight`, a `rope.scaling` key, among them `attn_factor` and
  `finetuned`, or `rope.scale_linear`), a rotary dimension other than the head dimension, biases,
  query/key norms, attention that is not causal (`attention.causal` false), or query or key rows
  that are not a whole number of heads is refused, since the float pass would compute another
  function;
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
  tables, `test_gguf_weights.py` gets 22 dense cases (one per refusal, and a file that states
  `causal` true is read), and its "another architecture" case now uses a name that stays
  unsupported.

[`export_gguf.py`](export_gguf.py) writes TinyLlama-1.1B-Chat-v1.0 (the revision and file pins of
Hallmark's SASH path, [`response-case.json`](../../sash/response-case.json)) as one GGUF file: the
BF16 matrices' bits unchanged, the norm gains in F32, the query and key rows permuted as llama.cpp's
converter does for `llama`, and `llama.attention.key_length 64`. `kit/` (pinned in [`kit.json`](kit.json)) holds the kit that
upstream's route (its `scales/README.md`, "A real model", steps 1 to 5) makes from that file, with
e_res 11 (the tools' default), `eg 7` and T = 64: the configuration (`policy.txt`, `KL.bin`, `layout.txt`),
the setup files beside the weights (`wsums.txt`, `fold.txt`) and the fit (`words.json`,
`maxima.json`). Its base parameter file is check 6's, from `tinyllama_plan.py`, `eg 7` included.

**The committed table is sized for this model.** The base file's line `table` is not the full
kit's (`table 11 32 402653184 44040191 0 18`, Maverick's): with it `cp_commit` would encode 2^11
codewords of length 2^32 for a model whose data fill a 307th of a row's entries. W, the matrix of
weight digits, has 10,240 rows of 2^18 columns here (upstream's `layoutW()`; Maverick has
3,076,096). `tinyllama_plan.py` writes `table 11 24 1376256 65535 0 18`, 2^35 table entries in place
of 2^43:

- the first and last numbers, 2^11 long rows and 2^18 columns, are the full kit's (18 is also the
  provers' most at this vocabulary, log2 V rounded up + 3);
- the masks of a row, 65,535, are upstream's small example kits'. This is a choice: they bound the
  columns that may ever be opened on one stored copy, and `cp_commit`'s log on the GPU counted 1,128 batches
  with them (upstream's guide gives 688,127 at full size; the full kit's 44,040,191 would force a row of at least 2^26 entries);
- the message entries of a row are the least that `HfTableParams::table_derive()` admits: at least
  the row's 1,310,720 data entries and three pads, and a multiple of 2^16;
- the code's length is 2^(b + 3), b the bits of the row's last coefficient index, as in each of
  upstream's example kits.

These are criteria read off upstream's three kits and its code, not a rule of upstream's: it
records none for these lines (`prover/examples/TARGET-KITS.md`, "How the small configurations'
lines were chosen"). With upstream's own rows and masks, `table_numbers()` gives the line of each
of its example kits, and `weight_rows()` the 59,392 and 3,076,096 rows that upstream's code names
(tested). The full kit's line was never run on this model, so nothing is known of what it would
have cost; the sized line's cost is measured below. The other proof-system lines
(`group`, `stages`, the matrices' sizes) stay the full kit's: every consumer admits them, and they
are larger than this model needs.

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
| fit | `calibrate.py` runs (its fit is at `eg 8`); [`fit_eg.py`](fit_eg.py), the same call of `fit_scales.fit()`, gives `calibrate.py`'s `words.json` byte for byte at `eg 8` and `kit/words.json` byte for byte at `kit.json`'s `eg 7`; `maxima.json` is compared and recorded |
| configuration | `words_to_policy.py` ends `w2p end 0`; it and `pack_layout.py` give `kit/policy.txt`, `KL.bin` and `layout.txt` byte for byte |
| planning | on the kit and on SASH's request (66 words: 25 prompt ids, 8 answer ids, 31 filler 0, r = 24, e = 32), the patched planner prints its whole ending in the standard parameter form with three tables and the pinned `h_par` and `configuration`; the unpatched build ends `NO: policy-tables`; the provers' parameter-file reader admits the kit; a part of `cp_commit`'s and `cp_prove`'s set-up that needs only the kit, run on a CPU with upstream's code ([`prover_setup.cpp`](prover_setup.cpp): the weight table's parameters for W's rows, the ranges of each layer's scale words, the plan of attention of each layer group with the column rule of its committed matrices, and each group's layer layout), admits it, and refuses four changed kits each for its own reason: `kFM` one higher (the plan of attention), one step less of the table's message entries (`policy-pad-place`), `lmh 10` (a column that is no multiple of D) and layer 1's `kd` 25 (outside its digit form's range). Not run there, though they need only the kit: `5 <= lmbA <= kT + 5`, the pad tree's size, attention's and the layers' forest plans (among them `shape-call` on the lines `iota`), the limit on a lookup call's pieces, the table digests and the output head's set-up, so a kit can pass here and still stop at `cp_prove`'s start. Of the two controls added last, the patched `cp_verify` also refuses the second (`KL-shift-range`); the first is a rule that only `cp_prove` has; the kit's line `table` is the one that `tinyllama_plan.table_numbers()` gives for the rows that upstream's `layoutW()` counts |
| upstream's scale tests | `test_policy_tables.py`, `table_ties.py`, `test_calibrate.py` and `test_gguf_weights.py` pass on the patched tree, the first two with `SUITE.md`'s commands (the other two have no row). Status 77, passed in part, is accepted with an "in part" last line: the target builds no `make_model` and sets no `SMALL_MODEL`, so those cases skip on every host |
| float reference | upstream's float pass on the GGUF agrees with Hugging Face transformers on the safetensors at the final norm (relative L2 at most 1e-4), and transformers' greedy decoding gives SASH's answer |
| **the answer** | upstream's integer model (patched, torch 2.13.0), with every file pinned and the weights' integers checked against `wsums.txt`, run over the whole 64-token window: no range condition fails at any position; the greedy choices at positions 24 to 31 are SASH's `[1576, 7483, 310, 3444, 338, 3681, 29889, 2]`; the gate's key stays at most 32,639 (the integer model's condition for a key without the clamp `gcl`; it is meant to match the provers' refusal, but upstream has not compared its range tests with the released `cp_prove`); and at each checked position the integer logits over 2^s have a slope of 0.75 to 1.33 and a correlation of at least 0.998 against the float logits. The slope bound catches an error in the logits' overall scale (`MF` off by one bit doubles every logit). The correlation floor, 0.998, was set by hand after samples showed one-bit shift errors passing the earlier 0.99: between the kit's 0.9991 (then at `eg 8`) and the layer-10 control's least, 0.9942 (both in the [earlier run](../../../evidence/flashprove-tinyllama-answer-2026-11-23.json)). A one-off sweep on macOS ([`shift_sweep.py`](shift_sweep.py)), which changes every layer's `kv`, `ko` and `kd` by one bit, 66 kits, caught all 66 on the `eg 7` kit (4 changed the tokens; the other 62 kept SASH's tokens and slopes within the bound, and fell below the floor, the closest at 0.9963). With the earlier floor of 0.99, 28 of the 66 would have passed. The margin is thin, and other errors may still pass |
| mutation controls | the same check on two one-bit changes of the kit must fail at least one condition: `MF` + 1 in `policy.txt`, and layer 10's `kv` + 1 in `KL.bin` (not the line `s`: the forward pass reads it only to add seeded noise, and this check gives no seed, so changing it would test only this target's divisor); each control's slopes and least correlation are recorded |
| `cp_commit`'s worst-case check, on a CPU | `worst_case_cpu.py` and `gate_float_bound.py` (next section) on the kit: every site must fit (`kit.json`'s `worst_case.short` is empty), and the two must agree on the gate: the same layers over the limit (none), and in each layer the integer key bound must be the float one times the norm's ball radius over 2,048, within 0.1 % |

**The sized kit** (`kit/policy.txt` with the line `table 11 24 1376256 65535 0 18`; `KL.bin`,
`layout.txt` and every other kit file unchanged): **passed on Linux** on 2026-11-25 at `40186c6`,
every row as required, on a clean clone with every input downloaded, 1,959 s
([evidence](../../../evidence/flashprove-tinyllama-gpu-2026-11-25.json)). The kit rebuilt byte for
byte; the integer model's figures are those below (SASH's eight tokens, the gate's key at most
3,970, correlations of at least 0.99908, both controls failing as before); upstream's `layoutW()`
counts 10,240 rows and `HfParams::from_policy()` admits the line; the planner's bounds are a proof
of at most 13,685,772 bytes and 126.21 bits (its own accounting, not a security level), with 58
columns a batch. On macOS arm64 it passed at the same commit (234 s, model files from a checked
local folder; the kit rebuilt, SASH's eight tokens, no site short), run by an AI-assisted review session on the contributor's laptop (not an independent human reviewer).

**The kit before the table was sized** (`scales.patch`, the lock files and `kit/` as then pinned in
`kit.json`, at `eg 7`):
**passed on Linux and on macOS** on 2026-11-24 at `bf3add1`, every row as required
([evidence](../../../evidence/flashprove-tinyllama-answer-2026-11-24.json), with both runs'
`result.json`): Ubuntu 24.04 x86_64 (a clone, every input downloaded, 1,870 s) and macOS arm64
(inputs from a checked local folder, 377 s).

- both environments hold exactly their lock files (43 and 19 distributions);
- every required kit file, the GGUF file and the calibration set rebuilt byte for byte on both
  hosts; `maxima.json` (recorded only) equals the committed one on macOS and differs in its last
  digits on Linux, as upstream's `scales/README.md` expects across devices, while `words.json` (the
  scales) is the same; the calibration set's drawn rows have upstream's `DRAWS_SHA256`;
- float reference: relative L2 3.9e-6 (macOS) and 1.3e-6 (Linux) at the final norm; transformers'
  greedy reply is SASH's;
- the integer model: SASH's eight tokens, no failed range condition over the 64 positions, the
  gate's key at most 3,970, with every recorded peak, margin and clamped cell identical on the two
  hosts (the slopes and correlations, against each host's own float pass, differ by at most 1.1e-6);
  logit slopes 0.993 to 1.005 and correlations of at least 0.99908; the smallest top-1 margin 0.77
  logits, at the full stop; one value entry beyond its clamp, `vw` in layer 2 at
  position 32 (the final EOS), 2,241 against 2,047;
- the mutation controls, identical on the two hosts: with `MF` + 1 the tokens stay and the slope is
  1.99 to 2.01 at every checked position; with layer 10's `kv` + 1 the tokens stay, the slopes
  move to 0.982 to 1.008 (within the bound), and the correlation falls to 0.9939, below the floor;
- the worst-case check, identical on the two hosts: every site fits (next section);
- the planner's bounds for that kit, with the full kit's line `table`: a proof of at most
  14,061,196 bytes, 126.21 bits, the same as check 6 then.

## cp_commit's worst-case check, on a CPU

Before it commits, `cp_commit` computes upper bounds, over all prompts, on every value of the
forward pass that a prover's cell must hold, and makes no commitment (status 7) if one can leave its
range (upstream's `prover/weights_commitment/worst_case.cuh`, derived in
`scales/worst-case-bounds.md`). Every commitment made from model files runs it, `-public-weights`
included (only a test build's `-test-seed` leaves it out), and `cp_prove` does not start without
its record (`isk.txt`, `ranges`). Upstream computes it on a GPU only. Step 7 of
`make verify-flashprove-tinyllama-answer` computes it on a CPU:

- [`worst_case_cpu.py`](worst_case_cpu.py), Hallmark's port of `worst_case.cuh` for NE 0, on the
  committed integers (read by the integer model, checked against `wsums.txt`). The sites that a
  row's bound decides (q, k, v, gate, up) are computed as `k_row_bounds()` computes them: every row
  in floating point, and the rows that can decide a site again in pure Python with the GPU's
  rounding upwards emulated exactly (the two must agree within 1). The norms' quotients and the
  head's margin follow from public parameters, as in `lines()`. The stream's sites use Frobenius
  norms where the GPU uses its bound on the largest singular value: an upper bound on the GPU's, so
  a stream site that fits here fits there.
- [`gate_float_bound.py`](gate_float_bound.py), which shares no code or file with it: from the float
  safetensors, sqrt(d) ||W_gate[j] * g|| is the largest input of gate row j over all normed rows.

**At the default `eg 8` a kit fails at one site** (the kit as first fitted; the same on both
hosts, to the last digit;
[`retained_failures`](../../../evidence/flashprove-tinyllama-answer-2026-11-24.json)):

| site | worst layer | bound | limit | |
|---|---|---|---|---|
| gate | 21 (shift `kg` 10) | a key of 47,481 | 32,639 | over by 0.54 bits (a shortfall of 1 whole bit); one row of 5,632 |
| gate, the other 21 layers | 2 | a key of 14,631 | 32,639 | fits |
| q, k, v, up | 20, 0, 3, 21 | | | fit, 2.97, 1.53, 1.20 and 1.28 bits to spare |
| the stream, by Frobenius norms | after the last layer | 2^35.25 | 2^38.99 | fits |
| the norms' quotients, the head's margin | | | | fit (the margin by 0.64 bits) |

The float bound agrees: row 1,104 of layer 21 can reach a gate input of 177.4 (a key of 45,406 at
`eg` 8), no other layer above 54.7. In every layer the integer key bound is the float one times the
norm's ball radius over 2,048 (1.0457), within 2e-4, as it should be when the kit's scales are right.
A real request is far below it: on SASH's prompt the gate's key reaches at most 7,108 at `eg 8`
([earlier run](../../../evidence/flashprove-tinyllama-answer-2026-11-23.json)) and 3,970 at `eg 7`.

What follows:

- No change of the scales at `eg 8` fixes it. In real units the bound is the row's own,
  sqrt(d) ||W_j * g||, for any scales; `scales/worst_case.py` cannot make a gate's shift coarser
  (its key reads a public table with a fixed step).
- Upstream's remedy is the line `gcl 1`, which clamps the gate's key and raises its limit to
  4,194,175. The clamp is read from the sigmoid table (`FS`), which only a model with experts forms,
  and the verifier refuses `gcl` at NE 0 (`policy-gcl`); the patches keep that refusal.
- Upstream's other remedy is a coarser key: each step of `eg` below 8 halves the key and raises the
  gate's limit by one bit (`scales/worst_case.py`; `fit_scales.py --eg`), but any `eg` takes a file
  out of the standard parameter form (`no-range-words`). So `verifier.patch` admits `eg 7` at NE 0,
  and the kit is fitted at `eg 7`: [`fit_eg.py`](fit_eg.py) makes `calibrate.py`'s call of
  `fit_scales.fit()` again at `eg 7` (at `eg 8` it gives `calibrate.py`'s `words.json` byte for
  byte), which raises each layer's `kg` by one and changes nothing else.

**At `eg 7`, the committed kit: every site fits** (both hosts, to the last digit):

| site | worst layer | bound | limit | |
|---|---|---|---|---|
| gate | 21 (shift `kg` 11) | a key of 23,741 | 32,639 | fits, 0.45 bits to spare; no other layer's above 7,316 |
| q, k, v, up, the norms, the head's margin | | | | as above |
| the stream, by Frobenius norms | after the last layer | 2^35.25 | 2^38.99 | fits |

The float bound at `eg 7` puts that row's key at 22,703; no layer is over the limit. The cost is the
SiLU table's input resolution, 2^-7 in place of 2^-8: on SASH's request the logits' correlation with
the float logits stays at least 0.99908 (0.99911 at `eg 8`, in the earlier run), and the smallest top-1 margin is
0.77 logits.

- Not covered: the GPU's guards on its matrix products, which can only make a site fail; and that
  `cp_commit`'s own conversion gives the integers read here (it is compared with `wsums.txt` only
  through that file). The GPU's own table, from one run, agrees with the port ("The first dense
  proof").

## The first dense proof

On 2026-11-24 and 2026-11-25 (UTC) the sized kit went through `cp_commit`, `cp_prove` and
`cp_verify` on a rented H100 ([evidence](../../../evidence/flashprove-tinyllama-gpu-2026-11-25.json)).
**This was the contributor's own run**: on the contributor's own RunPod account, paid by the
contributor, with no owner-set cap. #184 says that paid GPU runs need an owner-approved budget, #186
puts unbudgeted GPU runs out of scope, `docs/OPERATIONS.md` says to stop before a paid job or an
upload for which no authority exists, and `research/AUTHORITY.json` lists Modal, Vast.ai and Hugging
Face Jobs as compute providers, not RunPod. No authority record was made or changed for it. It
spent USD 1.6056 (0.3308 and 1.2748 for the two pods), by the fall of the account's balance,
read again unchanged at 02:44 UTC on 25 Nov. A reading right after pod 2 ended, before the
provider's last charge, would give 1.1387 for that pod; by rate and seconds the pods come to
0.3305 and 1.2719. `AGENTS.md` says no system-package installs: the pod, a rented machine,
had `apt-get` and `pip` installs. It is
evidence for the proposal, not an approved Hallmark run.

- **Programs.** `prover/build.sh serving_cp_commit` and `serving_cp_prove` built the two provers
  from the pinned archive with the three patches (sources' digest `7f304c42…`, nvcc 13.2.86) on a
  CPU host; `cp_verify` is the patched build (`a163a90a…`). They were built below a world-writable
  directory, so `build.sh` set no run path. On the pod, `pod/bootstrap.sh` adds the folder of the
  driver's `libcuda.so.1` to the loader's configuration; how the programs found the CUDA runtime
  that `pod/user_setup.sh` installs is in no kept script (the session's notes say that its folder
  was added with `ldconfig`), and that script's own comment, which speaks of a run path, is wrong
  for these programs. Nothing but public files went to the pods: the programs, the kit,
  the GGUF file, the request and upstream's published seed file
  (`examples/maverick/public-seed.bin` of the pinned archive, 32 bytes, SHA-256
  `630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd`).
- **How the pods were driven.** The contributor's AI-assisted session created and terminated the
  two pods through RunPod's API from another host, with a watchdog there set to terminate every
  pod of the account after 2,700 s (pod 1) and 3,600 s (pod 2). On each pod it ran
  [`pod/bootstrap.sh`](pod/bootstrap.sh) as root (host checks, system packages with `apt-get`, an
  ordinary account), then as that account [`pod/user_setup.sh`](pod/user_setup.sh) (the CUDA
  runtime by `pip`) and the worker; the three timed proofs ran through
  [`pod/run3.sh`](pod/run3.sh). The three files are kept as they ran; no target runs them.
- **Worker.** [`gpu_job.sh`](gpu_job.sh), run as an ordinary account, with upstream's commands:
  `cp_commit -weights … -config … -M 1 -ivk … -isk … -public-weights <seed>
  -worst-case-table …`, `cp_prove -isk … -public-weights <seed> -in … -out …` and the client's
  `cp_verify <config> <ivk> <request> <y> <proof> -id <ID>`.
- **Hosts.** Two pods of one type, one after the other, each with one H100 SXM 80 GB (sm_90),
  driver 595.112.07, a Xeon Platinum 8470 (SHA instructions), Ubuntu 24.04, 29 vCPU and 287 GB for
  the container; CUDA runtime 13.2.86 and cuBLAS 13.4.1.3 from `cuda-toolkit[cublas,cudart]==13.2.2`,
  a subset of upstream's documented pip install. Whether the two pods were one machine is not
  recorded.
- **`cp_commit`** ran three times (once alone, twice before proofs) and each time ended with status
  0, "the worst-case check: every bounded site fits", in 34.3 to 35.0 s of wall time at a peak of at
  most 4,252,728 KiB (4.4 GB) of host memory. All three gave the same commitment ID,
  `5ffd1aa39d64f2211de266442dd69b851b0b1e611a19dbbb6b5f2acf4d0406bd`, and the same `key.bin`
  (2,148,512 bytes) and leaf file (2,181,432 bytes): with `-public-weights` and a public seed the
  commitment can be recomputed by anyone with such a GPU, the public weights and the kit.
- **The GPU's worst-case table against the CPU port** ([`compare_worst_case.py`](compare_worst_case.py)):
  the same 202 sites, none short. At the 156 sites that the port computes as the GPU does (q, k, v,
  gate, up, the norms' quotients, the head's margin) the bounds agree within 0.001 bits, the table's
  resolution; at the stream's 46 sites the port's Frobenius bound is at or above the GPU's within that
  resolution (by up to 7.93 bits above), as it is meant to be. The GPU's tightest sites: the gate of layer 21 (0.458 bits to spare),
  the head's margin (0.642), v of layer 3 (1.199), up of layer 21 (1.280).
- **`cp_prove`** made four proofs of one request (SASH's 25 prompt ids and 8 answer ids in this
  kit's 64-token window, with 31 filler tokens; r = 24, e = 32), each a call of its own after a commitment: 8.41 s, and 9.79, 9.67 and 9.31 s
  with `nvidia-smi` sampling the GPU twice a second beside it; at most 3,650,064 KiB (3.7 GB) of host
  memory; proofs
  of 13,238,500 to 13,253,292 bytes (the planner's bound: 13,685,772). The largest GPU memory
  sampled over a commitment and three proofs was 74,799 MiB; which program held it was not
  separated.
- **`cp_verify`** on the pod answered `YES` to each of the four under the commitment ID
  (`claim=exact-match`, `tokens=25..32`, `y=no`, the patched build's digest), in about 12 s each.
- **The caller's check of the first proof, on a CPU.** [`scripts/flashprove_tinyllama_proof.py`](../../../../scripts/flashprove_tinyllama_proof.py)
  rebuilds both verifiers from the pinned archive and requires: the configuration is the committed
  kit and the request SASH's, byte for byte; upstream's `check_proof.py` with the patched
  `cp_verify` accepts under the caller's commitment ID, range `24:32`, end token 2, exact-match
  claim and build digest; a plain call ends `YES`; the unpatched `cp_verify` ends
  `NO: policy-tables`; and eight altered copies each end `NO:` with status 1. Seven of them are
  refused by the checks of the proof, the key, the leaf file or the ID; the token file's copy is
  refused on the statement's form, by a rule that does not depend on the proof's content. It passed on Linux x86_64 (g++ 13.3,
  724 s):

  | altered | the patched verifier's last line |
  |---|---|
  | a bit in the middle of the proof | `NO: proof-refused` |
  | the proof's last bit | `NO: proof-refused` |
  | the token file (y, which this request does not cover) | `NO: statement-range` |
  | the first prompt token | `NO: proof-refused` |
  | the first answer token | `NO: proof-refused` |
  | a bit of `key.bin` | `NO: key-refused` |
  | a bit of the leaf file | `NO: entry-refused` |
  | another commitment ID | `NO: model-statement` |

  An AI-assisted review session on the contributor's laptop (not an independent human reviewer)
  ran the same script on the same files on macOS arm64 (Apple clang 17) at `bddda05`:
  passed in 155 s with the same eight last lines; its `result.json` names the same file hashes as
  the Linux one. That run was not made in the session that made the proof.

  The script as it then was took any `NO:` as a refusal and compared the caller's folder before
  copying it. It now requires each copy's own refusal (the table above) and compares the copy that
  the verifiers read; it passed again on both hosts on the same files, and on Linux on the first of the three later
  proofs, which differs in size and got the same eight refusals
  ([follow-up evidence](../../../evidence/flashprove-tinyllama-followup-2026-11-25.json), which also holds both
  earlier `result.json` files and the pods' `inputs.sha256` and `binaries.sha256`). The commitment
  ID that the check is given is `cp_commit`'s own print on the pod: the check does not recompute it,
  so a pass does not tie the proof to TinyLlama's weights. The proof, key and leaf files are held by
  the contributor and are not published, so nobody else can repeat the check yet.

What a pass shows is in decision 0094, "What a patched `YES` means". These times are those of two pods of one type,
on lines of which only `table` is sized for this model: not a benchmark.

**Retained failures of the run** (none after a proof existed): the target's verifier build
refused a clone below a group-writable directory twice (upstream's `verifier/build.sh` asks for
directories that only the caller can change; `chmod go-w`); the first pod's first bootstrap
stopped silently at the package step and passed when run again (cause not established); a command
line of the controlling session's own for the three timed proofs ended with status 127 before anything ran
and held the pod for about ten minutes (about USD 0.75 of the spend).

## Not checked

- **The prover patch has run on one request, one kit and two pods of one type.** `cp_commit` and `cp_prove`
  made a commitment and four proofs at NE 0 (above). Another prompt, window length or model can
  reach code that this run did not: the pinned repository holds no NE 0 kit, and nothing here
  tests the patched CUDA code beyond that run.
- **The patched tree's own documents still state upstream's rules.** No patch touches
  `verifier/docs/parameter-file.md`, `docs/format/FORMAT.md` 13.7 or `docs/model.md` 11.3, so in a
  patched tree they still say that NE 0 never ends in acceptance and that c_0 = RN32(sqrt(d) / 64).
  This README and decision 0094 are the record of what the patches change.
- **Tests that need x86-64 SHA instructions or Linux did not run on the Mac.**
  - Upstream's `weight_sums/tests/run_tests.sh`;
  - `integer_model`'s `tests/test_gguf_files.py` (it builds `make_model`);
  - the prover rows `parameter_file_one_bits_test` and `parameter_file_one_place_test`.
- **The commitment is a test commitment.** It was made with `-public-weights` and upstream's
  published seed file, so it hides nothing, and its ID (`5ffd1aa3…`) is the caller's own: nothing
  enrols it, and how a verifier would tie an ID to the public checkpoint is not designed (decision
  0094). The kit's `fe` (below) and its proof-system lines enter that ID and may still change.
- **The kit's proof-system lines are check 6's**, not sized for speed: the full kit's values, with
  the lines that TinyLlama's shape forces (`ll`, which 22 layers force from 2 to 1; `la`, `lB`,
  `lmS`, `iotaA_0`, `iotaH_0` to `iotaH_2`, `ranges-part`) set from the verifier's own rules
  and refusals, and `kFM` and `kFD` (14 and 15) by the rule of `cp_prove`'s plan of attention
  (`tinyllama_plan.py`, `softmax_rows()`; `cp_verify` does not read `kFM`). The fit changes only
  the scales and the six lines that depend on them. They are placeholders that `cp_prove` has
  now judged on one request (above). Only the line `table` is sized for this model, by criteria read off upstream's kits and
  not for speed; the times above are two like pods' on these lines, not a benchmark, and #187 may
  want smaller lines.
- **One request.** The answer check covers SASH's one prompt in one window of T = 64. On it one value
  cell clamps (`v` in layer 2, 2,241 to 2,047, at position 32, the final EOS, after the checked
  positions); another prompt
  can clamp other cells or fail a range condition. #187's sweep over the number of checked tokens needs a
  kit for each T.
- **Equal tokens are a smoke test.** They show that this integer model agrees with SASH's answer on
  this request, not that FlashProve's and SASH's integer functions are equal. The mutation controls
  show that the bounds catch two kinds of one-bit error; other single errors may still pass.
- **The calibration text is WikiText-2** (S. Merity et al., "Pointer Sentinel Mixture Models",
  arXiv:1609.07843), as in upstream's rule, at the revision upstream pins. Licence: CC BY-SA (its
  dataset card's tags name CC BY-SA 3.0 and GFDL, its text CC BY-SA 4.0). No text and no token id
  from it is committed, but numbers computed from it are: `kit/maxima.json`, `words.json`, `KL.bin`,
  six lines of `policy.txt`, and the calibration set's SHA-256 in `kit.json`. The owner has not
  cleared this source: the card's tags and its text name different versions, and
  `research/AUTHORITY.json` excludes ambiguous licences. TinyLlama's `tokenizer.json`, which the calibration set is made with, is byte for byte the Llama 2
  tokenizer that upstream's rule pins; upstream's `THIRD_PARTY.md` calls that file Meta's, from a
  gated repository, and `research/AUTHORITY.json` has `gated_terms: false`. Here it comes from
  TinyLlama's own ungated repository, at the revision of Hallmark's SASH path; whether that settles
  it is the owner's call too. Until the owner decides, the kit and the
  answer check are a proposal; a no means refitting on another text. The three program patches do not depend on it;
  `scales.patch` pins the calibration set's SHA-256 (`TINYLLAMA_SET_SHA256`) and would change with
  the text, and so would its pin in `kit.json`.
- **The environments pin every distribution by hash, but not Python's patch version** (uv's choice
  of 3.13, which uv may download if the host has none; the retained runs had 3.13.15) nor uv's own
  version, and only for macOS arm64 and Linux x86_64. They are installed from PyPI and, for torch on
  Linux, from PyTorch's CPU index.
- **The dense proofs' checks are few.** The 108 equal calls of `make verify-flashprove-tinyllama`
  are all on upstream's MoE proofs. The verifier's NE 0 branches have checked four proofs of one
  request and refused eight altered copies of one of them (above). That is far from upstream's 99
  altered sets, and no table of altered items has been made for a dense proof.
- **Only the patched verifier answers `YES`.** Upstream's `cp_verify` refuses the same files
  (`NO: policy-tables`), and Lean covers neither form. Whoever relies on a dense `YES` relies on
  Hallmark's patch.
- **The worker as it now stands has not run on a GPU.** The pods ran `gpu_job.sh` as at `bddda05`
  (pod 1 ran the working-tree file that was committed as `bddda05` one minute after that pod
  ended). `35c7dde` then made it require that the first GPU is sm_90 and that `cp_verify`'s last
  line is `YES`, and later commits made it refuse a `PROOFS` that is no whole number below 1,000, check the
  host before it makes its work folder, compare an optional `expected.sha256` (strictly, and it
  must name the kit's three files, the GGUF file and the seed) and an existing marker file. The changed script has run only against stand-in programs on a Linux host without a
  GPU ([follow-up evidence](../../../evidence/flashprove-tinyllama-followup-2026-11-25.json)). It
  still has no time limit of its own, and unless `expected.sha256` is given it compares no input
  with a pin.
- **What three reviews of 2026-11-25 found and this branch does not fix** (the patches are left as
  they were when the proofs were made, so that the recorded runs stay those of the pinned files):
  - Upstream's own text counts a model without routed experts among the forms that "cannot pass"
    and marks that as read, not tested (`lean/OnBytes/LIMITS.md`). The patched verifier's `YES` at
    NE 0 therefore rests on branches that upstream never ran against a lying prover: the gate's
    two-limb tensor, the key built from it and its quotient bound. They were read by two reviews
    and no fault was found; nothing more stands behind them than that and one request's proofs.
  - For a file with experts the patched verifier accepts and refuses exactly what upstream's does,
    but the name after `NO:` can differ for a malformed file with two faults (three digest groups
    are now refused after the layout, not in the scan).
  - A `YES` does not show that the model is dense or that its SiLU key is `eg 7`: upstream's
    verdict names the open optional lines, and the patch adds none. The commitment ID covers both.
  - No test case in `verifier.patch` gives `eg 7` with the `eg 8` SiLU digest, `eg 7` without
    digests, `smax` alone at NE 0, or `eg 4` and `eg 5`; all go through the tested expression.
  - In `tools.patch` the integer model's hint for a dense file without digest lines says that
    `policy_tables.py` does not write them at NE 0; with `scales.patch` it does. Its constants
    test compares the two literals with themselves, and its dense fixture keeps call lines that
    `cp_verify` would refuse. Upstream's `table_ties.py` says it covers every table of the standard
    form and does not look at `eg 7` (each `eg 7` entry is an `eg 8` entry: T7[c] = T8[2c], clamped).
  - Hallmark's own tests leave most failure branches of the three scripts unguarded: a review
    reported that 60 of 93 one-line breakages passed the four test files as they then were (its
    run is not in the repository). A few tests
    were added since; `export_gguf.py`, `fit_eg.py`, `float_reference.py`, `answer_check.py`,
    `gate_float_bound.py`, `gpu_job.sh` and `prover_setup.cpp` still have none of their own and
    are exercised only by the targets.
  - A checkout whose path holds a space breaks the stand-in verifier of the first target and four
    tests (the interpreter line of a generated script).
  - `worst_case_cpu.py` does not itself repeat the refusals of `cp_commit`'s reader of the scale
    words (`Check::read()`: the ranges of the shifts and of the norms' words, an explicit `eg 8` or
    `xl 4`); a kit must pass the provers' reader and the set-up check for those.
  - The comparison of the GPU's worst-case table with the port has the table's resolution, 0.001
    bits (about 18 either way on the key of 23,741): it shows agreement to that, not that the port's
    exact rounding is the GPU's.
- **The macOS runs on the sized kit were made by an AI-assisted review session** on the
  contributor's laptop, not by an independent reviewer. The result files of the runs at `40186c6`
  are held by the contributor; those of the later reruns are in the
  [follow-up evidence](../../../evidence/flashprove-tinyllama-followup-2026-11-25.json).
- **The float reference is Hugging Face transformers 5.15.1**, run as a check of the export. It is
  not part of any statement.
- **The kit's `fe` is the base file's 8,192, Maverick's expert width.** At NE 0 the verifier reads
  it and uses it for nothing, and the provers' reader only requires it to be at most 2^20
  (`cp_commit`'s worst-case check also sizes a buffer by it). But it is a line of the parameter
  file, so it enters `h_par`: any other value would name the same model differently. It was not
  set on purpose (the dense reader's shape says 0) before the first commitment, whose ID therefore
  carries 8,192. At NE 0 nothing ties `fe` to the model: a dense model has one valid parameter file,
  and one commitment ID, for each value of it.

## Prior work

[`marcellobardus/FlashProve-TinyLlama1.1B`](https://github.com/marcellobardus/FlashProve-TinyLlama1.1B)
at `1a94f01` (Apache-2.0) reports two single-token proofs of a dense model at its own shape, by its own
logs, which Hallmark has not reproduced. Hallmark uses none of its code, proofs, kits, seeds or
commitments, and these patches were written from upstream's sources and documents.

Two differences from it:

- its verifier keeps a count per table, where this patch allows only the one gap where `tsig` would
  be;
- it changes `moe_at_layer`, which this patch finds unnecessary (above).

Its commit `1c5c2aa` sets `kFM` by the same rule of `cp_prove`'s plan of attention (14 at
TinyLlama's shape), as `tinyllama_plan.py` does.
