#!/bin/bash
# One dense TinyLlama commitment, its proofs and their check on an sm_90 GPU host, with upstream's
# own commands (prover/examples/run_target_example.sh for the three programs and the work
# directory; examples/maverick/run.sh for a model's weight files under -public-weights). Run it as
# an ordinary account. Nothing is hidden: the seed file is public, as the weights are.
#
# usage: gpu_job.sh STAGE WORK [PROOFS]
# STAGE holds what was prepared without a GPU, all of it public:
#   hc/cp_commit, hc/cp_prove, hc/cp_verify   the patched programs (prover/build.sh, verifier/build.sh)
#   config/policy.txt, KL.bin, layout.txt     the kit's configuration
#   weights/<one file>.gguf, wsums.txt, fold.txt   the exported model and the kit's two setup files
#   example.req                               the request
#   public-seed.bin                           32 bytes, the published seed
#   expected.sha256                           optional: sha256sum lines, which must then hold before
#                                             any program runs and must name the three files of
#                                             config/, the .gguf file and public-seed.bin (the
#                                             caller writes them from kit.json's pins)
# WORK is created (mode 0700) and keeps everything: config/, ivk/ (public), isk/ (the provider's
# directory, which stays on the host), in/, out.N/, the logs, cp_commit's worst-case table, and
# public/ with the files a client needs, filled only after cp_verify's YES. PROOFS (default 1) is
# the number of cp_prove calls on the same request, each timed apart from the commitment; 0 stops
# after the commitment.
# Exit status: 0 after every YES, or after the commitment alone if PROOFS is 0; 3 if the call or the
# host does not fit (PROOFS is no whole number below 1,000, a missing or unexpected file, no SHA instructions,
# no nvidia-smi, the first GPU is not sm_90), before WORK is made, or if a marker beside the model
# file names another file; otherwise the failing step's.
# No step has a time limit: the caller's controller must end a host that hangs.
set -euo pipefail
stage=$(cd -- "${1:?usage: gpu_job.sh STAGE WORK [PROOFS]}" && pwd)
work=${2:?usage: gpu_job.sh STAGE WORK [PROOFS]}
proofs=${3:-1}
[[ $proofs =~ ^[0-9]{1,3}$ ]] || { echo "gpu_job.sh: PROOFS must be a whole number below 1,000" >&2; exit 3; }
seed=$stage/public-seed.bin
programs=(cp_commit cp_verify)
[ "$proofs" -eq 0 ] || programs+=(cp_prove)
for file in "${programs[@]/#/hc/}" config/policy.txt config/KL.bin config/layout.txt \
  weights/wsums.txt weights/fold.txt example.req public-seed.bin; do
  [ -f "$stage/$file" ] || { echo "gpu_job.sh: $stage/$file is missing" >&2; exit 3; }
done
gguf=("$stage"/weights/*.gguf)
[ "${#gguf[@]}" -eq 1 ] && [ -f "${gguf[0]}" ] || { echo "gpu_job.sh: one .gguf file is needed" >&2; exit 3; }
[ "$(stat -c %s "$seed")" -eq 32 ] || { echo "gpu_job.sh: the seed file is not 32 bytes" >&2; exit 3; }
if [ -e "$stage/expected.sha256" ]; then
  (cd "$stage" && sha256sum -c --strict --quiet expected.sha256) ||
    { echo "gpu_job.sh: a file of STAGE is not the one that expected.sha256 names" >&2; exit 3; }
  for file in config/policy.txt config/KL.bin config/layout.txt public-seed.bin "weights/$(basename "${gguf[0]}")"; do
    awk -v f="$file" '$2 == f && length($1) == 64 { found = 1 } END { exit !found }' "$stage/expected.sha256" ||
      { echo "gpu_job.sh: expected.sha256 does not name $file" >&2; exit 3; }
  done
fi
# The host must fit before WORK is made.
grep -qw sha_ni /proc/cpuinfo || { echo "gpu_job.sh: this CPU has no SHA instructions" >&2; exit 3; }
command -v nvidia-smi > /dev/null || { echo "gpu_job.sh: nvidia-smi is not found" >&2; exit 3; }
# The programs use the first GPU that they see.
[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')" = 9.0 ] ||
  { echo "gpu_job.sh: the first GPU is not sm_90" >&2; exit 3; }
mkdir -m 700 -- "$work"
cd -- "$work"
work=$PWD # cp_prove takes absolute paths
mkdir config in home tmp public

# The host, recorded before any long step.
grep -m1 -ow sha_ni /proc/cpuinfo > sha_ni.txt
grep -m1 'model name' /proc/cpuinfo > cpu.txt || true
nproc >> cpu.txt
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo > memory.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader > gpu.txt
cat cpu.txt memory.txt gpu.txt
(cd "$stage/hc" && sha256sum "${programs[@]}") > binaries.sha256
(cd "$stage" && sha256sum config/policy.txt config/KL.bin config/layout.txt weights/wsums.txt weights/fold.txt \
  example.req public-seed.bin "weights/$(basename "${gguf[0]}")") > inputs.sha256

echo "1. the public configuration and the request"
cp "$stage"/config/policy.txt "$stage"/config/KL.bin "$stage"/config/layout.txt config/
cp "$stage/example.req" in/example.req
# The declaration that the weights are public (examples/maverick/README.md, "Run it").
# A marker that is there already must name this file's SHA-256.
marker=${gguf[0]}.weights-are-public
sum=$(tail -n 1 inputs.sha256 | cut -d' ' -f1)
if [ -e "$marker" ]; then
  [ "$(cut -d' ' -f1 "$marker")" = "$sum" ] || { echo "gpu_job.sh: $marker names another file" >&2; exit 3; }
else
  (umask 022; set -C; printf '%s  %s\n' "$sum" "$(basename "${gguf[0]}")" > "$marker")
fi

echo "2. cp_commit: the worst-case check, then the commitment"
status=0
/usr/bin/time -v -o cp_commit.time "$stage/hc/cp_commit" -weights "$stage/weights" -config config -M 1 \
  -ivk ivk -isk isk -public-weights "$seed" -worst-case-table worst-case-table.txt > cp_commit.log 2>&1 || status=$?
echo "$status" > cp_commit.status
grep -E 'Elapsed|Maximum resident' cp_commit.time || true
if [ "$status" -ne 0 ]; then tail -n 5 cp_commit.log | cut -c1-600; exit "$status"; fi
id=$(sed -n 's/^commitment ID: //p' cp_commit.log | tr -d ' ')
[[ $id =~ ^[0-9a-f]{64}$ ]] || { echo "gpu_job.sh: cp_commit.log has no commitment ID" >&2; exit 1; }
echo "$id" > commitment-id.txt
echo "   commitment ID: $id"
if [ "$proofs" -eq 0 ]; then echo "done: the commitment alone (PROOFS is 0)"; exit 0; fi

for n in $(seq 1 "$proofs"); do
  echo "3.$n cp_prove"
  status=0
  /usr/bin/time -v -o "cp_prove.$n.time" env HOME="$work/home" TMPDIR="$work/tmp" "$stage/hc/cp_prove" \
    -isk "$work/isk" -public-weights "$seed" -in "$work/in" -out "$work/out.$n" > "cp_prove.$n.log" 2>&1 || status=$?
  echo "$status" > "cp_prove.$n.status"
  grep -E 'Elapsed|Maximum resident' "cp_prove.$n.time" || true
  if [ "$status" -ne 0 ] || [ ! -e "out.$n/example.proof" ]; then
    tail -n 5 "cp_prove.$n.log" | cut -c1-600
    ls -A "out.$n" 2>/dev/null || true
    exit $((status ? status : 1))
  fi
  echo "4.$n cp_verify, under the commitment ID"
  status=0
  /usr/bin/time -v -o "cp_verify.$n.time" "$stage/hc/cp_verify" config ivk in/example.req "out.$n/example.y" \
    "out.$n/example.proof" -id "$id" > "cp_verify.$n.log" 2>&1 || status=$?
  echo "$status" > "cp_verify.$n.status"
  tail -n 2 "cp_verify.$n.log" | cut -c1-600
  [ "$status" -eq 0 ] || exit "$status"
  [ "$(tail -n 1 "cp_verify.$n.log")" = YES ] || { echo "gpu_job.sh: cp_verify's last line is not YES" >&2; exit 1; }
  stat -c '%n %s' "out.$n/example.proof" ivk/key.bin ivk/leaf/0.bin
done

# Public files only, after a YES: isk/, home/ and tmp/ stay here.
mkdir -p public/config public/ivk/leaf
cp config/* public/config/
cp ivk/key.bin public/ivk/
cp ivk/leaf/*.bin public/ivk/leaf/
cp in/example.req out.1/example.y out.1/example.proof public/
(cd public && find . -type f | sort | xargs sha256sum) > public.sha256
echo "done: $proofs proof(s), each accepted"
