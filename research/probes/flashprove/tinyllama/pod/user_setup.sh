#!/bin/bash
# As the ordinary account: the CUDA runtime libraries at the path the programs' run path names
# (upstream's documented pip install; research/probes/flashprove/job.sh, phase prove).
set -euo pipefail
cuda=/var/tmp/fp/cuda-13.2.86
[ -d "$cuda" ] || { mkdir "$cuda"; PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q --no-cache-dir --target "$cuda" 'cuda-toolkit[cublas,cudart]==13.2.2'; }
PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip list --path "$cuda" --format json
chmod -R go-w "$cuda"
ls "$cuda/nvidia/cu13/lib" | tr '\n' ' '; echo
ldd /var/tmp/fp/stage/hc/cp_prove | grep -E 'libcuda|libcudart|libcublas|not found' || true
ldd /var/tmp/fp/stage/hc/cp_commit | grep -E 'not found' && exit 4
echo "user setup ok"
