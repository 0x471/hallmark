#!/bin/bash
# The rented host's side of scripts/run_flashprove_tinyllama_gpu.py. The controller calls the steps
# in order, each under its own time limit, and copies STAGE (gpu_job.sh's, all of it public) to
# BASE/stage between the first and the second. Nothing here is trusted by the caller: the proof is
# judged again on the caller's own machine.
#
# usage, as root:  pod_job.sh host           host checks, the packages, an ordinary account, BASE
#                  pod_job.sh libraries      the CUDA runtime libraries (upstream's documented pip
#                                            install, as research/probes/flashprove/job.sh), the
#                                            loader's configuration, and STAGE handed to the account
#        as fp:    pod_job.sh prove PROOFS   gpu_job.sh on BASE/stage, into BASE/run/work
# The packages come from the image's own archive and from PyPI, as in decision 0093's worker; the
# host is one rented container that is deleted after the run.
set -euo pipefail
base=/var/tmp/fp
cuda=$base/cuda-13.2.86
case "${1:-}" in
host)
  mkdir -p "$base"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader | tee "$base/gpu.txt"
  cap=$(head -n 1 "$base/gpu.txt" | cut -d, -f2 | tr -d ' ')
  driver=$(head -n 1 "$base/gpu.txt" | cut -d, -f3 | tr -d ' ' | cut -d. -f1)
  [ "$cap" = 9.0 ] || { echo "pod_job.sh: compute capability $cap, not 9.0" >&2; exit 3; }
  [ "$driver" -ge 580 ] || { echo "pod_job.sh: driver $driver is below 580" >&2; exit 3; }
  grep -qw sha_ni /proc/cpuinfo || { echo "pod_job.sh: this CPU has no SHA instructions" >&2; exit 3; }
  grep -m1 'model name' /proc/cpuinfo
  nproc
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  df -h "$base" | tail -n 1
  . /etc/os-release
  echo "$PRETTY_NAME"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends python3 python3-pip time ca-certificates libstdc++6 > /dev/null
  id fp > /dev/null 2>&1 || useradd -m -s /bin/bash fp
  echo "host ok"
  ;;
libraries)
  [ -d "$cuda" ] || {
    mkdir "$cuda"
    PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q --no-cache-dir --target "$cuda" 'cuda-toolkit[cublas,cudart]==13.2.2'
  }
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip list --path "$cuda" --format json > "$base/cuda-packages.json"
  chmod -R go-w "$cuda"
  # The programs may carry no run path (upstream's build gives none below a directory that others
  # can write in), so the loader finds the runtime libraries and the driver's libcuda.so.1 by its
  # configuration.
  echo "$cuda/nvidia/cu13/lib" > /etc/ld.so.conf.d/fp-cuda.conf
  listed=$(ldconfig -p)
  if [[ $listed != *"libcuda.so.1 "* ]]; then
    driver=$(find / -xdev -name 'libcuda.so.1' 2> /dev/null | head -n 1 || true)
    [ -n "$driver" ] || { echo "pod_job.sh: the driver's libcuda.so.1 is not found" >&2; exit 4; }
    dirname "$driver" > /etc/ld.so.conf.d/fp-driver.conf
  fi
  ldconfig
  listed=$(ldconfig -p)
  for library in libcuda.so.1 libcudart.so.13 libcublas.so.13; do
    [[ $listed == *"$library "* ]] || { echo "pod_job.sh: the loader does not list $library" >&2; exit 4; }
  done
  for program in cp_commit cp_prove; do
    needs=$(ldd "$base/stage/hc/$program") || { echo "pod_job.sh: ldd failed on $program" >&2; exit 4; }
    if [[ $needs == *"not found"* ]]; then
      echo "pod_job.sh: $program lacks a library" >&2
      exit 4
    fi
  done
  chmod 755 "$base"
  chown -R fp:fp "$base/stage"
  install -d -o fp -g fp -m 755 "$base/run"
  echo "libraries ok"
  ;;
prove)
  cd "$base/run"
  bash "$base/stage/gpu_job.sh" "$base/stage" "$base/run/work" "${2:?PROOFS}"
  ;;
*)
  echo "usage: pod_job.sh host|libraries|prove PROOFS" >&2
  exit 2
  ;;
esac
