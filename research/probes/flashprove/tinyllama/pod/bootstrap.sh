#!/bin/bash
# Root side of the pod: host checks first (cheap, before any long step), then an ordinary account.
set -euo pipefail
echo "== host checks"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader | tee /root/gpu.txt
driver=$(cut -d, -f3 /root/gpu.txt | tr -d ' ' | cut -d. -f1)
cap=$(cut -d, -f2 /root/gpu.txt | tr -d ' ')
[ "$cap" = 9.0 ] || { echo "CHECK FAILED: compute capability $cap, not 9.0"; exit 3; }
[ "$driver" -ge 580 ] || { echo "CHECK FAILED: driver $driver is below 580"; exit 3; }
[ "$(grep -c -w sha_ni /proc/cpuinfo)" -gt 0 ] || { echo "CHECK FAILED: no SHA-NI"; exit 3; }
grep -m1 'model name' /proc/cpuinfo; nproc
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
cat /sys/fs/cgroup/memory.max 2>/dev/null || true
df -h /var/tmp | tail -1
. /etc/os-release; echo "$PRETTY_NAME"; ldd --version | head -1
echo "== packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3 python3-pip time rsync ca-certificates libstdc++6 >/dev/null
echo "== loader"
if ! ldconfig -p | grep -q 'libcuda.so.1 '; then
  dir=$(dirname "$(find / -xdev -name 'libcuda.so.1' 2>/dev/null | head -1)")
  echo "libcuda.so.1 is not in the loader's configuration; adding $dir"
  echo "$dir" > /etc/ld.so.conf.d/nvidia-driver.conf
  ldconfig
fi
ldconfig -p | grep 'libcuda.so.1 '
echo "== account"
id fp >/dev/null 2>&1 || useradd -m -s /bin/bash fp
install -d -o fp -g fp -m 700 /home/fp/.ssh
install -o fp -g fp -m 600 /root/.ssh/authorized_keys /home/fp/.ssh/authorized_keys
install -d -o fp -g fp -m 755 /var/tmp/fp
echo "bootstrap ok"
