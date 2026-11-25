#!/bin/bash
# Three timed proofs after a fresh commitment, with the GPU's memory sampled twice a second.
cd /var/tmp/fp || exit 2
rm -f gpu-mem3.txt
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done > gpu-mem3.txt 2>/dev/null ) &
sampler=$!
bash stage/gpu_job.sh stage /var/tmp/fp/work3 3
status=$?
kill "$sampler"; wait "$sampler" 2>/dev/null
cp gpu-mem3.txt work3/gpu-memory-samples.txt
echo "job exit=$status; GPU memory samples: $(grep -c . gpu-mem3.txt), most $(sort -n gpu-mem3.txt | tail -1) MiB"
