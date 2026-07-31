#!/bin/bash
# nscan_sweep (2026-07-08): N-dependence of the frustration capacity knee B0(n).
#
# At N=120 the knee obeys  B0 ~ 1/p*(n) ~ (2n+1)^1.9,  with p*.B0 ~ N/15.
# This tests whether B0 ∝ N (i.e. B0/N depends only on n) across an 8x ladder
# N in {120,240,480,960} on random 3-regular graphs, distance-6 pairs, M=8.
# Key enabler: the --markov forward (no N*K^M state), so N=960 -- previously
# OOM on the AR joint state -- is now reachable.
#
# Design: a FIXED B/N grid at every N so acc-vs-(B/N) curves overlay; if the
# knee sits at fixed B/N the law B0 ∝ N holds.  n=1,2,3 for N<=480; n=1,2 for
# N=960 (its deeper knees would need B>2N).  seeds 1-4.
#
# markov forward is ~1GB so 2 concurrent shards on one GPU are fine; OMP=4
# each keeps within a ~16-core budget.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet
export MARKOV=1 GPU=0

run() {  # run <N> <poolB> <objlist> <Blist>
  local N=$1 POOL=$2 OBJS=$3 BL=$4
  export QA_PREFIX="randreg_N${N}_K3_M8_B${POOL}_D6"
  for OBJ in $OBJS; do
    B_LIST="$BL" OMP_NUM_THREADS=4 ./run_grover_sweep.sh 1 4 $OBJ \
      > nscan_N${N}_${OBJ}.log 2>&1
    echo "[nscan] N=${N} ${OBJ} done: $(date)"
  done
}

# Group A (gpu0): N=120 then N=480
(
  run 120 384  "grover grover2 grover3" "8 16 32 64 96 128 192 256 320"
  run 480 1344 "grover grover2 grover3" "32 64 128 256 384 512 768 1024 1280"
  echo "[nscan] GROUP A done: $(date)"
) >> nscan.log 2>&1 &
PA=$!

# Group B (gpu0): N=240 then N=960 (n=1,2 only)
(
  run 240 704  "grover grover2 grover3" "16 32 64 128 192 256 384 512 640"
  run 960 1600 "grover grover2"         "64 128 256 512 768 1024 1536"
  echo "[nscan] GROUP B done: $(date)"
) >> nscan.log 2>&1 &
PB=$!

wait $PA $PB
echo "[nscan] ALL DONE: $(date)" >> nscan.log
