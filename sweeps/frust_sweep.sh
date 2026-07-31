#!/bin/bash
# frust_sweep: frustration-scaling study. N=120 sliding
# puzzle, distance-6, pushed to larger B on a 768-pair pool so the deep-n
# knees m0(n) become visible (they are censored at B<=128 on the old 192-pool).
#
# Measures acc(B, 2n+1) = mean A_n after n rounds vs B, for n=1..4, to fit the
# capacity threshold m0(n) (where the plateau ends) and its slope.  Single-GPU
# memory caps B at ~256 (the AR K^M=6561 joint state); shard n=1,2 and n=3,4
# across two GPUs if available, seeds 1-4, B in {8..256}.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # avoid fragmentation at B>=192
export QA_PREFIX=sliding_puzzle_N120_K3_M8_B768_D6
export B_LIST="8 16 32 64 96 128 192 256"

(
  GPU=0 OMP_NUM_THREADS=4 ./run_grover_sweep.sh 1 4 grover  > frust_g1.log 2>&1
  echo "[frust] gpu0 n=1 done: $(date)"
  GPU=0 OMP_NUM_THREADS=4 ./run_grover_sweep.sh 1 4 grover2 > frust_g2.log 2>&1
  echo "[frust] gpu0 n=1,2 done: $(date)"
) >> frust.log 2>&1 &
P0=$!

(
  GPU=1 OMP_NUM_THREADS=4 ./run_grover_sweep.sh 1 4 grover3 > frust_g3.log 2>&1
  echo "[frust] shard n=3 done: $(date)"
  GPU=1 OMP_NUM_THREADS=4 ./run_grover_sweep.sh 1 4 grover4 > frust_g4.log 2>&1
  echo "[frust] shard n=3,4 done: $(date)"
) >> frust.log 2>&1 &
P1=$!

wait $P0 $P1
echo "[frust] ALL DONE: $(date)" >> frust.log
