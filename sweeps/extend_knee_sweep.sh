#!/bin/bash
# Uncensor the n=6 (and extend n=5) capacity knee: large-B Grover-n runs on the
# extended distance-6 pool B1280 (max train 1216), markov forward, single GPU.
# The pool's first 704 pairs == the B768
# pool's, so these stitch onto the cached B<=704 sweep; held-out 64 unchanged.
#   n=6: B in {768,896,1024,1216}  (brackets the ~1000 knee, currently censored
#        at B=704 where <A_6>=0.919 still > 0.9)
#   n=5: B in {768,896,1024}       (extends the resolved-but-shallow tail)
# 4 per-seed shards share GPU0 (markov ~20MB each).
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet
export QA_PREFIX=sliding_puzzle_N120_K3_M8_B1280_D6
export MARKOV=1

for S in 1 2 3 4; do
(
  GPU=0 B_LIST="768 896 1024 1216" OMP_NUM_THREADS=2 ./run_grover_sweep.sh $S $S grover6 > ek_g6_s$S.log 2>&1
  GPU=0 B_LIST="768 896 1024"      OMP_NUM_THREADS=2 ./run_grover_sweep.sh $S $S grover5 > ek_g5_s$S.log 2>&1
  echo "[ek] seed $S done: $(date)"
) >> ek.log 2>&1 &
done
wait
echo "[ek] ALL DONE: $(date)" >> ek.log
