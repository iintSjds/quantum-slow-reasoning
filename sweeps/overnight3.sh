#!/bin/bash
# overnight3 (2026-07-06): budget-scaling ladders for the utility/exchange-rate
# analysis. Everything sliding puzzle N=120, seeds 1-8, protocol of
# run_grover_sweep.sh (adam lr=0.05, 200ep + ES).
#   E1 (CPU): CoNet exact best-of-k ladder, k in {4,8,16,32,64} (k=2 archived)
#   E2 (GPU): same-architecture QuCoNet-AR best-of-k ladder ("semi" control),
#             k in {2,4,8,16,32,64}, B in {8,32,128}
#   E3 (GPU): grover n=5,6 (budgets 11,13) extending the quantum ladder
# House rules: gpu0 only, max 2 concurrent GPU shards, ~16 CPU cores total.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet

(
  for K in 4 8 16 32 64; do
    GPU="" OMP_NUM_THREADS=2 BK=$K ./run_grover_sweep.sh 1 8 cbestkx > o3_cbX${K}.log 2>&1
    echo "[overnight3] cbX k=$K done: $(date)"
  done
  echo "[overnight3] CPU cbX ladder ALL done: $(date)"
) >> overnight3.log 2>&1 &
CPU_PID=$!

for K in 2 4 8 16 32 64; do
  OMP_NUM_THREADS=1 QBK=$K B_LIST="8 32 128" ./run_grover_sweep.sh 1 4 qbestk > o3_qbk${K}_s14.log 2>&1 &
  P1=$!
  OMP_NUM_THREADS=1 QBK=$K B_LIST="8 32 128" ./run_grover_sweep.sh 5 8 qbestk > o3_qbk${K}_s58.log 2>&1 &
  P2=$!
  wait $P1 $P2 || true
  echo "[overnight3] qbk k=$K done: $(date)" >> overnight3.log
done
OMP_NUM_THREADS=1 B_LIST="8 32 128" ./run_grover_sweep.sh 1 8 grover5 > o3_g5.log 2>&1 &
P1=$!
OMP_NUM_THREADS=1 B_LIST="8 32 128" ./run_grover_sweep.sh 1 8 grover6 > o3_g6.log 2>&1 &
P2=$!
wait $P1 $P2 || true
echo "[overnight3] grover5/6 done: $(date)" >> overnight3.log
wait $CPU_PID || true
echo "[overnight3] ALL DONE: $(date)" >> overnight3.log
