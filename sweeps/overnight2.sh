#!/bin/bash
# overnight2.sh — 2-day batch (armed 2026-07-03, harvest ~2026-07-06+).
# Three goals, in priority order:
#   A. N-scaling: random-regular N in {240,480} (reuse 120; stretch 960),
#      grover/std (GPU) + cbestkx/cstdx (CPU), B in {32,128}, seeds 1-8.
#      -> "the catch-up budget grows with problem size" panel.
#   B. Deep-n completion: rr N=120 grover-n2/3/4 (B 8/32/128) + sliding
#      grover3/4 full B-grid (8 seeds) + B=32 seeds 9-32.
#      -> held-out-vs-n curve robust across B and graph family.
#   C. Max-entropy classical controls (CPU): {centropy,cbentropy} x
#      ENTC {0.01,0.03,0.1}, B {8,32,128}, seeds 1-8.
#      -> "just add an entropy bonus" referee counter.
#
# Resource budget: one GPU, <=16 CPU threads total.
# GPU stages: N>=240 strictly ONE shard (OOM insurance: 10-16GB/proc at
# N=120 B>=96; scales with N); N=120 stages max 2 shards.
# run_grover_sweep.sh has per-run fail-forward (one OOM never kills a chain).
set -u
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet
export PYTORCH_ALLOC_CONF=expandable_segments:True
QAD=from4090/expr4/graph_qa

echo "[overnight2] armed $(date) — waiting for running sweeps to drain..."
while pgrep -f "quconet_rl_training_ar|conet_adam_training" > /dev/null; do sleep 60; done
echo "[overnight2] start: $(date)"

# ── Stage 0 (CPU): generate random-regular QA files for the N-scan ──
for N in 240 480 960; do
  for S in 1 2 3 4 5 6 7 8; do
    OUT="$QAD/randreg_N${N}_K3_M8_B192_seed${S}.pt"
    [ -f "$OUT" ] && continue
    [ "$N" = "960" ] && [ "$S" -gt 4 ] && continue   # stretch: 4 instances
    python conet/generate_graph_qa.py -N $N -K 3 -M 8 -B 192 \
        --path_min 3 --path_max 6 --seed $S --output_dir $QAD \
      && mv $QAD/quconet_ar_demo_N${N}_K3_M8_B192_*_graph_qa.pt "$OUT" \
      || echo "!! QA gen failed N=$N seed=$S"
  done
done
echo "[overnight2] QA files ready: $(ls $QAD | grep -c randreg) randreg files"

# ── CPU chain (background; OMP 2 x <=3 procs alongside 1-thread GPU procs) ──
(
  # C1: exact-gradient classical controls for the N-scan (fast DP)
  for N in 240 480; do
    for OBJ in cbestkx cstdx; do
      GPU="" OMP_NUM_THREADS=2 QA_PREFIX=randreg_N${N}_K3_M8_B192 QA_DIR=$QAD \
        B_LIST="32 128" ./run_grover_sweep.sh 1 8 $OBJ \
        > o2_c1_${OBJ}_N${N}.log 2>&1
    done
  done
  echo "[overnight2] C1 (N-scan exact controls) done: $(date)"
  # C2: max-entropy controls, two parallel shards (centropy | cbentropy)
  (
    for E in 0.01 0.03 0.1; do
      GPU="" OMP_NUM_THREADS=2 ENTC=$E B_LIST="8 32 128" \
        ./run_grover_sweep.sh 1 8 centropy > o2_c2_centH${E}.log 2>&1
    done
  ) &
  (
    for E in 0.01 0.03 0.1; do
      GPU="" OMP_NUM_THREADS=2 ENTC=$E B_LIST="8 32 128" \
        ./run_grover_sweep.sh 1 8 cbentropy > o2_c2_cbentH${E}.log 2>&1
    done
  ) &
  wait
  echo "[overnight2] C2 (entropy controls) done: $(date)"
  # C3: stretch — exact controls at N=960
  GPU="" OMP_NUM_THREADS=2 QA_PREFIX=randreg_N960_K3_M8_B192 QA_DIR=$QAD \
    B_LIST="128" ./run_grover_sweep.sh 1 4 cbestkx > o2_c3_cbestkx_N960.log 2>&1
  echo "[overnight2] CPU chain ALL done: $(date)"
) &
CPU_PID=$!

# ── GPU chain (gpu0) ──
# G1/G2: N-scan quantum, N=240 then N=480 — ONE shard at a time (memory)
for N in 240 480; do
  for OBJ in grover standard; do
    OMP_NUM_THREADS=1 QA_PREFIX=randreg_N${N}_K3_M8_B192 QA_DIR=$QAD \
      B_LIST="32 128" ./run_grover_sweep.sh 1 8 $OBJ \
      > o2_g_${OBJ}_N${N}.log 2>&1
    echo "[overnight2] GPU ${OBJ} N=${N} done: $(date)"
  done
done

# G5: rr N=120 deep-n (N=120-sized: two shards OK)
OMP_NUM_THREADS=1 QA_PREFIX=randreg_N120_K3_M8_B192 QA_DIR=$QAD \
  B_LIST="8 32 128" ./run_grover_sweep.sh 1 8 grover2 > o2_g5_rr_g2.log 2>&1 &
P1=$!
OMP_NUM_THREADS=1 QA_PREFIX=randreg_N120_K3_M8_B192 QA_DIR=$QAD \
  B_LIST="8 32 128" ./run_grover_sweep.sh 1 8 grover3 > o2_g5_rr_g3.log 2>&1 &
P2=$!
wait $P1 $P2 || true
OMP_NUM_THREADS=1 QA_PREFIX=randreg_N120_K3_M8_B192 QA_DIR=$QAD \
  B_LIST="8 32 128" ./run_grover_sweep.sh 1 8 grover4 > o2_g5_rr_g4.log 2>&1
echo "[overnight2] G5 (rr deep-n) done: $(date)"

# G6: sliding deep-n — full B-grid at 8 seeds (B=32 already done 2026-07-02)
# + B=32 seeds 9-32 (2 shards)
OMP_NUM_THREADS=1 B_LIST="8 16 48 64 96 128" ./run_grover_sweep.sh 1 8 grover3 > o2_g6_sl_g3.log 2>&1 &
P1=$!
OMP_NUM_THREADS=1 B_LIST="8 16 48 64 96 128" ./run_grover_sweep.sh 1 8 grover4 > o2_g6_sl_g4.log 2>&1 &
P2=$!
wait $P1 $P2 || true
OMP_NUM_THREADS=1 B_LIST="32" ./run_grover_sweep.sh 9 32 grover3 > o2_g6_sl_g3_s932.log 2>&1 &
P1=$!
OMP_NUM_THREADS=1 B_LIST="32" ./run_grover_sweep.sh 9 32 grover4 > o2_g6_sl_g4_s932.log 2>&1 &
P2=$!
wait $P1 $P2 || true
echo "[overnight2] G6 (sliding deep-n grid) done: $(date)"

# G7: stretch — N=960 grover, B=128, 4 instances (single shard)
OMP_NUM_THREADS=1 QA_PREFIX=randreg_N960_K3_M8_B192 QA_DIR=$QAD \
  B_LIST="128" ./run_grover_sweep.sh 1 4 grover > o2_g7_grover_N960.log 2>&1
OMP_NUM_THREADS=1 QA_PREFIX=randreg_N960_K3_M8_B192 QA_DIR=$QAD \
  B_LIST="128" ./run_grover_sweep.sh 1 4 standard > o2_g7_std_N960.log 2>&1
echo "[overnight2] G7 (N=960 stretch) done: $(date)"

wait $CPU_PID 2>/dev/null || true
echo "[overnight2] ALL DONE: $(date)"
