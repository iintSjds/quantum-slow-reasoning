#!/bin/bash
# Transformer-coin stages 2+3 (single GPU).
#   Stage 2 (attractor scan): B=32, d=32 — grover n=1..4 + standard + bestk3
#                             on the eight distance-6 sliding pools.
#   Stage 3 (capacity vs MODEL SIZE): n=1, d_model in {8,16,32,64},
#                             B in {8,16,32,64,128,256}, pool seeds 1..4,
#                             + classical-objective controls at d=32.
# Run detached:  setsid nohup ./tcoin_sweep.sh > tcoin_sweep.out 2>&1 &
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet
export CUDA_VISIBLE_DEVICES=0
QA=archive/expr4/graph_qa
OUT=results_tcoin_sweep
LOG=logs_tcoin
mkdir -p $OUT $LOG
T=examples/transformer_coin_training.py
COMMON="--num-val 64 --epochs 300 --early-stop --eval-freq 10 --device cuda"

run() { local label=$1; shift
  python $T "$@" $COMMON --out-dir $OUT --label "$label" \
      > $LOG/$label.log 2>&1
}

pool() { echo $QA/sliding_puzzle_N120_K3_M8_B768_D6_seed$1.pt; }

stage2() { local par=$1                      # seeds of one parity
  for S in $(seq $((1 + par)) 2 8); do
    for n in 1 2 3 4; do
      run tc_g${n}_s${S}_B32_d32 -f $(pool $S) -B 32 \
          --loss-type grover --grover-n $n --seed $S
    done
    run tc_std_s${S}_B32_d32 -f $(pool $S) -B 32 --loss-type standard --seed $S
    run tc_bk3_s${S}_B32_d32 -f $(pool $S) -B 32 \
        --loss-type bestk --best-k 3 --seed $S
  done
}

stage3() { local par=$1
  for S in $(seq $((1 + par)) 2 4); do
    for d in 8 16 32 64; do
      for B in 8 16 32 64 128; do
        run tc_g1_s${S}_B${B}_d${d} -f $(pool $S) -B $B \
            --loss-type grover --grover-n 1 --seed $S --d-model $d
      done
    done
    for B in 8 128; do                       # B=32 controls come from stage 2
      run tc_std_s${S}_B${B}_d32 -f $(pool $S) -B $B \
          --loss-type standard --seed $S
      run tc_bk3_s${S}_B${B}_d32 -f $(pool $S) -B $B \
          --loss-type bestk --best-k 3 --seed $S
    done
  done
}

stage3big() {                                # B=256 single-shard (memory)
  for S in 1 2 3 4; do
    for d in 8 16 32 64; do
      run tc_g1_s${S}_B256_d${d} -f $(pool $S) -B 256 \
          --loss-type grover --grover-n 1 --seed $S --d-model $d --prune 1e-9
    done
  done
}

echo "[tc] stage2 start $(date)" >> $LOG/sweep.log
stage2 0 & stage2 1 & wait
echo "[tc] stage2 done, stage3 start $(date)" >> $LOG/sweep.log
stage3 0 & stage3 1 & wait
echo "[tc] stage3 small-B done, B=256 start $(date)" >> $LOG/sweep.log
stage3big
echo "[tc] ALL DONE $(date)" >> $LOG/sweep.log
