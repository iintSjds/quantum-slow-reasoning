#!/bin/bash
# Route-2 Grover sweep for QuCoNet (inference-aware training).
#
# Trains QuCoNet-AR with the Grover-n<=1 objective (--loss-type grover --grover-n 1),
# which prevents policy collapse, over a B sweep WITH a held-out valid set.
# Later evaluate with one Grover (A_1) vs classical best-of-2:
#   python docs/discussion/scripts/route2_grover_experiment.py --stage analyze ...
# (or glob results/ by label with amplification_step1-style loaders).
#
# GPU: default = GPU 0; override per shard with
#      GPU=1 ./run_grover_sweep.sh ...  or force CPU with GPU="".
#      QuCoNet-AR is tiny (720 params) so several shards share one GPU fine;
#      the classical exact-gradient runs are CPU-bound either way.
#
# Usage (shard by seed for parallelism; ~32 cores here, so 4-6 shards):
#   conda activate conet
#   OMP_NUM_THREADS=6 nohup ./run_grover_sweep.sh 1  8  grover > g_grover_1-8.log  2>&1 &
#   OMP_NUM_THREADS=6 nohup ./run_grover_sweep.sh 9  16 grover > g_grover_9-16.log 2>&1 &
#   # matched one-shot baseline (same optimizer/lr) for the collapse reference:
#   OMP_NUM_THREADS=6 nohup ./run_grover_sweep.sh 1  8  standard > g_std_1-8.log   2>&1 &
#   # classical inference-aware control (CoNet trained for best-of-2) + matched std:
#   OMP_NUM_THREADS=6 nohup ./run_grover_sweep.sh 1  2  cbestk > g_cbestk_1-2.log  2>&1 &
#   OMP_NUM_THREADS=6 nohup ./run_grover_sweep.sh 1  2  cstd   > g_cstd_1-2.log    2>&1 &
#   # exact-gradient classical controls (fast, single shard covers all seeds):
#   OMP_NUM_THREADS=4 nohup ./run_grover_sweep.sh 1  8  cbestkx > g_cbestkx.log    2>&1 &
#   OMP_NUM_THREADS=4 nohup ./run_grover_sweep.sh 1  8  cstdx   > g_cstdx.log      2>&1 &
#   # grover-n=2 (shard like grover):
#   OMP_NUM_THREADS=6 nohup ./run_grover_sweep.sh 1  2  grover2 > g_grover2_1-2.log 2>&1 &
set -e

SEED_START=${1:?Usage: ./run_grover_sweep.sh <seed_start> <seed_end> [grover|standard|cbestk|cstd]}
SEED_END=${2:?Usage: ./run_grover_sweep.sh <seed_start> <seed_end> [grover|standard|cbestk|cstd]}
OBJ=${3:-grover}   # quconet: grover (n=1) | standard.  classical: cbestk (best-of-2) | cstd

export CUDA_VISIBLE_DEVICES=${GPU-0}      # default GPU 0; GPU="" for CPU.
QA_DIR=${QA_DIR:-archive/expr4/graph_qa}
QA_PREFIX=${QA_PREFIX:-sliding_puzzle_N120_K3_M8_B192_D6}   # e.g. randreg_N120_K3_M8_B192
EPOCHS=200
LR=0.05
NUM_VAL=64
EVAL=10
ES="--early-stop --es-tol 1e-4 --es-window 5"
MARKOV_ARG=${MARKOV:+--markov}             # MARKOV=1 -> cheap which-path-exact forward (grover only)
B_LIST=${B_LIST:-"8 16 32 48 64 96 128"}  # env-overridable (e.g. B_LIST="32" for n-scans)

# Classical runs use conet_adam_training.py with the SAME adam/lr=0.05/ES
# protocol (num_rollouts=64000 default, matching the archived expr4 conet_adam
# family; no grad clip, also matching it -- quconet needs clip 1.0, classical
# never activated it).  cbestk = inference-aware classical control:
# best-of-2 objective via per-QA REINFORCE reweight k(1-p_hat)^(k-1).
case $OBJ in
    grover)   SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type grover --grover-n 1 --grad-clip 1.0"; TAG="grover_n1" ;;
    standard) SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type standard --grad-clip 1.0";            TAG="std"       ;;
    cbestk)   SCRIPT="conet/conet_adam_training.py"
              LOSS_ARGS="--loss-type bestk --best-k 2 --ckpt-freq 0";      TAG="cbestk2"   ;;
    cstd)     SCRIPT="conet/conet_adam_training.py"
              LOSS_ARGS="--loss-type standard --ckpt-freq 0";              TAG="cstd_adam" ;;
    # exact-gradient classical controls (differentiable DP for p -- no
    # REINFORCE dead-pair blindness; isolates pure objective shape).
    # These are FAST (~seconds/run): one shard for all 8 seeds suffices.
    # BK env overrides k for the budget-scaling ladder (default 2, the
    # original control; TAG stays cbestkX2 for k=2).
    cbestkx)  SCRIPT="conet/conet_adam_training.py"
              LOSS_ARGS="--loss-type bestk_exact --best-k ${BK:-2} --ckpt-freq 0"
              TAG="cbestkX${BK:-2}" ;;
    cstdx)    SCRIPT="conet/conet_adam_training.py"
              LOSS_ARGS="--loss-type bestk_exact --best-k 1 --ckpt-freq 0"; TAG="cstdX1"   ;;
    # grover-n=2..6 (attractors p* = 0.095/0.049/0.030/0.020/0.015)
    grover2)  SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type grover --grover-n 2 --grad-clip 1.0";  TAG="grover_n2" ;;
    grover3)  SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type grover --grover-n 3 --grad-clip 1.0";  TAG="grover_n3" ;;
    grover4)  SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type grover --grover-n 4 --grad-clip 1.0";  TAG="grover_n4" ;;
    grover5)  SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type grover --grover-n 5 --grad-clip 1.0";  TAG="grover_n5" ;;
    grover6)  SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type grover --grover-n 6 --grad-clip 1.0";  TAG="grover_n6" ;;
    # same-architecture classical control: QuCoNet-AR trained on the exact
    # best-of-k objective (the "semi-QuCoNet" ladder; QBK env sets k)
    qbestk)   SCRIPT="examples/quconet_rl_training_ar.py"
              LOSS_ARGS="--loss-type bestk --best-k ${QBK:-2} --grad-clip 1.0"
              TAG="qbkX${QBK:-2}" ;;
    # max-entropy controls ("just add an entropy bonus" referee counter);
    # coef via ENTC env (default 0.03): centropy = one-shot + H,
    # cbentropy = best-of-2 + H (strongest classical combo).
    centropy)  SCRIPT="conet/conet_adam_training.py"
               LOSS_ARGS="--loss-type standard --entropy-coef ${ENTC:-0.03} --ckpt-freq 0"
               TAG="centH${ENTC:-0.03}" ;;
    # capped exact-gradient control ("don't train above the cap"): an
    # imported confidence target, the classical analogue of p*(n).
    ccap)      SCRIPT="conet/conet_adam_training.py"
               LOSS_ARGS="--loss-type capped_exact --cap ${CAP:-0.25} --ckpt-freq 0"
               TAG="ccapX${CAP:-0.25}" ;;
    cbentropy) SCRIPT="conet/conet_adam_training.py"
               LOSS_ARGS="--loss-type bestk --best-k 2 --entropy-coef ${ENTC:-0.03} --ckpt-freq 0"
               TAG="cbentH${ENTC:-0.03}" ;;
    *) echo "Unknown objective: $OBJ (use grover|grover2..6|standard|qbestk|cbestk|cstd|cbestkx|cstdx|centropy|cbentropy|ccap)"; exit 1 ;;
esac

TOTAL_B=$(echo $B_LIST | wc -w)
TOTAL=$(( (SEED_END - SEED_START + 1) * TOTAL_B ))
echo "=== ${TAG}: seeds ${SEED_START}-${SEED_END}, lr=${LR} (adam,CPU), ${TOTAL} runs ==="
echo "Started: $(date)"; echo ""

COUNT=0
for SEED in $(seq $SEED_START $SEED_END); do
    QA="${QA_DIR}/${QA_PREFIX}_seed${SEED}.pt"
    if [ ! -f "$QA" ]; then echo "SKIP: $QA not found"; continue; fi
    for B in $B_LIST; do
        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}] seed=${SEED} B=${B} ${TAG} ($(date +%H:%M:%S))"
        python $SCRIPT \
            -f "$QA" -B $B --num-val $NUM_VAL \
            --epochs $EPOCHS --lr $LR --optimizer adam \
            $LOSS_ARGS $MARKOV_ARG \
            --eval-freq $EVAL --seed 42 --label "${TAG}_s${SEED}_B${B}" $ES \
            || echo "  !! FAILED (continuing): ${TAG} seed=${SEED} B=${B}"
    done
done
echo ""; echo "=== ${TAG} seeds ${SEED_START}-${SEED_END} done: $(date) ==="
