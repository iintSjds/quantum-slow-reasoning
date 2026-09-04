#!/bin/bash
# Robustness-audit training runs (CPU-friendly; ~35 min per run):
#   seedaudit -- optimizer-seed cross-audit: {grover n=1, grover n=4,
#                best-of-4} x question pools 1-4 x torch seeds {7,123,2026}
#                (seed 42 is the archived mainline), 36 runs;
#   relabel   -- node-relabeling invariance: grover n=1 on pools 1-8 x 2
#                uniformly random node permutations (seed 42), 16 runs.
# Analysis: scripts/robustness_audits.py (compute mode reads results/).
# Usage, from the repository root with the environment active:
#   ./sweeps/robustness_audits.sh {seedaudit|relabel}
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
QA_DIR=${QA_DIR:-from4090/expr4/graph_qa}
QA_RELABEL=${QA_RELABEL:-from4090/expr4/graph_qa_relabel}
PREFIX=sliding_puzzle_N120_K3_M8_B192_D6
COMMON="-B 32 --num-val 64 --epochs 200 --lr 0.05 --optimizer adam --grad-clip 1.0 --eval-freq 10 --early-stop --es-tol 1e-4 --es-window 5"
mkdir -p logs

run_one () {  # run_one <qa-file> <loss-args> <label> <torch-seed> <log>
  python examples/quconet_rl_training_ar.py -f "$1" $COMMON $2 \
    --seed "$4" --label "$3" >> "logs/$5" 2>&1 \
    || echo "!! FAILED $3" >> "logs/$5"
}

case ${1:?job} in
  seedaudit)
    for TS in 7 123 2026; do
      (
        for POOL in 1 2 3 4; do
          QA=$QA_DIR/${PREFIX}_seed${POOL}.pt
          run_one "$QA" "--loss-type grover --grover-n 1" "audit_g1_s${POOL}_t${TS}"  "$TS" "seedaudit_t${TS}.log"
          run_one "$QA" "--loss-type grover --grover-n 4" "audit_g4_s${POOL}_t${TS}"  "$TS" "seedaudit_t${TS}.log"
          run_one "$QA" "--loss-type bestk --best-k 4"    "audit_qbk4_s${POOL}_t${TS}" "$TS" "seedaudit_t${TS}.log"
        done
      ) &
    done
    wait
    ;;
  relabel)
    python scripts/relabel_qa.py --qa-dir "$QA_DIR" \
      --out "$QA_RELABEL" --seeds 1-8 --perms 2 >> logs/relabel.log 2>&1
    for SHARD in "1 2" "3 4" "5 6" "7 8"; do
      (
        for POOL in $SHARD; do
          for PERM in 1 2; do
            QA=$QA_RELABEL/${PREFIX}_seed${POOL}_perm${PERM}.pt
            run_one "$QA" "--loss-type grover --grover-n 1" \
              "relabel_g1_s${POOL}_p${PERM}" 42 "relabel_s${POOL}.log"
          done
        done
      ) &
    done
    wait
    ;;
esac
