# Training quantum reasoning models to think slowly: RLVR with Grover amplification

Reproduction code for the paper (arXiv link to be added).
Xiansheng Cai, Xiuhao Deng, and Kun Chen.

A quantum walker that prepares a coherent superposition over complete
reasoning trajectories is trained with reinforcement learning on the
*post-amplification* success probability, i.e. the reward is the measured
verdict of the deployed circuit after n rounds of Grover amplification.
Training then steers each question's success amplitude toward the interior
attractor p*(n) = sin²[π/(2(2n+1))] instead of collapsing onto single paths,
and the held-out advantage grows with the intended inference budget.  All
experiments are exact classical simulations (statevector or position-Markov),
so every probability in the paper is enumerated, not sampled.

## Layout

```
quconet/     importable package: AR quantum walker, U(3) coin operators,
             transformer (neural) coin, sliding-puzzle graph, trainer
conet/       classical softmax walker (exact-gradient Adam training),
             graph/QA pool generators, exact path enumerators
examples/    training drivers: quconet_rl_training_ar.py (coin walker,
             --loss-type grover/bestk/standard), transformer_coin_training.py
sweeps/      the exact sweep launchers that produced the archived runs
scripts/     analysis + figure scripts; scripts/_sweep_out/ holds the
             distilled per-question probability caches (tracked)
figs/        the manuscript figures as tracked reference artifacts
from4090/    data archive mount (not tracked; see from4090/README.md)
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -r requirements.txt                     # torch, numpy, matplotlib, networkx, hydra-core, omegaconf
```

Everything runs on CPU; a GPU only accelerates training sweeps.

## Reproducing the reported numbers

Two tiers.  **Tier 1 (no download):** the distilled caches shipped in
`scripts/_sweep_out/` contain the exact per-question success probabilities of
every trained model, so the main quantitative claims re-derive offline.
**Tier 2 (archive):** checkpoint-level analyses need the raw archive in
`from4090/` (Zenodo DOI at submission; or set `QSR_ROOT`, see
`from4090/README.md`).

| artifact | script | needs |
|---|---|---|
| Fig. 1 (architecture + routing + target) | `scripts/plot_prl_fig1_routing_target.py` | archive (routing panel loads one trained walker) |
| Fig. 2 (loss, collapse, payoff) | `scripts/plot_prl_fig2_collapse_payoff.py` | caches |
| Fig. 3 (capacity knee) | `scripts/plot_capacity_knee.py` | caches |
| Table I ladder, blind schedule, knee errors, deep-n IPR | `scripts/blind_schedule_analysis.py` (`--no-ipr` for cache-only parts) | caches (IPR part: archive) |
| Table I controls envelope, untrained row, cross-budget matrix | `scripts/control_audit_analysis.py` | caches |
| SM attractor + deep-budget grids (S4, S6) | `scripts/grover_sweep_analysis.py` | archive (`--figs-only`: caches) |
| SM path-support / IPR figure (D2) | `scripts/grover_ipr_analysis.py` | caches (rebuild: archive) |
| SM flow portraits (D4) | `scripts/grover_network_plot.py` | archive |
| SM mapping schematic (Q3) | `scripts/plot_mapping_schematic.py` | nothing |
| SM neural-coin figure + table (Q5) | `scripts/plot_tcoin_prl_sm.py` | archive (`tcoin_merged/`) |
| SM size scan (S7) | `scripts/grover_sizescan_analysis.py` | caches |
| SM query-scaling figure (F3) | `scripts/amplification_scaling.py --cache --figs F3` | caches (full rebuild: archive) |
| SM circuit verification table | `scripts/circuit_verify.py` | archive (one checkpoint) |
| SM two-sided imported-target controls | `scripts/twosided_target_control.py`, then `scripts/twosided_coin_control.py` | archive |
| SM untrained-walker size/difficulty scaling | `scripts/untrained_scaling.py` | archive |
| SM query-count exchange rate | `scripts/grover_exchange_rate.py` | caches |
| capacity-knee cache rebuild (large-B / N-scan) | `scripts/frust_analysis.py`, `scripts/frust_nscan_analysis.py` | archive |

Each script prints the numbers it supports and/or writes figures to
`figs/` or `scripts/_sweep_out/`.  `scripts/twosided_coin_control.py` first
validates its forward pipeline against the archived runs (max deviation
~1e-7) before training its controls.

## Training from scratch

The archived sweeps were produced by the launchers in `sweeps/` (they record
every hyperparameter; protocol: Adam, lr 0.05, 200 epochs, torch seed 42,
checkpoint selection by the objective-aligned training metric every 10
epochs):

| launcher | produces |
|---|---|
| `sweeps/run_grover_sweep.sh <s0> <s1> <family>` | `grover_sweep/` families (grover-n, classical one-shot / best-of-k / capped / entropy, semi) |
| `sweeps/overnight2.sh` | deep-n grid, entropy controls, random-regular size scan (`grover_sweep_rr*`) |
| `sweeps/overnight3.sh` | best-of-k budget ladders (classical and semi), grover n=5,6 |
| `sweeps/frust_sweep.sh`, `sweeps/extend_knee_sweep.sh` | large-B capacity scan (`grover_sweep_bigB/`, `grover_sweep_ext/`) |
| `sweeps/nscan_gen.sh`, `sweeps/nscan_sweep.sh` | capacity-knee N-scan (`grover_sweep_nscan/`) |
| `sweeps/tcoin_sweep.sh` | transformer-coin stages (`tcoin_merged/` was merged from 32-seed re-runs of these) |

Single runs, e.g.:

```bash
# quantum coin walker, Grover-n=2 objective, exact position-Markov forward
python examples/quconet_rl_training_ar.py -f from4090/expr4/graph_qa/sliding_puzzle_N120_K3_M8_B192_D6_seed1.pt \
    -B 32 --num-val 64 --epochs 200 --lr 0.05 --loss-type grover --grover-n 2 --markov

# classical softmax walker, exact-gradient best-of-3
python conet/conet_adam_training.py -f from4090/expr4/graph_qa/sliding_puzzle_N120_K3_M8_B192_D6_seed1.pt \
    -B 32 --num-val 64 --epochs 200 --lr 0.05 --loss-type bestk --best-k 3

# neural (transformer) coin, Grover-n=1
python examples/transformer_coin_training.py -f from4090/expr4/graph_qa/sliding_puzzle_N120_K3_M8_B192_D6_seed1.pt \
    -B 32 --num-val 64 --epochs 200 --loss-type grover --grover-n 1
```

New pools: `conet/generate_graph_qa.py` (sliding puzzle),
`conet/gen_randreg_frust.py` (random-regular, fixed graph distance).

## Notes

- `scripts/_sweep_out/classical_bstar.json` (classical train accuracy vs B,
  Fig. 3 right panel) is distilled from direct `conet/conet_adam_training.py`
  runs on the 768-pair distance-6 pools at each B (two seeds); the raw logs
  ship with the archive.
- Held-out accuracies reported in the paper use the blind fixed-n schedule;
  the caches store raw per-question p, and the analysis scripts apply the
  schedule, so both conventions can be recomputed from the same caches.
- The sliding-puzzle graph (N=120, K=3) is deterministic; all randomness
  enters through the pool seed (QA split) and the training seed (fixed at 42).

## License

MIT (see `LICENSE`).
