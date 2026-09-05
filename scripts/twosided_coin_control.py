#!/usr/bin/env python3
"""Two-sided imported-target control ON THE COIN ARCHITECTURE (semi-QuCoNet).

Attribution test: the softmax two-sided control's held-out
lift on the puzzle may come from the MODEL CLASS (one shared softmax row per
node) rather than the LOSS.  The clean control trains the coin architecture
itself -- the same U(3)-coin walker as the Grover runs, measured every step,
i.e. a classical Markov walker with the question-conditioned channel pattern --
on the identical two-sided loss L = sum_i (p_i - c)^2.  If its held-out ladder
matches the native Grover-trained walker (no lift), the softmax upset is an
architecture artifact; if it lifts like the softmax, the loss is doing it.

Protocol identical to the archived quconet_adam family: Adam lr=0.05,
200 epochs, torch seed 42, init_scale=0.1 (flat coin + uniform noise),
selection by the objective every 10 epochs; pools and shift map taken from
the archived run directories (validated against p_cache to float32).

Run from repo root:  python scripts/twosided_coin_control.py
"""
import os, sys, json, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATAROOT = os.environ.get("QSR_ROOT", REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from quconet.operators_ar import CoinOperatorAR              # noqa: E402
from quconet.quconet_ar import QuantumCoNetAR                # noqa: E402
from twosided_target_control import blind, cached_family     # noqa: E402

SWEEP = os.path.join(DATAROOT, "archive", "grover_sweep")
OUT = os.path.join(HERE, "_sweep_out")
CPU = torch.device("cpu")
CAPS = {"tscoin25": 0.25, "tscoin095": 0.0955,
        "tscoin0495": 0.0495, "tscoin0302": 0.0302}          # p*(1..4)
SEEDS = range(1, 9)
B, M, N, K, EPOCHS, LR, EVERY, TSEED = 32, 8, 120, 3, 200, 0.05, 10, 42


def seed_assets(seed):
    """Shift map from the archived grover_n1 checkpoint; QA from its results.json
    (falling back to the classical run's results.json for the pairs)."""
    qjs = glob.glob(f"{SWEEP}/quconet_adam_sliding_puzzle_N120*_seed{seed}_s42_N120_K3_M8_B{B}_*/"
                    f"grover_n1_s{seed}_B{B}_*results.json")
    assert qjs, f"no archived quantum run for seed {seed}"
    rj = qjs[0]
    ck = torch.load(rj.replace("_results.json", "_best_model.pt"),
                    map_location="cpu", weights_only=False)
    shift_map = ck["model_state_dict"]["shift.shift_node_map"].long()
    d = json.load(open(rj))
    if "train_qa" not in d:
        cjs = glob.glob(f"{SWEEP}/conet_adam_sliding_puzzle_N120*_seed{seed}_B{B}_*/"
                        f"ccapX0.25_s{seed}_B{B}_*results.json")
        d = json.load(open(cjs[0]))
    return shift_map, [tuple(q) for q in d["train_qa"]], [tuple(q) for q in d["valid_qa"]], \
        ck["model_state_dict"]["coin.hamiltonian_params"]


def channel_patterns(qa):
    return torch.tensor([QuantumCoNetAR.generate_unique_coin_state(N, K, int(q), int(a), max_length=M)
                         for q, a in qa], dtype=torch.long)


def coin_probs(coin):
    H = coin.build_hamiltonian_batch(1)          # (1,N,K,K)
    C = coin.build_coin_operators(H)[0]          # (N,K,K) complex
    return C.abs() ** 2                          # (N,Kin,Kout)


def markov_p(cp, shift_map, pats, qa):
    nB = len(qa)
    Q = torch.tensor([q for q, _ in qa]); A = torch.tensor([a for _, a in qa])
    ar = torch.arange(nB)
    idx = shift_map.reshape(-1)
    oneA = torch.zeros(nB, N); oneA[ar, A] = 1.0
    dist = torch.zeros(nB, N); dist[ar, Q] = 1.0
    succ = torch.zeros(nB)
    succ = succ + (dist * oneA).sum(1) * 0.0     # start==target excluded by pools
    for s in range(M):
        succ = succ + (dist * oneA).sum(1)
        dist = dist * (1.0 - oneA)
        Ts = cp[:, pats[:, s], :].permute(1, 0, 2)          # (B,N,Kout)
        contrib = (dist.unsqueeze(-1) * Ts).reshape(nB, N * K)
        dist = torch.zeros(nB, N).index_add(1, idx, contrib)
    return (succ + (dist * oneA).sum(1)).clamp(0.0, 1.0)


def validate():
    cache = json.load(open(os.path.join(OUT, "p_cache.json")))
    worst = 0.0
    for seed in SEEDS:
        shift_map, tq, vq, ham = seed_assets(seed)
        coin = CoinOperatorAR(N, K, device=CPU, init_scale=0.1)
        coin.hamiltonian_params.data = ham.to(CPU)
        cp = coin_probs(coin).detach()
        for split, qa in (("train", tq), ("valid", vq)):
            ref = np.array(cache[f"grover|{seed}|{B}|{split}"])
            mine = markov_p(cp, shift_map, channel_patterns(qa), qa).numpy()
            worst = max(worst, float(np.abs(mine - ref).max()))
    print(f"replica validation vs p_cache (8 seeds, both splits): max|diff| = {worst:.2e}")
    assert worst < 1e-5, "replica does not reproduce the archived runs"


def train(shift_map, tq, c):
    torch.manual_seed(TSEED)
    coin = CoinOperatorAR(N, K, device=CPU, init_scale=0.1)
    opt = torch.optim.Adam(coin.parameters(), lr=LR)
    pats = channel_patterns(tq)
    best = (float("inf"), None)
    for ep in range(1, EPOCHS + 1):
        p = markov_p(coin_probs(coin), shift_map, pats, tq)
        loss = ((p - c) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % EVERY == 0 and loss.item() < best[0]:
            best = (loss.item(), coin.hamiltonian_params.detach().clone())
    coin.hamiltonian_params.data = best[1]
    return coin


def main():
    validate()
    results, sto = {}, {}
    for tag, c in CAPS.items():
        trs, vas = [], []
        for seed in SEEDS:
            shift_map, tq, vq, _ = seed_assets(seed)
            coin = train(shift_map, tq, c)
            cp = coin_probs(coin).detach()
            trs.append(markov_p(cp, shift_map, channel_patterns(tq), tq).numpy())
            vas.append(markov_p(cp, shift_map, channel_patterns(vq), vq).numpy())
            for split, arr in (("train", trs[-1]), ("valid", vas[-1])):
                results[f"{tag}|{seed}|{B}|{split}"] = [float(x) for x in arr]
        sto[tag] = (trs, vas)
        print(f"{tag} (c={c}): train mean p {np.concatenate(trs).mean():.4f}, "
              f"within c±0.05 {np.mean(np.abs(np.concatenate(trs)-c)<=0.05):.3f}, "
              f"heldout mean p {np.concatenate(vas).mean():.4f}")

    ts_soft = json.load(open(os.path.join(OUT, "p_twosided.json")))
    soft_tags = {1: "twosided25", 2: "twosided095", 3: "twosided0495", 4: "twosided0302"}
    coin_tags = {1: "tscoin25", 2: "tscoin095", 3: "tscoin0495", 4: "tscoin0302"}
    natives = {1: "grover", 2: "grover2", 3: "grover3", 4: "grover4"}

    def soft(nn, split):
        return [np.asarray(ts_soft[f"{soft_tags[nn]}|{s}|{B}|{split}"]) for s in SEEDS]

    print("\n== matched-depth blind ladder, coin-architecture attribution test ==")
    print(f"{'system':<30}" + "".join(f"{'n='+str(nn):>9}" for nn in (1, 2, 3, 4)))
    for label, get in (
        ("native Grover-n   heldout", lambda nn: np.mean([blind(p, nn).mean() for p in cached_family(natives[nn])[1]])),
        ("twosided COIN     heldout", lambda nn: np.mean([blind(p, nn).mean() for p in sto[coin_tags[nn]][1]])),
        ("twosided softmax  heldout", lambda nn: np.mean([blind(p, nn).mean() for p in soft(nn, "valid")])),
        ("native Grover-n   train-blind", lambda nn: np.mean([blind(p, nn).mean() for p in cached_family(natives[nn])[0]])),
        ("twosided COIN     train-blind", lambda nn: np.mean([blind(p, nn).mean() for p in sto[coin_tags[nn]][0]])),
        ("native   heldout mean p", lambda nn: np.mean([p.mean() for p in cached_family(natives[nn])[1]])),
        ("ts-COIN  heldout mean p", lambda nn: np.mean([p.mean() for p in sto[coin_tags[nn]][1]])),
        ("ts-soft  heldout mean p", lambda nn: np.mean([p.mean() for p in soft(nn, "valid")])),
    ):
        print(f"{label:<30}" + "".join(f"{get(nn):>9.3f}" for nn in (1, 2, 3, 4)))

    with open(os.path.join(OUT, "p_twosided_coin.json"), "w") as f:
        json.dump(results, f)
    print(f"\nwrote {os.path.join(OUT, 'p_twosided_coin.json')}")


if __name__ == "__main__":
    main()
