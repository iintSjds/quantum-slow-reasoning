#!/usr/bin/env python3
"""Two-sided imported-target control: exact-gradient training of the softmax
walker on L = sum_i (p_i - c)^2, the symmetric analogue of the one-sided
capped loss -sum_i min(p_i, c).

Motivation: the one-sided cap fails at *import* (mass
drifts above the shelf, 44% above 0.3 at c=0.25) as well as at conversion,
so it overstates the paper's case.  A two-sided penalty holds the ridge
cleanly from both sides; it still cannot convert (best-of-3 at p=0.25 is
C3 = 0.578), and granting it the quantum read-out tests the clean
dissociation: target import is classical, conversion is the read-out's.

Protocol matched to the capped control family (grover_sweep archive):
Adam lr=0.05, 200 epochs, torch seed 42, softmax logits, eval every 10
epochs, selection by the objective-aligned training metric.  Question
pools seed 1..8 at B=32 on the D6/M8 sliding puzzle; pairs and graph are
taken verbatim from the archived run directories, so the comparison rows
(ccap25, grover) share the identical questions.

Run:  python scripts/twosided_target_control.py
"""
import os, sys, json, glob, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATAROOT = os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, "..")))
SWEEP = os.path.join(DATAROOT, "archive", "grover_sweep")
OUT = os.path.join(HERE, "_sweep_out")

CAPS = {"twosided25": 0.25, "twosided095": 0.0955,
        "twosided0495": 0.0495, "twosided0302": 0.0302}   # p*(1..4)
SEEDS = range(1, 9)
B, M, EPOCHS, LR, EVAL_EVERY, TORCH_SEED = 32, 8, 200, 0.05, 10, 42


def exact_success_probs(probs, neighbors, qa_pairs, M):
    """Absorbing DP, verbatim semantics of conet/conet_adam_training.py."""
    device = probs.device
    N, K = probs.shape
    nB = len(qa_pairs)
    starts = torch.tensor([q for q, _ in qa_pairs], dtype=torch.long, device=device)
    targets = torch.tensor([a for _, a in qa_pairs], dtype=torch.long, device=device)
    absorb = torch.ones(nB, N, device=device)
    absorb[torch.arange(nB), targets] = 0.0
    mass = torch.zeros(nB, N, device=device)
    mass[torch.arange(nB), starts] = 1.0
    p = mass[torch.arange(nB), targets].clone()
    mass = mass * absorb
    flat_nbr = neighbors.reshape(-1)
    for _ in range(M):
        flow = (mass.unsqueeze(2) * probs.unsqueeze(0)).reshape(nB, -1)
        new_mass = torch.zeros(nB, N, device=device)
        new_mass = new_mass.index_add(1, flat_nbr, flow)
        p = p + new_mass[torch.arange(nB), targets]
        mass = new_mass * absorb
    return p.clamp(0.0, 1.0)


def load_seed_assets(seed):
    """Graph neighbors + train/valid pairs from the archived ccap25 B=32 run."""
    pats = [f"{SWEEP}/conet_adam_sliding_puzzle_N120*_seed{seed}_B{B}_*/ccapX0.25_s{seed}_B{B}_*results.json",
            f"{SWEEP}/conet_adam_sliding_puzzle_N120*_seed{seed}_B{B}_*/cbestkX2_s{seed}_B{B}_*results.json"]
    rj = None
    for p in pats:
        hits = glob.glob(p)
        if hits:
            rj = hits[0]
            break
    assert rj, f"no archived run found for seed {seed}"
    d = json.load(open(rj))
    ck = glob.glob(rj.replace("_results.json", "_best_model.pt"))
    assert ck, f"no checkpoint next to {rj}"
    sd = torch.load(ck[0], map_location="cpu", weights_only=False)["model_state_dict"]
    nbr = sd["node_neighbors"].long()
    return nbr, [tuple(q) for q in d["train_qa"]], [tuple(q) for q in d["valid_qa"]]


def train_twosided(nbr, train_qa, c):
    torch.manual_seed(TORCH_SEED)
    logits = torch.zeros(nbr.shape[0], 3, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=LR)
    best = (float("inf"), None)
    for ep in range(1, EPOCHS + 1):
        probs = torch.softmax(logits, dim=-1)
        p = exact_success_probs(probs, nbr, train_qa, M)
        loss = ((p - c) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % EVAL_EVERY == 0 and loss.item() < best[0]:
            best = (loss.item(), logits.detach().clone())
    return best[1]


def eval_p(logits, nbr, qa):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        return exact_success_probs(probs, nbr, qa, M).numpy()


def blind(p, n):
    p = np.clip(p, 0.0, 1.0)
    return np.sin((2 * n + 1) * np.arcsin(np.sqrt(p))) ** 2


def bok(p, k):
    return 1.0 - (1.0 - np.clip(p, 0.0, 1.0)) ** k


def summarize(tag, tr_by_seed, va_by_seed, c):
    tr_all = np.concatenate(tr_by_seed)
    rows = {}
    rows["mean trained p"] = tr_all.mean()
    rows["interior (0,1)"] = float(np.mean((tr_all > 1e-6) & (tr_all < 1 - 1e-6)))
    rows[f"mass within c±0.05"] = float(np.mean(np.abs(tr_all - c) <= 0.05))
    rows["mass above c+0.05"] = float(np.mean(tr_all > c + 0.05))
    print(f"\n== {tag} (c={c}) ==")
    for k, v in rows.items():
        print(f"  {k:>22}: {v:.4f}")
    for split, by_seed in (("train", tr_by_seed), ("heldout", va_by_seed)):
        bo = [np.mean([bok(p, 2 * n + 1).mean() for p in by_seed]) for n in (1, 2, 3, 4)]
        bl = [np.mean([blind(p, n).mean() for p in by_seed]) for n in (1, 2, 3, 4)]
        print(f"  {split:>7} best-of-(2n+1): " + "  ".join(f"{v:.3f}" for v in bo))
        print(f"  {split:>7} +blind Grover-n: " + "  ".join(f"{v:.3f}" for v in bl))
    return rows


def cached_family(fam):
    cache = json.load(open(os.path.join(OUT, "p_cache.json")))
    tr, va = [], []
    for k, ps in cache.items():
        f, s, bb, split = k.split("|")
        if f == fam and bb == str(B):
            (tr if split == "train" else va).append(np.asarray(ps))
    return tr, va


def uniform_rows():
    u = json.load(open(os.path.join(OUT, "p_uniform.json")))
    tr = [np.asarray(u[f"uniform|{s}|{B}|train"]) for s in SEEDS if f"uniform|{s}|{B}|train" in u]
    va = [np.asarray(u[f"uniform|{s}|{B}|valid"]) for s in SEEDS if f"uniform|{s}|{B}|valid" in u]
    return tr, va


def main():
    results, sto = {}, {}
    for tag, c in CAPS.items():
        tr_by_seed, va_by_seed = [], []
        for seed in SEEDS:
            nbr, train_qa, valid_qa = load_seed_assets(seed)
            logits = train_twosided(nbr, train_qa, c)
            tr_by_seed.append(eval_p(logits, nbr, train_qa))
            va_by_seed.append(eval_p(logits, nbr, valid_qa))
            for split, arr in (("train", tr_by_seed[-1]), ("valid", va_by_seed[-1])):
                results[f"{tag}|{seed}|{B}|{split}"] = [float(x) for x in arr]
        summarize(tag, tr_by_seed, va_by_seed, c)
        sto[tag] = (tr_by_seed, va_by_seed)

    for fam in ("ccap25", "grover"):
        tr, va = cached_family(fam)
        if tr:
            summarize(f"[cached] {fam}", tr, va, 0.25)

    # matched-depth ladder: each system read out blind at its intended n
    natives = {1: "grover", 2: "grover2", 3: "grover3", 4: "grover4"}
    tags = {1: "twosided25", 2: "twosided095", 3: "twosided0495", 4: "twosided0302"}
    utr, uva = uniform_rows()
    print("\n== matched-depth blind ladder (n = 1..4) ==")
    print(f"{'system':<28}" + "".join(f"{'n='+str(n):>9}" for n in (1, 2, 3, 4)))
    for label, get in (
        ("native Grover-n  train", lambda n: np.mean([blind(p, n).mean() for p in cached_family(natives[n])[0]])),
        ("twosided p*(n)   train", lambda n: np.mean([blind(p, n).mean() for p in sto[tags[n]][0]])),
        ("untrained        train", lambda n: np.mean([blind(p, n).mean() for p in utr])),
        ("native Grover-n  heldout", lambda n: np.mean([blind(p, n).mean() for p in cached_family(natives[n])[1]])),
        ("twosided p*(n)   heldout", lambda n: np.mean([blind(p, n).mean() for p in sto[tags[n]][1]])),
        ("untrained        heldout", lambda n: np.mean([blind(p, n).mean() for p in uva])),
    ):
        print(f"{label:<28}" + "".join(f"{get(n):>9.3f}" for n in (1, 2, 3, 4)))
    for label, get in (
        ("native   heldout mean p", lambda n: np.mean([p.mean() for p in cached_family(natives[n])[1]])),
        ("twosided heldout mean p", lambda n: np.mean([p.mean() for p in sto[tags[n]][1]])),
        ("untrained heldout mean p", lambda n: np.mean([p.mean() for p in uva])),
    ):
        print(f"{label:<28}" + "".join(f"{get(n):>9.4f}" for n in (1, 2, 3, 4)))

    with open(os.path.join(OUT, "p_twosided.json"), "w") as f:
        json.dump(results, f)
    print(f"\nwrote {os.path.join(OUT, 'p_twosided.json')}")


if __name__ == "__main__":
    main()
