#!/usr/bin/env python3
"""Train the transformer coin (TransformerCoinAR) with exact gradients.

Same data conventions as quconet_rl_training_ar.py: the pool file's first B
QA pairs train, the last --num-val are held out. The forward is the exact
prefix-tree sweep (no sampling anywhere); losses are the same response
functions as the tabular model (standard / grover-n / best-of-k).

Usage:
    python transformer_coin_training.py -f pool.pt -B 8 --num-val 64 \
        --loss-type grover --grover-n 1 --epochs 200
"""
import os
import sys
import json
import time
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quconet.transformer_coin import TransformerCoinAR
from quconet.quconet_ar import QuantumCoNetAR
from quconet.graph import load_graph_with_qa_pairs, graph_to_adjacency_list


def interior_mass(p, lo=0.02, hi=0.98):
    p = np.asarray(p)
    return float(((p > lo) & (p < hi)).mean()) if len(p) else 0.0


def main():
    ap = argparse.ArgumentParser(description="Transformer-coin exact training")
    ap.add_argument("--fname", "-f", required=True, help="graph_qa .pt pool")
    ap.add_argument("-B", type=int, default=8)
    ap.add_argument("--num-val", type=int, default=64)
    ap.add_argument("-M", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss-type", choices=["standard", "grover", "bestk"],
                    default="grover")
    ap.add_argument("--grover-n", type=int, default=1)
    ap.add_argument("--best-k", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=64)
    ap.add_argument("--context", choices=["path", "node"], default="path")
    ap.add_argument("--prune", type=float, default=1e-12,
                    help="branch-weight prune during training (0 = exact; "
                         "1e-12 changes p by <1e-9, see tests)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-freq", type=int, default=10)
    ap.add_argument("--early-stop", action="store_true",
                    help="stop when the objective-aligned train metric "
                         "plateaus (max-min < es-tol over es-window evals)")
    ap.add_argument("--es-tol", type=float, default=1e-4)
    ap.add_argument("--es-window", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=60)
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results_tcoin")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device) if args.device else None

    G, all_qa = load_graph_with_qa_pairs(args.fname)
    N, K = G.number_of_nodes(), G.degree(0)
    adj = graph_to_adjacency_list(G)
    train_qa = [tuple(map(int, q)) for q in all_qa[:args.B]]
    valid_qa = [tuple(map(int, q)) for q in all_qa[-args.num_val:]] \
        if args.num_val > 0 else []

    model = TransformerCoinAR(N, K, args.M, adj, d_model=args.d_model,
                              n_layers=args.n_layers, n_heads=args.n_heads,
                              d_ff=args.d_ff, context=args.context, device=dev)
    n_params = sum(p.numel() for p in model.parameters())
    stem = os.path.splitext(os.path.basename(args.fname))[0]
    label = args.label or (f"tcoin_{args.loss_type}"
                           f"{args.grover_n if args.loss_type == 'grover' else ''}"
                           f"_d{args.d_model}_{args.context}_s{args.seed}_B{args.B}")
    print(f"pool={stem}  N={N} K={K} M={args.M}  "
          f"train={len(train_qa)} valid={len(valid_qa)}")
    print(f"model: d={args.d_model} L={args.n_layers} h={args.n_heads} "
          f"ctx={args.context}  params={n_params}  device={model.device}")
    print(f"objective: {args.loss_type}"
          + (f" n={args.grover_n}" if args.loss_type == "grover" else "")
          + (f" k={args.best_k}" if args.loss_type == "bestk" else ""))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)
    history, best = [], {"metric": -1.0}

    def evaluate(qa):
        if not qa:
            return None
        with torch.no_grad():
            p = model.success_probs(qa, prune=1e-12).cpu().numpy()
        A = QuantumCoNetAR.grover_acc(torch.tensor(p), args.grover_n).numpy()
        clk = 1.0 - (1.0 - p) ** args.best_k
        return dict(p=p.tolist(), mean_p=float(p.mean()),
                    mean_acc=float(A.mean()), mean_clk=float(clk.mean()),
                    interior=interior_mass(p))

    def aligned(tr):
        """Objective-aligned train metric (selection + early stopping)."""
        return {"grover": tr["mean_acc"], "standard": tr["mean_p"],
                "bestk": tr["mean_clk"]}[args.loss_type]

    t0 = time.time()
    for epoch in range(args.epochs + 1):
        if epoch > 0:
            model.train()
            opt.zero_grad()
            loss, p = model.loss(train_qa, loss_type=args.loss_type,
                                 grover_n=args.grover_n, best_k=args.best_k,
                                 prune=args.prune)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
            opt.step()

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            tr = evaluate(train_qa)
            va = evaluate(valid_qa)
            rec = {"epoch": epoch, "train": tr, "valid": va,
                   "wall": round(time.time() - t0, 1)}
            history.append(rec)
            print(f"ep {epoch:4d}  train p={tr['mean_p']:.4f} "
                  f"A{args.grover_n}={tr['mean_acc']:.4f} "
                  f"int={tr['interior']:.2f}"
                  + (f"  valid p={va['mean_p']:.4f} A={va['mean_acc']:.4f}"
                     if va else "")
                  + f"  [{rec['wall']}s]")
            # best-model selection on the objective-aligned train metric
            # (one-shot selection catches the pre-attractor transient for
            # grover runs; both metrics are recorded in the history)
            if aligned(tr) > best["metric"]:
                best = {"metric": aligned(tr), "epoch": epoch}
                torch.save({"state_dict": model.state_dict(),
                            "config": dict(N=N, K=K, M=args.M,
                                           d_model=args.d_model,
                                           n_layers=args.n_layers,
                                           n_heads=args.n_heads,
                                           d_ff=args.d_ff,
                                           context=args.context),
                            "args": vars(args), "epoch": epoch,
                            "pool": args.fname},
                           os.path.join(args.out_dir,
                                        f"{label}_best_model.pt"))
        with open(os.path.join(args.out_dir, f"{label}_history.json"),
                  "w") as fh:
            json.dump({"args": vars(args), "n_params": n_params,
                       "history": history, "best": best}, fh)

        if (args.early_stop and epoch >= args.min_epochs
                and len(history) >= args.es_window):
            win = [aligned(h["train"]) for h in history[-args.es_window:]]
            if max(win) - min(win) < args.es_tol:
                print(f"early stop at ep {epoch} "
                      f"(aligned metric plateau {max(win) - min(win):.2e})")
                break

    print(f"done in {time.time() - t0:.0f}s; best aligned train metric "
          f"{best['metric']:.4f} @ep{best['epoch']}")


if __name__ == "__main__":
    main()
