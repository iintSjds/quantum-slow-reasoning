#!/usr/bin/env python3
"""
QuCoNet AR RL Training with IPR tracking and validation.

Trains a QuantumCoNetAR model using reinforcement learning, with:
- Batch-averaged IPR (path diversity) tracking during training
- Average success path length tracking
- Train/validation split for generalization measurement

Usage:
    # With pre-saved graph (first B train, last num_val valid)
    python quconet_rl_training_ar.py -f graph_qa.pt -B 64 --num-val 64

    # Create new graph on the fly
    python quconet_rl_training_ar.py -N 80 -K 3 -B 16 --num-val 16 --epochs 80
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quconet import QuCoNetTrainer, create_default_config
from quconet.graph import (
    create_regular_graph, graph_to_adjacency_list, analyze_graph_properties,
    find_qa_pairs_with_distance, save_graph_with_qa_pairs, load_graph_with_qa_pairs
)
import networkx as nx


# ─── IPR computation ──────────────────────────────────────────────────

def _generate_coin_state(N, K, Q, A, max_length):
    """Generate unique coin state for QA pair (matches QuantumCoNetAR)."""
    n = 0
    while K ** (n + 1) <= N:
        n += 1

    def to_k_base(x, base, digits):
        out = []
        for _ in range(digits):
            out.append(x % base)
            x //= base
        return out[::-1]

    q_d = to_k_base(Q, K, n + 1)
    a_d = to_k_base(A, K, n + 1)
    cs = []
    for i in range(n + 1):
        cs.append(q_d[i])
        cs.append(a_d[i])
    if max_length > len(cs):
        rep = []
        while len(rep) < max_length - len(cs):
            rep.extend(cs)
        cs.extend(rep[:max_length - len(cs)])
    elif max_length < len(cs):
        cs = cs[:max_length]
    return cs


def _extract_coin_matrices(model):
    """Build unitary coin matrices from live model parameters."""
    with torch.no_grad():
        params = model.coin.hamiltonian_params.cpu()
        idx = model.coin.triu_indices.cpu()
        N, K = model.N, model.K
        coins = torch.zeros(N, K, K, dtype=torch.complex64)
        for node in range(N):
            H = torch.zeros(K, K)
            H[idx[0], idx[1]] = params[node]
            H[idx[1], idx[0]] = params[node]
            coins[node] = torch.matrix_exp(1j * H)
    return coins


def _compute_qa_ipr(coin_probs, snm, N, K, M, Q, A, threshold=1e-15):
    """Exact path enumeration -> (IPR, total_success_prob) for one QA pair."""
    cs = _generate_coin_state(N, K, Q, A, M)
    path_probs = Counter()
    stack = [(Q, 0, (Q,), 1.0)]
    while stack:
        nd, st, path, cp = stack.pop()
        if nd == A:
            path_probs[path] += cp
            continue
        if st >= M:
            continue
        ci = cs[st]
        for k in range(K):
            p = coin_probs[nd, ci, k]
            new_p = cp * p
            if new_p < threshold:
                continue
            nbr = int(snm[nd, k])
            stack.append((nbr, st + 1, path + (nbr,), new_p))
    total = sum(path_probs.values())
    if total < 1e-15:
        return 0.0, total
    raw = np.array(list(path_probs.values()))
    norm = raw / total
    return 1.0 / np.sum(norm ** 2), total


def compute_batch_ipr(model, qa_pairs, M):
    """Accuracy-weighted mean IPR over a batch of QA pairs.

    Returns (weighted_ipr, mean_acc).
    """
    coins = _extract_coin_matrices(model)
    coin_probs = (coins.abs() ** 2).numpy()
    snm = model.shift.shift_node_map.cpu().numpy()
    N, K = model.N, model.K
    iprs, accs = [], []
    for Q, A in qa_pairs:
        ipr, acc = _compute_qa_ipr(coin_probs, snm, N, K, M, Q, A)
        iprs.append(ipr)
        accs.append(acc)
    iprs, accs = np.array(iprs), np.array(accs)
    s = accs.sum()
    return (accs * iprs).sum() / s if s > 0 else 0.0, accs.mean()


# ─── Plotting ─────────────────────────────────────────────────────────

def plot_training_curves(history, filename, B, num_val):
    """3-panel plot: accuracy, path length, IPR."""
    has_valid = num_val > 0
    has_ipr = any('train_ipr' in h for h in history)
    n_panels = 2 + int(has_ipr)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True)
    epochs = [h['epoch'] for h in history]

    # Panel 1: Accuracy
    ax = axes[0]
    ax.plot(epochs, [h['train_sr'] for h in history], 'b-', lw=1.5, label='Train')
    if has_valid:
        ev = [(h['epoch'], h['valid_sr']) for h in history if 'valid_sr' in h]
        if ev:
            ax.plot(*zip(*ev), 'r--o', lw=1.5, ms=3, label='Valid')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'QuCoNet AR Training  (B_train={B}, B_valid={num_val})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: Avg steps
    ax = axes[1]

    def _finite(lst):
        return [(e, v) for e, v in lst if v is not None and np.isfinite(v)]

    tr = _finite([(h['epoch'], h.get('train_steps')) for h in history])
    if tr:
        ax.plot(*zip(*tr), 'b-', lw=1.5, label='Train')
    if has_valid:
        va = _finite([(h['epoch'], h.get('valid_steps')) for h in history if 'valid_steps' in h])
        if va:
            ax.plot(*zip(*va), 'r--o', lw=1.5, ms=3, label='Valid')
    ax.set_ylabel('Avg Steps to Success')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: IPR
    if has_ipr:
        ax = axes[2]
        ev = [(h['epoch'], h['train_ipr']) for h in history if 'train_ipr' in h]
        ax.plot(*zip(*ev), 'b-o', lw=1.5, ms=3, label='Train $\\langle$IPR$\\rangle_w$')
        if has_valid:
            vev = [(h['epoch'], h['valid_ipr']) for h in history if 'valid_ipr' in h]
            if vev:
                ax.plot(*zip(*vev), 'r--s', lw=1.5, ms=3, label='Valid $\\langle$IPR$\\rangle_w$')
        ax.set_ylabel('$\\langle$IPR$\\rangle_w$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuCoNet AR RL Training")
    parser.add_argument("--fname", "-f", default=None, help="Input graph_qa.pt file")
    parser.add_argument("-B", type=int, default=None, help="Number of training QA pairs")
    parser.add_argument("-N", type=int, default=80)
    parser.add_argument("-K", type=int, default=3)
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=0,
                        help="Number of validation QA pairs (taken from end of QA list)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--eval-freq", type=int, default=5,
                        help="Evaluation & IPR frequency in epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Directory to save per-epoch checkpoints")
    parser.add_argument("--ckpt-freq", type=int, default=5,
                        help="Checkpoint save frequency in epochs")
    parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam',
                        help="Optimizer (default: adam)")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Gradient clipping max norm (default: None)")
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping when SR and IPR converge")
    parser.add_argument("--es-tol", type=float, default=1e-4,
                        help="Early stopping tolerance (max-min over window)")
    parser.add_argument("--loss-type", choices=['standard', 'log_prob', 'grover', 'bestk'],
                        default='standard',
                        help="Training objective: standard/log_prob = one-shot success; "
                             "grover = amplitude-amplification-n accuracy (route 2); "
                             "bestk = classical best-of-k on the same architecture "
                             "(exact-gradient semi-QuCoNet control)")
    parser.add_argument("--grover-n", type=int, default=1,
                        help="Grover iterations for --loss-type grover (0 recovers standard)")
    parser.add_argument("--best-k", type=int, default=2,
                        help="k for --loss-type bestk (1 recovers standard)")
    parser.add_argument("--markov", action="store_true",
                        help="compute the grover loss via the which-path-exact "
                             "position-Markov DP (O(B*N*K*M) memory, no N*K^M "
                             "state); loss-identical to the AR forward, enables "
                             "large B / large N. Only affects --loss-type grover.")
    parser.add_argument("--es-window", type=int, default=5,
                        help="Number of eval points to check for convergence")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load or create graph ──────────────────────────────────────────
    if args.fname:
        print(f"Loading graph from {args.fname}")
        G, all_qa = load_graph_with_qa_pairs(args.fname)
        N = G.number_of_nodes()
        K = G.degree(0)
        M = args.M
        if args.B is None:
            args.B = len(all_qa) - args.num_val
    else:
        N, K, M = args.N, args.K, args.M
        total_needed = (args.B or 16) + args.num_val
        print(f"Creating N={N} K={K} graph with {total_needed} QA pairs")
        G = create_regular_graph(N, K, "random")
        all_qa = find_qa_pairs_with_distance(
            G, batch_size=total_needed, min_distance=4, max_distance=6)
        if args.B is None:
            args.B = total_needed - args.num_val

    B = args.B
    num_val = args.num_val

    # ── Split train / valid ───────────────────────────────────────────
    if B + num_val > len(all_qa):
        print(f"ERROR: B={B} + num_val={num_val} = {B + num_val} "
              f"exceeds available QA pairs ({len(all_qa)})")
        sys.exit(1)

    train_qa = list(all_qa[:B])
    valid_qa = list(all_qa[-num_val:]) if num_val > 0 else []

    # Sanity: check no overlap
    train_set = set(map(tuple, train_qa))
    valid_set = set(map(tuple, valid_qa))
    overlap = train_set & valid_set
    if overlap:
        print(f"WARNING: {len(overlap)} QA pairs overlap between train and valid")

    # Extract graph stem for experiment naming
    graph_stem = os.path.splitext(os.path.basename(args.fname))[0] if args.fname else f"gen_N{N}_K{K}"

    props = analyze_graph_properties(G)
    adj_list = graph_to_adjacency_list(G)
    print(f"Graph: N={N} K={K} M={M}, diameter={props['diameter']}")
    print(f"QA split: {B} train + {num_val} valid (of {len(all_qa)} total)")

    # ── Setup trainer ─────────────────────────────────────────────────
    cfg = create_default_config(
        model={"N": N, "K": K, "max_steps": M, "use_ar": True,
               "loss_type": args.loss_type, "init_scale": 0.1},
        training={"num_epochs": args.epochs, "learning_rate": args.lr,
                  "batch_size": B, "eval_frequency": 1,
                  "save_checkpoints": True,
                  "gradient_clip": args.grad_clip},
        data={"num_qa_pairs": B, "max_distance": 6},
        experiment_name=f"quconet_{args.optimizer}_{graph_stem}_s{args.seed}"
    )
    trainer = QuCoNetTrainer(cfg)
    trainer.adjacency_list = adj_list
    trainer._setup_model()
    trainer.model.grover_n = args.grover_n          # used only when loss_type == "grover"
    trainer.model.best_k = args.best_k              # used only when loss_type == "bestk"
    trainer.model.markov_grover = args.markov       # cheap which-path-exact forward
    if args.markov:
        print("Forward: position-Markov DP (which-path-exact, memory-light)")
    if args.loss_type == "grover":
        print(f"Objective: grover-n (amplitude amplification), n={args.grover_n}")
    elif args.loss_type == "bestk":
        print(f"Objective: best-of-k (classical, same architecture), k={args.best_k}")
    trainer._setup_optimizer()
    # Override optimizer if SGD requested
    if args.optimizer == 'sgd':
        trainer.optimizer = torch.optim.SGD(
            trainer.model.parameters(), lr=args.lr)
    trainer._setup_model = lambda: None  # prevent re-init

    model = trainer.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")
    print(f"Training: {args.epochs} epochs, lr={args.lr} ({args.optimizer.upper()}), "
          f"eval every {args.eval_freq} epochs\n")

    # ── Epoch-0 evaluation (untrained model) ─────────────────────────
    history = []
    best_sr = 0.0
    best_state = None
    best_epoch = 0
    t0 = time.time()

    # Evaluate before any training so all curves start at epoch 0
    entry0 = {'epoch': 0, 'train_sr': 0.0, 'train_loss': None, 'train_steps': None}
    with torch.no_grad():
        # Train set eval
        tr_met = trainer.evaluate(train_qa)
        entry0['train_sr'] = float(tr_met['success_rate'])
        if not args.markov:          # markov: no N*K^M joint state (steps=nan anyway)
            starts = [q for q, a in train_qa]
            ends = [a for q, a in train_qa]
            psi = model.generate_batched_initial_states(
                starts, ends, initial_coin_state="unique")
            _, fwd = model(psi, ends, return_metrics=True)
            ts = fwd.get('avg_steps_to_success', float('inf'))
            ts = ts.item() if hasattr(ts, 'item') else float(ts)
            entry0['train_steps'] = ts if math.isfinite(ts) else None

        # Valid set eval
        if valid_qa:
            v_met = trainer.evaluate(valid_qa)
            entry0['valid_sr'] = float(v_met['success_rate'])
            if not args.markov:
                starts_v = [q for q, a in valid_qa]
                ends_v = [a for q, a in valid_qa]
                psi_v = model.generate_batched_initial_states(
                    starts_v, ends_v, initial_coin_state="unique")
                _, fwd_v = model(psi_v, ends_v, return_metrics=True)
                vs = fwd_v.get('avg_steps_to_success', float('inf'))
                vs = vs.item() if hasattr(vs, 'item') else float(vs)
                entry0['valid_steps'] = vs if math.isfinite(vs) else None

    # IPR at epoch 0
    t_ipr, _ = compute_batch_ipr(model, train_qa, M)
    entry0['train_ipr'] = float(t_ipr)
    if valid_qa:
        v_ipr, _ = compute_batch_ipr(model, valid_qa, M)
        entry0['valid_ipr'] = float(v_ipr)

    parts = [f"train_sr={entry0['train_sr']:.4f}"]
    if valid_qa:
        parts.append(f"valid_sr={entry0['valid_sr']:.4f}")
    if entry0['train_steps'] is not None:
        parts.append(f"steps={entry0['train_steps']:.1f}")
    parts.append(f"IPR={t_ipr:.3f}")
    if valid_qa:
        parts.append(f"v_IPR={v_ipr:.3f}")
    print(f"  Epoch   0: {', '.join(parts)}  (untrained)")

    history.append(entry0)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        trainer.current_epoch = epoch
        ep_metrics = trainer.train_epoch(train_qa)

        # train_epoch returns the dict; we add epoch and record it
        ep_metrics['epoch'] = epoch
        trainer.training_history.append(ep_metrics)

        train_sr = ep_metrics.get('average_success_rate',
                                  ep_metrics.get('success_rate', 0.0))
        train_loss = ep_metrics['loss']
        train_steps = ep_metrics.get('avg_steps_to_success', None)
        if train_steps is not None and hasattr(train_steps, 'item'):
            train_steps = train_steps.item()

        entry = {
            'epoch': epoch,
            'train_sr': float(train_sr),
            'train_loss': float(train_loss),
            'train_steps': float(train_steps) if train_steps is not None else None,
        }

        # Track best model
        if train_sr > best_sr:
            best_sr = train_sr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        # Periodic detailed evaluation
        is_eval = (epoch % args.eval_freq == 0) or epoch == args.epochs

        if is_eval:
            # Validation set evaluation
            if valid_qa:
                val_metrics = trainer.evaluate(valid_qa)
                entry['valid_sr'] = float(val_metrics['success_rate'])
                # avg_steps from a fresh forward pass on valid set
                if not args.markov:      # markov: no N*K^M joint state
                    with torch.no_grad():
                        starts = [q for q, a in valid_qa]
                        ends = [a for q, a in valid_qa]
                        psi = model.generate_batched_initial_states(
                            starts, ends, initial_coin_state="unique")
                        _, fwd_m = model(psi, ends, return_metrics=True)
                        vs = fwd_m.get('avg_steps_to_success', float('inf'))
                        vs = vs.item() if hasattr(vs, 'item') else float(vs)
                        entry['valid_steps'] = vs if math.isfinite(vs) else None

            # IPR (exact path enumeration)
            t_ipr, _ = compute_batch_ipr(model, train_qa, M)
            entry['train_ipr'] = float(t_ipr)
            if valid_qa:
                v_ipr, _ = compute_batch_ipr(model, valid_qa, M)
                entry['valid_ipr'] = float(v_ipr)

            # Print summary
            parts = [f"train_sr={train_sr:.4f}"]
            if valid_qa:
                parts.append(f"valid_sr={entry['valid_sr']:.4f}")
            if train_steps is not None and math.isfinite(train_steps):
                parts.append(f"steps={train_steps:.1f}")
            parts.append(f"IPR={t_ipr:.3f}")
            if valid_qa:
                parts.append(f"v_IPR={v_ipr:.3f}")
            print(f"  Epoch {epoch:3d}: {', '.join(parts)}")

            # Early stopping check
            if args.early_stop and 'train_ipr' in entry:
                past = [h for h in history if 'train_ipr' in h]
                recent = (past + [entry])[-args.es_window:]
                if len(recent) >= args.es_window:
                    sr_vals = [h['train_sr'] for h in recent]
                    ipr_vals = [h['train_ipr'] for h in recent]
                    sr_range = max(sr_vals) - min(sr_vals)
                    ipr_range = max(ipr_vals) - min(ipr_vals)
                    if sr_range < args.es_tol and ipr_range < args.es_tol:
                        print(f"  Early stop at epoch {epoch}: "
                              f"SR range={sr_range:.2e}, IPR range={ipr_range:.2e}")
                        history.append(entry)
                        break

        history.append(entry)

        # Save per-epoch checkpoint if requested
        if args.ckpt_dir and epoch % args.ckpt_freq == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                'config': {'N': N, 'K': K, 'M': M, 'B': B},
            }, ckpt_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  Best train SR={best_sr:.4f} at epoch {best_epoch}")

    # ── Save outputs ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Q0, A0 = train_qa[0]
    tag = f"{graph_stem}_quconet_{args.optimizer}_N{N}_K{K}_M{M}_B{B}_Q{Q0}_A{A0}_{timestamp}"
    if args.label:
        tag = f"{args.label}_{tag}"

    # Output directory (compatible with real_world_demo_ar.py layout)
    output_dir = str(trainer.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Best model checkpoint
    model_path = os.path.join(output_dir, f"{tag}_best_model.pt")
    torch.save({
        'model_state_dict': best_state,
        'config': {'N': N, 'K': K, 'M': M, 'B': B},
        'best_epoch': best_epoch,
        'best_sr': best_sr,
    }, model_path)
    print(f"Model: {model_path}")

    # Results JSON
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    results_path = os.path.join(output_dir, f"{tag}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'N': N, 'K': K, 'M': M, 'B': B, 'num_val': num_val,
                'lr': args.lr, 'epochs': args.epochs, 'seed': args.seed,
                'loss_type': args.loss_type, 'grover_n': args.grover_n,
                'best_k': args.best_k, 'markov': args.markov
            },
            'metrics': {
                'best_sr': best_sr, 'best_epoch': best_epoch,
                'elapsed_s': elapsed
            },
            'train_qa': train_qa,
            'valid_qa': valid_qa,
            'history': history,
        }, f, indent=2, default=_default)
    print(f"Results: {results_path}")

    # Training curves plot
    plot_path = os.path.join(output_dir, f"{tag}_curves.png")
    plot_training_curves(history, plot_path, B, num_val)
    print(f"Output dir: {output_dir}")

    return history


if __name__ == "__main__":
    main()
