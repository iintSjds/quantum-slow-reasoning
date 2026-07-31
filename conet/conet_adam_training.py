#!/usr/bin/env python3
"""
CoNet with Adam optimizer (standard GRPO-style training).

Same softmax routing model as CoNet, but trained with the same
algorithm as semi-QuCoNet (Adam variant):
  - Softmax logits parameterization (one distribution per node)
  - Classical rollouts, sample action from π_x = softmax(θ_x)
  - Reward: {0, 1}
  - Advantage: (R - per_QA_mean), no std normalization
  - Loss: -(advantage * log_prob).mean()
  - Optimizer: Adam

This isolates the parameterization effect: comparing CoNet-Adam vs
Semi-QuCoNet-Adam, both use identical training but differ in
softmax vs unitary coin routing.

Usage:
    python conet_adam_training.py -f graph_qa.pt -B 3 --num-val 64 --epochs 120
"""

import os, sys, json, math, time, argparse
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We only need graph utilities from quconet
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'quconet'))
from quconet.graph import (
    create_regular_graph, graph_to_adjacency_list, analyze_graph_properties,
    find_qa_pairs_with_distance, load_graph_with_qa_pairs
)


# ─── Softmax routing model ──────────────────────────────────────────

class SoftmaxRouter(nn.Module):
    """Softmax routing policy: one distribution per node, shared by all QA pairs."""

    def __init__(self, N, K, edge_index):
        super().__init__()
        self.N = N
        self.K = K
        # Logits: [N, K] — initialized near-uniform
        self.logits = nn.Parameter(torch.zeros(N, K))
        # Build neighbor map from edge_index: node_neighbors[x, k] = neighbor
        self.register_buffer('node_neighbors', self._build_neighbor_map(edge_index, N, K))

    def _build_neighbor_map(self, edge_index, N, K):
        """Build [N, K] neighbor lookup from edge_index [2, num_edges]."""
        neighbors = torch.full((N, K), -1, dtype=torch.long)
        count = torch.zeros(N, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])
            k = int(count[src])
            if k < K:
                neighbors[src, k] = dst
                count[src] += 1
        return neighbors

    def get_probs(self):
        """Return routing probabilities [N, K]."""
        return torch.softmax(self.logits, dim=-1)

    def get_probs_numpy(self):
        with torch.no_grad():
            return self.get_probs().cpu().numpy()


# ─── IPR computation ─────────────────────────────────────────────────

def _compute_qa_ipr(probs_np, neighbors_np, N, K, M, Q, A, threshold=1e-15):
    """Exact path enumeration for softmax routing (QA-independent)."""
    path_probs = Counter()
    stack = [(Q, 0, (Q,), 1.0)]
    while stack:
        nd, st, path, cp = stack.pop()
        if nd == A:
            path_probs[path] += cp
            continue
        if st >= M:
            continue
        for k in range(K):
            p = probs_np[nd, k]
            new_p = cp * p
            if new_p < threshold:
                continue
            nbr = int(neighbors_np[nd, k])
            stack.append((nbr, st + 1, path + (nbr,), new_p))
    total = sum(path_probs.values())
    if total < 1e-15:
        return 0.0, total
    raw = np.array(list(path_probs.values()))
    norm = raw / total
    return 1.0 / np.sum(norm ** 2), total


def compute_batch_ipr(model, qa_pairs, M):
    probs_np = model.get_probs_numpy()
    neighbors_np = model.node_neighbors.cpu().numpy()
    N, K = model.N, model.K
    iprs, accs = [], []
    for Q, A in qa_pairs:
        ipr, acc = _compute_qa_ipr(probs_np, neighbors_np, N, K, M, Q, A)
        iprs.append(ipr)
        accs.append(acc)
    iprs, accs = np.array(iprs), np.array(accs)
    s = accs.sum()
    return (accs * iprs).sum() / s if s > 0 else 0.0, accs.mean()


# ─── Classical rollout sampling ──────────────────────────────────────

def sample_rollouts(probs_np, neighbors_np, qa_pairs, M, num_rollouts):
    """Sample classical trajectories from softmax routing (QA-independent)."""
    N, K = probs_np.shape
    B = len(qa_pairs)
    total = B * num_rollouts

    starts = np.array([Q for Q, A in qa_pairs], dtype=np.int64)
    targets = np.array([A for Q, A in qa_pairs], dtype=np.int64)
    current = np.repeat(starts, num_rollouts)
    target = np.repeat(targets, num_rollouts)

    all_nodes, all_actions = [], []
    success = np.zeros(total, dtype=bool)
    path_lengths = np.full(total, M, dtype=np.int64)
    active = np.ones(total, dtype=bool)

    for step in range(M):
        # Routing is QA-independent: probs depend only on current node
        node_probs = probs_np[current]  # [total, K]

        cumprobs = np.cumsum(node_probs, axis=1)
        u = np.random.uniform(0, 1, size=total)
        actions = (cumprobs < u[:, None]).sum(axis=1).astype(np.int64)
        actions = np.clip(actions, 0, K - 1)
        actions[~active] = 0

        all_nodes.append(current.copy())
        all_actions.append(actions.copy())

        new_nodes = neighbors_np[current, actions]
        current = np.where(active, new_nodes, current)

        newly_reached = (current == target) & active & ~success
        success |= newly_reached
        path_lengths[newly_reached] = step + 1
        active &= ~success

    return all_nodes, all_actions, success, path_lengths


# ─── Exact-gradient training (differentiable absorbing DP) ───────────

def exact_success_probs(probs, neighbors, qa_pairs, M):
    """Differentiable per-QA success probability via absorbing DP.

    probs: [N, K] transition probs WITH grad; neighbors: [N, K] long.
    Walker semantics match sample_rollouts / the path enumerator exactly:
    absorb on first arrival at the target, at most M steps.
    """
    device = probs.device
    N, K = probs.shape
    B = len(qa_pairs)
    starts = torch.tensor([q for q, _ in qa_pairs], dtype=torch.long, device=device)
    targets = torch.tensor([a for _, a in qa_pairs], dtype=torch.long, device=device)

    absorb = torch.ones(B, N, device=device)
    absorb[torch.arange(B), targets] = 0.0        # constant mask, no grad needed

    mass = torch.zeros(B, N, device=device)
    mass[torch.arange(B), starts] = 1.0
    p = mass[torch.arange(B), targets].clone()    # start == target edge case
    mass = mass * absorb

    flat_nbr = neighbors.reshape(-1)              # [N*K]
    for _ in range(M):
        flow = (mass.unsqueeze(2) * probs.unsqueeze(0)).reshape(B, -1)  # [B, N*K]
        new_mass = torch.zeros(B, N, device=device)
        new_mass = new_mass.index_add(1, flat_nbr, flow)
        p = p + new_mass[torch.arange(B), targets]
        mass = new_mass * absorb
    return p.clamp(0.0, 1.0)


def exact_bestk_step(model, optimizer, qa_pairs, M, best_k, grad_clip=None):
    """One epoch of exact-gradient best-of-k training.

    Maximizes sum_QA [1 - (1-p)^k] with p from the differentiable DP -- no
    rollout sampling, so (unlike REINFORCE) pairs with p ~ 0 still receive
    gradient. best_k=1 recovers the exact-gradient one-shot objective.
    """
    probs = model.get_probs()                     # [N, K] with grad
    p = exact_success_probs(probs, model.node_neighbors, qa_pairs, M)
    loss = -(1.0 - (1.0 - p) ** best_k).sum()

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        train_sr = float(p.mean())
        train_bestk = float((1.0 - (1.0 - p) ** best_k).mean())
    return train_sr, float(loss.item()), float('inf'), train_bestk


def exact_capped_step(model, optimizer, qa_pairs, M, cap, grad_clip=None,
                      best_k=2):
    """One epoch of exact-gradient CAPPED one-shot training.

    Maximizes sum_QA min(p, cap): a question above the cap receives no
    gradient ("don't train what is already confident enough").  The
    clamped objective is still monotone (a shelf, not a peak -- no
    restoring force from above); the cap is a hand-imposed confidence
    target, the classical analogue of the amplification attractor.
    Control for whether an imported target closes the gap to grover-n.
    Reported train_bestk uses best_k for the aligned selection metric.
    """
    probs = model.get_probs()                     # [N, K] with grad
    p = exact_success_probs(probs, model.node_neighbors, qa_pairs, M)
    loss = -(torch.clamp(p, max=cap)).sum()

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        train_sr = float(p.mean())
        train_obj = float(torch.clamp(p, max=cap).mean() / cap)
    return train_sr, float(loss.item()), float('inf'), train_obj


# ─── REINFORCE training step (Adam, matching semi-QuCoNet-Adam) ──────

def reinforce_step(model, optimizer, qa_pairs, M, num_rollouts, grad_clip=None,
                   loss_type="standard", best_k=2, entropy_coef=0.0):
    """One epoch: matches semi-QuCoNet-Adam training exactly.

    - Reward: {0, 1}
    - Advantage: (R - per_QA_mean), no std normalization
    - Loss: -(advantage * log_prob).mean()
    - Optimizer: Adam (set externally)

    loss_type="bestk" maximizes the best-of-k objective E_QA[1-(1-p)^k]
    instead of E_QA[p]: chain rule reweights each QA's REINFORCE gradient
    by f'(p_hat) = k(1-p_hat)^(k-1), p_hat = per-QA rollout success mean.
    f'(1)=0, so solved pairs stop contributing (anti-collapse water-filling);
    pairs with p_hat=0 give no signal either way (REINFORCE has no successful
    trajectory to reinforce -- unlike the exact-forward quantum grover loss).

    entropy_coef > 0 adds the maximum-entropy regularizer
    -coef * sum_x H(pi(.|x)) over the full routing table (Ziebart/SAC
    style): keeps interior mass without designating a target confidence.
    Control for the "just add an entropy bonus" alternative to the
    interior-attractor objective.
    """
    N, K = model.N, model.K
    B = len(qa_pairs)
    total = B * num_rollouts
    device = model.logits.device

    # 1. Sample trajectories (no grad)
    probs_np = model.get_probs_numpy()
    neighbors_np = model.node_neighbors.cpu().numpy()

    all_nodes, all_actions, success, path_lengths = \
        sample_rollouts(probs_np, neighbors_np, qa_pairs, M, num_rollouts)

    train_sr = success.mean()
    avg_succ_steps = path_lengths[success].mean() if success.any() else float('inf')

    # 2. Compute log-probs WITH gradient
    probs_t = model.get_probs()  # [N, K] with grad

    nodes_t = [torch.tensor(n, dtype=torch.long, device=device) for n in all_nodes]
    actions_t = [torch.tensor(a, dtype=torch.long, device=device) for a in all_actions]
    pl_t = torch.tensor(path_lengths, dtype=torch.long, device=device)

    log_prob = torch.zeros(total, device=device)
    for step in range(M):
        nd = nodes_t[step]
        act = actions_t[step]
        step_probs = probs_t[nd, act]  # [total]
        step_log = torch.log(step_probs.clamp(min=1e-15))
        active_mask = (step < pl_t).float()
        log_prob = log_prob + active_mask * step_log

    # 3. Advantage: {0, 1} rewards, per-QA baseline (matching semi-QuCoNet-Adam)
    rewards = torch.tensor(success.astype(np.float32), device=device)
    rewards_per_qa = rewards.view(B, num_rollouts)
    baseline = rewards_per_qa.mean(dim=1, keepdim=True)
    advantage = (rewards_per_qa - baseline).view(-1)

    # 4. Loss: sum over QA pairs, mean over rollouts (gradient ∝ B)
    per_qa = -(advantage.detach() * log_prob).view(B, num_rollouts).mean(dim=1)
    if loss_type == "bestk":
        w = best_k * (1.0 - baseline.squeeze(1)).clamp(min=0.0) ** (best_k - 1)
        per_qa = w.detach() * per_qa
    loss = per_qa.sum()
    if entropy_coef > 0.0:
        ent = -(probs_t * torch.log(probs_t.clamp(min=1e-15))).sum()
        loss = loss - entropy_coef * ent

    # 5. Update
    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    p_hat = success.reshape(B, num_rollouts).mean(axis=1)
    train_bestk = float(np.mean(1.0 - (1.0 - p_hat) ** best_k))

    return float(train_sr), float(loss.item()), float(avg_succ_steps), train_bestk


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate_qa_set(model, qa_pairs, M, num_rollouts=10000, best_k=2):
    probs_np = model.get_probs_numpy()
    neighbors_np = model.node_neighbors.cpu().numpy()
    _, _, success, path_lengths = \
        sample_rollouts(probs_np, neighbors_np, qa_pairs, M, num_rollouts)
    sr = success.mean()
    avg_steps = path_lengths[success].mean() if success.any() else float('inf')
    p_hat = success.reshape(len(qa_pairs), num_rollouts).mean(axis=1)
    bestk_sr = float(np.mean(1.0 - (1.0 - p_hat) ** best_k))
    return sr, avg_steps, bestk_sr


# ─── Plotting ────────────────────────────────────────────────────────

def plot_training_curves(history, filename, B, num_val):
    has_valid = num_val > 0
    has_ipr = any('train_ipr' in h for h in history)
    n_panels = 2 + int(has_ipr)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True)
    epochs = [h['epoch'] for h in history]

    ax = axes[0]
    ax.plot(epochs, [h['train_sr'] for h in history], 'b-', lw=1.5, label='Train')
    if has_valid:
        ev = [(h['epoch'], h['valid_sr']) for h in history if 'valid_sr' in h]
        if ev:
            ax.plot(*zip(*ev), 'r--o', lw=1.5, ms=3, label='Valid')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'CoNet-Adam Training  (B_train={B}, B_valid={num_val})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    def _finite(lst):
        return [(e, v) for e, v in lst if v is not None and np.isfinite(v)]
    tr = _finite([(h['epoch'], h.get('train_steps')) for h in history])
    if tr:
        ax.plot(*zip(*tr), 'b-', lw=1.5, label='Train')
    if has_valid:
        va = _finite([(h['epoch'], h.get('valid_steps')) for h in history
                       if 'valid_steps' in h])
        if va:
            ax.plot(*zip(*va), 'r--o', lw=1.5, ms=3, label='Valid')
    ax.set_ylabel('Avg Steps to Success')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if has_ipr:
        ax = axes[2]
        ev = [(h['epoch'], h['train_ipr']) for h in history if 'train_ipr' in h]
        ax.plot(*zip(*ev), 'b-o', lw=1.5, ms=3, label='Train IPR')
        if has_valid:
            vev = [(h['epoch'], h['valid_ipr']) for h in history
                    if 'valid_ipr' in h]
            if vev:
                ax.plot(*zip(*vev), 'r--s', lw=1.5, ms=3, label='Valid IPR')
        ax.set_ylabel('IPR')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoNet with Adam optimizer (GRPO-style training)")
    parser.add_argument("--fname", "-f", default=None)
    parser.add_argument("-B", type=int, default=None)
    parser.add_argument("-N", type=int, default=80)
    parser.add_argument("-K", type=int, default=3)
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (default 0.05, matching semi-QuCoNet-Adam)")
    parser.add_argument("--num-rollouts", type=int, default=64000)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--eval-rollouts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--ckpt-freq", type=int, default=5,
                        help="Checkpoint freq (<=0 to disable)")
    parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam',
                        help="Optimizer (default: adam)")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Gradient clipping max norm (default: None)")
    parser.add_argument("--loss-type",
                        choices=['standard', 'bestk', 'bestk_exact',
                                 'capped_exact'],
                        default='standard',
                        help="standard: one-shot E[p] (REINFORCE); bestk: "
                             "best-of-k E[1-(1-p)^k] (REINFORCE reweight); "
                             "bestk_exact: same objective with exact gradients "
                             "via differentiable DP (k=1 -> exact one-shot); "
                             "capped_exact: E[min(p, cap)] with exact "
                             "gradients (imported confidence target)")
    parser.add_argument("--cap", type=float, default=0.25,
                        help="confidence cap for --loss-type capped_exact")
    parser.add_argument("--best-k", type=int, default=2,
                        help="k for --loss-type bestk (default 2)")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="max-entropy bonus coef on the routing table "
                             "(REINFORCE loss types only; default 0 = off)")
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping when SR and IPR converge")
    parser.add_argument("--es-tol", type=float, default=1e-4,
                        help="Early stopping tolerance (max-min over window)")
    parser.add_argument("--es-window", type=int, default=5,
                        help="Number of eval points to check for convergence")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    if B + num_val > len(all_qa):
        print(f"ERROR: B + num_val exceeds available QA pairs")
        sys.exit(1)

    train_qa = list(all_qa[:B])
    valid_qa = list(all_qa[-num_val:]) if num_val > 0 else []

    graph_stem = os.path.splitext(os.path.basename(args.fname))[0] \
        if args.fname else f"gen_N{N}_K{K}"

    props = analyze_graph_properties(G)
    print(f"Graph: N={N} K={K} M={M}, diameter={props['diameter']}")
    print(f"QA split: {B} train + {num_val} valid")

    # ── Build edge_index from graph ───────────────────────────────────
    edge_list = []
    for u, v in G.edges():
        edge_list.append([u, v])
        edge_list.append([v, u])
    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # [2, num_edges]

    # ── Setup model ───────────────────────────────────────────────────
    model = SoftmaxRouter(N, K, edge_index)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    opt_name = args.optimizer.upper()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params} (N*K = {N*K} logits)")
    print(f"Training: {args.epochs} epochs, lr={args.lr} ({opt_name}), "
          f"rollouts={args.num_rollouts}\n")

    # ── Output directory ──────────────────────────────────────────────
    # Pattern: conet_{opt}_{graph_stem}_B{B}_s{seed}_{timestamp}
    # The graph_stem already contains seedX, e.g. sliding_puzzle_..._seed2
    # This pattern is parseable by plot_three_way_comparison.py
    opt_tag = 'adam' if args.optimizer == 'adam' else 'sgd'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results',
        f"conet_{opt_tag}_{graph_stem}_B{B}_s{args.seed}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Epoch-0 evaluation ────────────────────────────────────────────
    history = []
    best_sr = 0.0
    best_state = None
    best_epoch = 0
    t0 = time.time()

    entry0 = {'epoch': 0, 'train_sr': 0.0, 'train_loss': None,
              'train_steps': None}
    tr_sr, tr_steps, tr_bk = evaluate_qa_set(
        model, train_qa, M, args.eval_rollouts, args.best_k)
    entry0['train_sr'] = tr_sr
    entry0['train_steps'] = tr_steps if math.isfinite(tr_steps) else None
    entry0['train_bestk'] = tr_bk

    if valid_qa:
        v_sr, v_steps, v_bk = evaluate_qa_set(
            model, valid_qa, M, args.eval_rollouts, args.best_k)
        entry0['valid_sr'] = v_sr
        entry0['valid_steps'] = v_steps if math.isfinite(v_steps) else None
        entry0['valid_bestk'] = v_bk

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
    best_sel = 0.0
    best_bestk = 0.0
    bestk_aligned = args.loss_type in ('bestk', 'bestk_exact',
                                       'capped_exact')
    for epoch in range(1, args.epochs + 1):
        if args.loss_type == 'capped_exact':
            train_sr, loss_val, succ_steps, train_bestk = exact_capped_step(
                model, optimizer, train_qa, M, args.cap, args.grad_clip,
                args.best_k)
        elif args.loss_type == 'bestk_exact':
            train_sr, loss_val, succ_steps, train_bestk = exact_bestk_step(
                model, optimizer, train_qa, M, args.best_k, args.grad_clip)
        else:
            train_sr, loss_val, succ_steps, train_bestk = reinforce_step(
                model, optimizer, train_qa, M, args.num_rollouts,
                args.grad_clip, args.loss_type, args.best_k,
                args.entropy_coef)

        entry = {
            'epoch': epoch,
            'train_sr': train_sr,
            'train_loss': loss_val,
            'train_steps': succ_steps if math.isfinite(succ_steps) else None,
            'train_bestk': train_bestk,
        }

        # best model = highest training value of the objective being optimized
        sel = train_bestk if bestk_aligned else train_sr
        if sel > best_sel:
            best_sel = sel
            best_sr = train_sr
            best_bestk = train_bestk
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch

        is_eval = (epoch % args.eval_freq == 0) or epoch == args.epochs

        if is_eval:
            if valid_qa:
                v_sr, v_steps, v_bk = evaluate_qa_set(
                    model, valid_qa, M, args.eval_rollouts, args.best_k)
                entry['valid_sr'] = v_sr
                entry['valid_steps'] = v_steps if math.isfinite(v_steps) \
                    else None
                entry['valid_bestk'] = v_bk

            t_ipr, _ = compute_batch_ipr(model, train_qa, M)
            entry['train_ipr'] = float(t_ipr)
            if valid_qa:
                v_ipr, _ = compute_batch_ipr(model, valid_qa, M)
                entry['valid_ipr'] = float(v_ipr)

            parts = [f"train_sr={train_sr:.4f}"]
            if bestk_aligned:
                parts.append(f"b@{args.best_k}={train_bestk:.4f}")
            if valid_qa:
                parts.append(f"valid_sr={entry['valid_sr']:.4f}")
            if succ_steps and math.isfinite(succ_steps):
                parts.append(f"steps={succ_steps:.1f}")
            parts.append(f"IPR={t_ipr:.3f}")
            if valid_qa:
                parts.append(f"v_IPR={v_ipr:.3f}")
            print(f"  Epoch {epoch:3d}: {', '.join(parts)}")

            # Early stopping check (on the objective actually optimized)
            sr_key = 'train_bestk' if bestk_aligned else 'train_sr'
            if args.early_stop and 'train_ipr' in entry:
                past = [h for h in history if 'train_ipr' in h]
                recent = (past + [entry])[-args.es_window:]
                if len(recent) >= args.es_window:
                    sr_vals = [h[sr_key] for h in recent]
                    ipr_vals = [h['train_ipr'] for h in recent]
                    sr_range = max(sr_vals) - min(sr_vals)
                    ipr_range = max(ipr_vals) - min(ipr_vals)
                    if sr_range < args.es_tol and ipr_range < args.es_tol:
                        print(f"  Early stop at epoch {epoch}: "
                              f"SR range={sr_range:.2e}, IPR range={ipr_range:.2e}")
                        history.append(entry)
                        break

        history.append(entry)

        if args.ckpt_freq > 0 and epoch % args.ckpt_freq == 0:
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                'config': {'N': N, 'K': K, 'M': M, 'B': B},
            }, os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  "
          f"Best train SR={best_sr:.4f} at epoch {best_epoch}")

    # ── Save outputs ──────────────────────────────────────────────────
    Q0, A0 = train_qa[0]
    tag = (f"{graph_stem}_conet_{opt_tag}_N{N}_K{K}_M{M}_B{B}"
           f"_Q{Q0}_A{A0}_{timestamp}")
    if args.label:
        tag = f"{args.label}_{tag}"

    model_path = os.path.join(output_dir, f"{tag}_best_model.pt")
    torch.save({
        'model_state_dict': best_state,
        'config': {'N': N, 'K': K, 'M': M, 'B': B},
        'best_epoch': best_epoch, 'best_sr': best_sr,
    }, model_path)
    print(f"Model: {model_path}")

    def _default(o):
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, np.ndarray):      return o.tolist()
        if isinstance(o, tuple):           return list(o)
        return str(o)

    results_path = os.path.join(output_dir, f"{tag}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'N': N, 'K': K, 'M': M, 'B': B, 'num_val': num_val,
                'lr': args.lr, 'epochs': args.epochs, 'seed': args.seed,
                'num_rollouts': args.num_rollouts,
                'mode': f'conet_{opt_tag}',
                'optimizer': opt_name,
                'reward_coding': '{0, 1}',
                'parameterization': 'softmax_logits',
                'loss_type': args.loss_type,
                'best_k': args.best_k,
                'entropy_coef': args.entropy_coef,
                'cap': args.cap,
            },
            'metrics': {
                'best_sr': best_sr, 'best_bestk': best_bestk,
                'best_epoch': best_epoch,
                'elapsed_s': elapsed
            },
            'train_qa': train_qa, 'valid_qa': valid_qa,
            'history': history,
        }, f, indent=2, default=_default)
    print(f"Results: {results_path}")

    plot_path = os.path.join(output_dir, f"{tag}_curves.png")
    plot_training_curves(history, plot_path, B, num_val)
    print(f"Output dir: {output_dir}")

    return history


if __name__ == "__main__":
    main()
