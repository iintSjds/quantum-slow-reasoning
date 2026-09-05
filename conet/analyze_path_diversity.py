"""
Analyze path diversity for classical and quantum CoNet models.

Path diversity is measured via IPR (Inverse Participation Ratio):
    IPR = 1 / Σ q̃ᵢ²
where q̃ᵢ = pᵢ / Σpⱼ are the normalized path probabilities.

IPR equals 1 when there's a single deterministic path,
2 when there are two equally-weighted paths, etc.

Accuracy (total success probability) P = Σpᵢ is reported separately.
Batch summary uses accuracy-weighted mean: ⟨IPR⟩ = Σ(Pⱼ × IPRⱼ) / Σ(Pⱼ).
"""

import os
import sys
import json
import glob
import torch
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# For quantum model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'quconet'))


# ─── Classical path sampling ───────────────────────────────────────────

def load_classical_checkpoint(ckpt_path):
    """Load classical CoNet checkpoint, return graph info and transition probs."""
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    edge_index = d['edge_index'].long()
    transition_probs = d['transition_probs'].numpy()
    node_ptr = d['node_ptr'].long().numpy()
    N = len(node_ptr) - 1
    K = node_ptr[1] - node_ptr[0]
    return edge_index, transition_probs, node_ptr, N, K


def compute_classical_path_diversity(edge_index, transition_probs, node_ptr,
                                     start, target, M, prob_threshold=1e-15):
    """
    Exact enumeration of all paths from start to target within M steps.
    Path probability = product of transition probabilities along the path.
    Returns (IPR, num_unique, top_paths, total_success_prob).
    """
    neighbors = edge_index[1].numpy()
    path_probs = Counter()

    # DFS: (node, step, path_tuple, cumulative_prob)
    stack = [(start, 0, (start,), 1.0)]

    while stack:
        node, step, path, cum_prob = stack.pop()

        if node == target:
            path_probs[path] += cum_prob
            continue

        if step >= M:
            continue

        s, e = node_ptr[node], node_ptr[node + 1]
        nbrs = neighbors[s:e]
        probs = transition_probs[s:e]
        probs = probs / probs.sum()  # normalize

        for i in range(len(nbrs)):
            new_prob = cum_prob * probs[i]
            if new_prob < prob_threshold:
                continue
            new_path = path + (int(nbrs[i]),)
            stack.append((int(nbrs[i]), step + 1, new_path, new_prob))

    total_success = sum(path_probs.values())
    if total_success < 1e-15:
        return 0.0, 0, [], total_success

    raw_probs = np.array(list(path_probs.values()))
    norm_probs = raw_probs / total_success
    IPR = 1.0 / np.sum(norm_probs ** 2)

    top = path_probs.most_common(5)
    top_paths = [(list(p), prob / total_success) for p, prob in top]

    return IPR, len(path_probs), top_paths, total_success


# ─── Quantum path diversity ───────────────────────────────────────────

def load_quantum_checkpoint(checkpoint_path):
    """
    Load quantum checkpoint and compute coin matrices.
    Following the approach from compare_classical_quantum_v2.py.

    Returns coin_matrices (N, K, K), shift_node_map (N, K), N, K, config.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = checkpoint['model_state_dict']

    coin_params = sd['coin.hamiltonian_params']
    N, num_params = coin_params.shape
    K = int((np.sqrt(8 * num_params + 1) - 1) // 2)

    triu_indices = sd['coin.triu_indices']
    coin_matrices = torch.zeros(N, K, K, dtype=torch.complex64)
    for node in range(N):
        H = torch.zeros(K, K, dtype=torch.float32)
        H[triu_indices[0], triu_indices[1]] = coin_params[node]
        H[triu_indices[1], triu_indices[0]] = coin_params[node]
        C = torch.matrix_exp(1j * H)
        coin_matrices[node] = C

    shift_node_map = sd['shift.shift_node_map']
    shift_coin_map = sd['shift.shift_coin_map']

    return coin_matrices, shift_node_map, shift_coin_map, N, K, checkpoint['config']


def generate_unique_coin_state(N, K, Q, A, max_length=None):
    """Generate unique coin state for QA pair (same as QuantumCoNetAR)."""
    n = 0
    while K ** (n + 1) <= N:
        n += 1

    def to_k_base(x, base, num_digits):
        digits = []
        for _ in range(num_digits):
            digits.append(x % base)
            x //= base
        return digits[::-1]

    q_digits = to_k_base(Q, K, n + 1)
    a_digits = to_k_base(A, K, n + 1)

    coin_state = []
    for i in range(n + 1):
        coin_state.append(q_digits[i])
        coin_state.append(a_digits[i])

    if max_length is not None and max_length > len(coin_state):
        repeated = []
        while len(repeated) < max_length - len(coin_state):
            repeated.extend(coin_state)
        coin_state.extend(repeated[:max_length - len(coin_state)])
    elif max_length is not None and max_length < len(coin_state):
        coin_state = coin_state[:max_length]

    return coin_state


def compute_quantum_path_diversity(coin_matrices, shift_node_map, shift_coin_map,
                                   start, target, N, K, M, prob_threshold=1e-15):
    """
    Compute path diversity for quantum AR model using coin-matrix transition probs.

    In AR mode, each step s uses a DIFFERENT coin dimension. The coin input at step s
    is initial_coin_pattern[s] (set at initialization, independent of previous steps).
    Transition probability at step s, node x, direction k:
        p = |C_x[initial_coin_pattern[s], k]|^2

    Path probability = product of step-level transition probabilities.
    Returns (IPR, num_unique, top_paths, total_success_prob).
    """
    initial_coins = generate_unique_coin_state(N, K, start, target, max_length=M)

    coin_probs = (coin_matrices.abs() ** 2).numpy()  # (N, K, K)
    snm = shift_node_map.numpy()  # (N, K)

    path_probs = Counter()

    # DFS: (node, step, path_tuple, cumulative_prob)
    # coin_in at each step is fixed by initial_coins[step]
    stack = [(start, 0, (start,), 1.0)]

    while stack:
        node, step, path, cum_prob = stack.pop()

        if node == target:
            path_probs[path] += cum_prob
            continue

        if step >= M:
            continue

        coin_in = initial_coins[step]

        for k_out in range(K):
            trans_prob = coin_probs[node, coin_in, k_out]
            new_prob = cum_prob * trans_prob

            if new_prob < prob_threshold:
                continue

            neighbor = int(snm[node, k_out])
            new_path = path + (neighbor,)
            stack.append((neighbor, step + 1, new_path, new_prob))

    total_success = sum(path_probs.values())
    if total_success < 1e-15:
        return 0.0, 0, [], total_success

    raw_probs = np.array(list(path_probs.values()))
    norm_probs = raw_probs / total_success
    IPR = 1.0 / np.sum(norm_probs ** 2)

    top = path_probs.most_common(5)
    top_paths = [(list(p), prob / total_success) for p, prob in top]

    return IPR, len(path_probs), top_paths, total_success


def _print_batch_summary(results):
    """Print accuracy-weighted mean IPR summary grouped by batch size B."""
    by_B = defaultdict(list)
    for qa_label, qa_results in results.items():
        for r in qa_results:
            by_B[r['B']].append((r['IPR'], r['total_success_prob']))

    print(f"\n  {'─'*50}")
    print(f"  Batch summary (accuracy-weighted mean IPR):")
    for B in sorted(by_B.keys()):
        pairs = by_B[B]
        iprs = np.array([p[0] for p in pairs])
        accs = np.array([p[1] for p in pairs])
        sum_acc = accs.sum()
        if sum_acc > 0:
            weighted_ipr = (accs * iprs).sum() / sum_acc
        else:
            weighted_ipr = 0.0
        mean_acc = accs.mean()
        print(f"  B={B:3d}:  <IPR>_w = {weighted_ipr:.3f}, mean_acc = {mean_acc:.4f}  ({len(pairs)} QA pairs)")


def analyze_quantum_expr2(N=80, K=3, M=8, batch_sizes=None):
    """Analyze path diversity for expr2 quantum runs."""
    quantum_dir = "archive/expr2/quantum"

    if not os.path.exists(quantum_dir):
        print(f"Quantum directory not found: {quantum_dir}")
        return None

    import re

    all_files = os.listdir(quantum_dir)
    model_files = [f for f in all_files if f.endswith('_best_model.pt') and f'N{N}' in f]

    qa_groups = defaultdict(list)
    for f in model_files:
        m = re.match(rf'quconet_ar_demo_N{N}_K{K}_M{M}_B(\d+)_Q(\d+)_A(\d+)_(\d+_\d+)_best_model\.pt', f)
        if m:
            B_val = int(m.group(1))
            Q = int(m.group(2))
            A = int(m.group(3))
            timestamp = m.group(4)
            qa_groups[(Q, A)].append((B_val, f, timestamp))

    expr2_qa = {qa: sorted(runs) for qa, runs in qa_groups.items() if len(runs) >= 14}
    print(f"Found {len(expr2_qa)} expr2 QA pairs for quantum N={N}")

    results = {}

    for qa, runs in expr2_qa.items():
        Q, A = qa
        print(f"\n  QA pair: Q={Q}, A={A}")
        qa_results = []

        for B_val, model_file, timestamp in runs:
            if batch_sizes and B_val not in batch_sizes:
                continue

            model_path = os.path.join(quantum_dir, model_file)

            # Load results for success rate
            results_file = model_file.replace('_best_model.pt', '_results.json')
            results_path = os.path.join(quantum_dir, results_file)
            success_rate = None
            if os.path.exists(results_path):
                with open(results_path) as f:
                    rdata = json.load(f)
                    success_rate = rdata['metrics'].get('best_success_rate', None)

            # Load checkpoint and compute coin matrices
            coin_matrices, shift_node_map, shift_coin_map, N_loaded, K_loaded, config = \
                load_quantum_checkpoint(model_path)

            IPR, n_unique, top_paths, total_success = \
                compute_quantum_path_diversity(
                    coin_matrices, shift_node_map, shift_coin_map,
                    Q, A, N, K, M)

            sr_str = f"{success_rate:.4f}" if success_rate is not None else "N/A"
            print(f"    B={B_val:3d}: IPR={IPR:.3f}, unique={n_unique:4d}, "
                  f"p_success={total_success:.4f}, sr={sr_str}")
            for path, prob in top_paths[:3]:
                print(f"           path (len={len(path)-1}): {path} p={prob:.4f}")

            qa_results.append({
                'B': B_val,
                'IPR': IPR,
                'num_unique': n_unique,
                'total_success_prob': total_success,
                'success_rate': success_rate,
                'top_paths': [(list(p), prob) for p, prob in top_paths[:3]],
            })

        results[f"Q{Q}A{A}"] = qa_results

    _print_batch_summary(results)
    return results


# ─── Analysis over experiment sets ────────────────────────────────────

def analyze_classical_expr2(N=80, K=3, M=8, batch_sizes=None):
    """
    Analyze path diversity for expr2 classical runs.
    Expr2 QA pairs have 14 batch sizes; expr1 has 7.
    """
    ckpt_base = f"archive/classical_ckpts/N_{N}_k_{K}_M_{M}"
    results_base = f"archive/expr2/classical/N_{N}_k_{K}_M_{M}"

    if not os.path.exists(ckpt_base):
        print(f"Checkpoint directory not found: {ckpt_base}")
        return None

    all_dirs = sorted(os.listdir(ckpt_base))

    # Group by QA pair
    import re
    qa_groups = defaultdict(list)
    for d in all_dirs:
        m = re.match(r'archive_Q(\d+)A(\d+)__', d)
        if m:
            qa = (int(m.group(1)), int(m.group(2)))
            bm = re.search(r'num_questions_(\d+)', d)
            if bm:
                B = int(bm.group(1))
                qa_groups[qa].append((B, d))

    # Filter to expr2 QA pairs (14 batch sizes)
    expr2_qa = {qa: sorted(runs) for qa, runs in qa_groups.items() if len(runs) >= 14}
    print(f"Found {len(expr2_qa)} expr2 QA pairs")

    results = {}

    for qa, runs in expr2_qa.items():
        Q, A = qa
        print(f"\n  QA pair: Q={Q}, A={A}")
        qa_results = []

        for B, dirname in runs:
            if batch_sizes and B not in batch_sizes:
                continue

            ckpt_dir = os.path.join(ckpt_base, dirname)
            ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
            if not ckpt_files:
                continue
            last_ckpt = ckpt_files[-1]
            step = int(os.path.basename(last_ckpt).replace('step_', '').replace('.pt', ''))

            edge_index, tp, node_ptr, N_actual, K_actual = load_classical_checkpoint(last_ckpt)

            # Load success rate from results
            results_dir = os.path.join(results_base, dirname)
            success_rate = None
            if os.path.exists(os.path.join(results_dir, 'valid.json')):
                with open(os.path.join(results_dir, 'valid.json')) as f:
                    valid = json.load(f)
                    if valid:
                        success_rate = valid[-1].get('success_rate', None)

            # Exact path enumeration
            IPR, n_unique, top_paths, total_success = \
                compute_classical_path_diversity(edge_index, tp, node_ptr, Q, A, M)

            sr_str = f"{success_rate:.4f}" if success_rate is not None else "N/A"
            print(f"    B={B:3d}: IPR={IPR:.3f}, unique={n_unique:4d}, "
                  f"p_success={total_success:.4f}, sr={sr_str}")
            for path, prob in top_paths[:3]:
                print(f"           path (len={len(path)-1}): {list(path)} p={prob:.4f}")

            qa_results.append({
                'B': B,
                'IPR': IPR,
                'num_unique': n_unique,
                'total_success_prob': total_success,
                'success_rate': success_rate,
                'step': step,
                'top_paths': [(list(p), prob) for p, prob in top_paths[:3]],
            })

        results[f"Q{Q}A{A}"] = qa_results

    _print_batch_summary(results)
    return results


def plot_path_diversity(results, title="Path Diversity vs Batch Size", save_path=None):
    """Plot IPR and accuracy vs batch size B (two-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1, ax2 = axes

    for qa_label, qa_results in results.items():
        Bs = [r['B'] for r in qa_results]
        IPRs = [r['IPR'] for r in qa_results]
        accs = [r['total_success_prob'] for r in qa_results]

        ax1.plot(Bs, IPRs, 'o-', label=qa_label, markersize=4)
        ax2.plot(Bs, accs, 's-', label=qa_label, markersize=4)

    # Add accuracy-weighted mean IPR line
    by_B = defaultdict(list)
    for qa_label, qa_results in results.items():
        for r in qa_results:
            by_B[r['B']].append((r['IPR'], r['total_success_prob']))
    Bs_sorted = sorted(by_B.keys())
    mean_iprs = []
    for B in Bs_sorted:
        pairs = by_B[B]
        iprs = np.array([p[0] for p in pairs])
        accs = np.array([p[1] for p in pairs])
        s = accs.sum()
        mean_iprs.append((accs * iprs).sum() / s if s > 0 else 0.0)
    ax1.plot(Bs_sorted, mean_iprs, 'k^--', label='<IPR>_w', markersize=6, linewidth=2)

    ax1.set_xlabel('Batch size B')
    ax1.set_ylabel('IPR (effective # of paths)')
    ax1.set_xscale('log', base=2)
    ax1.set_title('Path Diversity (IPR)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Batch size B')
    ax2.set_ylabel('Accuracy (total success prob)')
    ax2.set_xscale('log', base=2)
    ax2.set_title('Accuracy')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=80)
    parser.add_argument('-K', type=int, default=3)
    parser.add_argument('-M', type=int, default=8)
    parser.add_argument('--batch-sizes', type=str, default=None,
                        help='Comma-separated batch sizes to analyze (default: all)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['classical', 'quantum', 'both'],
                        help='Which model to analyze')
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')] if args.batch_sizes else None

    print(f"Analyzing path diversity (IPR) for N={args.N} K={args.K} M={args.M}")
    print(f"Batch sizes: {batch_sizes or 'all'}")
    print(f"Using exact path enumeration (K^M = {args.K}^{args.M} = {args.K**args.M})")

    if args.mode in ('classical', 'both'):
        print(f"\n{'='*60}")
        print(f"CLASSICAL CoNet")
        print(f"{'='*60}")
        classical_results = analyze_classical_expr2(
            N=args.N, K=args.K, M=args.M, batch_sizes=batch_sizes)

    if args.mode in ('quantum', 'both'):
        print(f"\n{'='*60}")
        print(f"QUANTUM QuCoNet")
        print(f"{'='*60}")
        quantum_results = analyze_quantum_expr2(
            N=args.N, K=args.K, M=args.M, batch_sizes=batch_sizes)
