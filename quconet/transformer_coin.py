"""Transformer coin for QuCoNet-AR: a shared-weight neural policy in the coin slot.

Replaces the per-node U(3) lookup table (CoinOperatorAR) by a small causal
transformer that maps (question, move history so far) to the six angles of a
real-symmetric Hamiltonian, from which the step coin C = exp(iH) is built with
the exact same construction as the tabular model. Everything else -- the
question-dependent input channels, the flip-flop shift, first-arrival
absorption, the good subspace, the grover loss -- is unchanged.

Quantum status: at step s the angles are a deterministic classical function of
the which-path records already written, so the step is the block-diagonal
controlled unitary U_s = sum_prefix |prefix><prefix| (x) C_theta(prefix): a
branch-dependent coin. The walk therefore remains exactly classical in
distribution (which-path), while the prepared state stays a coherent pure
superposition that the Grover mirror can rotate.

Because the coin depends on the path prefix, the position-Markov DP shortcut
does not apply; exact success probabilities are computed by a level-synchronous
prefix-tree sweep instead (`success_probs`): at level t all live prefixes have
equal length, so they batch through the transformer without padding, children
landing on the target are absorbed (first arrival), and the accumulated
absorbed weight is the exact, differentiable p_i.

Context modes:
    "path": tokens = [question, (move_1, node_1), ..., (move_t, node_t)]
            -- the full AR reasoning model (default).
    "node": tokens = [question, node_t @ position t]
            -- Markov special case (policy sees only the current node + step),
            useful for bring-up and for cross-checks against tabular behavior.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .operators_ar import CoinOperatorAR
from .quconet_ar import QuantumCoNetAR


class TransformerCoinAR(nn.Module):
    """Causal-transformer coin over the move vocabulary, exact tree forward."""

    def __init__(self, N: int, K: int, max_steps: int,
                 adjacency_list: Sequence[Sequence[int]],
                 d_model: int = 32, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int = 64, context: str = "path",
                 device: Optional[torch.device] = None):
        super().__init__()
        assert context in ("path", "node")
        self.N, self.K, self.M = N, K, max_steps
        self.context = context
        self.d_model = d_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # graph: neighbor table nbr[x, k] = k-th neighbor of x (same order the
        # tabular ShiftOperatorAR uses to build shift_node_map)
        nbr = torch.tensor([[adjacency_list[x][k] for k in range(K)]
                            for x in range(N)], dtype=torch.long)
        self.register_buffer("nbr", nbr)

        # embeddings: question nodes (two tables so Q and A are distinguished),
        # visited node, move taken, position along the sequence
        self.q_emb = nn.Embedding(N, d_model)
        self.a_emb = nn.Embedding(N, d_model)
        self.node_emb = nn.Embedding(N, d_model)
        self.move_emb = nn.Embedding(K, d_model)
        self.pos_emb = nn.Embedding(max_steps + 1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # head -> 6 angles; bias initialized at the flat-coin Hamiltonian so
        # every prefix starts near the uniform 1/sqrt(K) coin, exactly like the
        # tabular init (identity coin would kill 2 of 3 routes)
        n_ang = K * (K + 1) // 2
        self.head = nn.Linear(d_model, n_ang)
        base_H = CoinOperatorAR._compute_flat_coin_hamiltonian(K)
        iu = torch.triu_indices(K, K, offset=0)
        self.register_buffer("triu_idx", iu)
        with torch.no_grad():
            self.head.bias.copy_(base_H[iu[0], iu[1]])
            self.head.weight.mul_(0.05)

        self.to(self.device)

    # ── coin construction (identical math to CoinOperatorAR) ─────────────
    def angles_to_probs(self, ang: torch.Tensor, channel: torch.Tensor
                        ) -> torch.Tensor:
        """(L, 6) angles + (L,) input channel -> (L, K) move probabilities.

        H symmetric from the upper triangle, C = exp(iH), probs = |C[c, .]|^2.
        """
        L, K = ang.shape[0], self.K
        H = torch.zeros(L, K, K, dtype=ang.dtype, device=ang.device)
        H[:, self.triu_idx[0], self.triu_idx[1]] = ang
        H = H + H.transpose(-1, -2)
        eye = torch.eye(K, dtype=torch.bool, device=ang.device)
        H[:, eye] = H[:, eye] / 2
        C = torch.matrix_exp(1j * H.to(torch.complex64))          # (L, K, K)
        row = C[torch.arange(L, device=ang.device), channel, :]   # (L, K)
        return row.abs() ** 2

    # ── transformer: batched prefixes of equal length -> angles ──────────
    def prefix_angles(self, Q: torch.Tensor, A: torch.Tensor,
                      moves: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """(L,) Q, (L,) A, (L, t) moves, (L, t) nodes-reached -> (L, 6).

        Level-synchronous: every prefix in the batch has the same length t,
        so no padding or mask is needed; reading the last position of a
        full-attention encoder equals the causal readout at that position.
        """
        L, t = moves.shape
        tok0 = (self.q_emb(Q) + self.a_emb(A)
                + self.pos_emb(torch.zeros_like(Q)))              # (L, d)
        if self.context == "path" or t == 0:
            toks = [tok0.unsqueeze(1)]
            if t > 0:
                pos = torch.arange(1, t + 1, device=Q.device)
                toks.append(self.move_emb(moves) + self.node_emb(xs)
                            + self.pos_emb(pos).unsqueeze(0))
            seq = torch.cat(toks, dim=1)                          # (L, t+1, d)
        else:                                                     # "node"
            pos = torch.full((L,), t, dtype=torch.long, device=Q.device)
            cur = (self.node_emb(xs[:, -1]) + self.pos_emb(pos)).unsqueeze(1)
            seq = torch.cat([tok0.unsqueeze(1), cur], dim=1)      # (L, 2, d)
        h = self.encoder(seq)[:, -1]                              # (L, d)
        return self.head(h)                                       # (L, 6)

    # ── exact success probabilities by prefix-tree sweep ─────────────────
    def success_probs(self, qa_pairs: Sequence[Tuple[int, int]],
                      prune: float = 0.0) -> torch.Tensor:
        """Exact differentiable p_i for each (Q, A), first-arrival within M.

        prune: drop live branches with weight < prune (0 = fully exact).
        """
        dev = self.device
        B = len(qa_pairs)
        Qb = torch.tensor([q for q, _ in qa_pairs], dtype=torch.long, device=dev)
        Ab = torch.tensor([a for _, a in qa_pairs], dtype=torch.long, device=dev)
        pat = torch.tensor(
            [QuantumCoNetAR.generate_unique_coin_state(self.N, self.K,
                                                       int(q), int(a),
                                                       max_length=self.M)
             for q, a in qa_pairs], dtype=torch.long, device=dev)  # (B, M)

        p_acc = torch.zeros(B, device=dev)
        qidx = torch.arange(B, device=dev)          # question of each prefix
        node = Qb.clone()                           # current node
        w = torch.ones(B, device=dev)               # path weight (grad-carrying)
        moves = torch.zeros(B, 0, dtype=torch.long, device=dev)
        xs = torch.zeros(B, 0, dtype=torch.long, device=dev)

        # absorb Q == A at the start (never happens in distance-6 pools)
        hit = node == Ab[qidx]
        if hit.any():
            p_acc = p_acc.index_add(0, qidx[hit], w[hit])
            keep = ~hit
            qidx, node, w = qidx[keep], node[keep], w[keep]
            moves, xs = moves[keep], xs[keep]

        for t in range(self.M):
            if qidx.numel() == 0:
                break
            ang = self.prefix_angles(Qb[qidx], Ab[qidx], moves, xs)
            probs = self.angles_to_probs(ang, pat[qidx, t])       # (L, K)
            L, K = probs.shape
            cw = (w.unsqueeze(1) * probs).reshape(-1)             # (L*K,)
            cq = qidx.repeat_interleave(K)
            cnode = self.nbr[node].reshape(-1)                    # (L*K,)
            cmoves = torch.cat(
                [moves.repeat_interleave(K, dim=0),
                 torch.arange(K, device=dev).repeat(L).unsqueeze(1)], dim=1)
            cxs = torch.cat(
                [xs.repeat_interleave(K, dim=0), cnode.unsqueeze(1)], dim=1)

            hit = cnode == Ab[cq]
            if hit.any():
                p_acc = p_acc.index_add(0, cq[hit], cw[hit])
            live = ~hit
            if t + 1 == self.M:
                break
            if prune > 0.0:
                live = live & (cw > prune)
            qidx, node, w = cq[live], cnode[live], cw[live]
            moves, xs = cmoves[live], cxs[live]
        return p_acc

    # ── losses (identical response functions to the tabular model) ───────
    def loss(self, qa_pairs, loss_type: str = "grover", grover_n: int = 1,
             best_k: int = 2, prune: float = 0.0):
        p = self.success_probs(qa_pairs, prune=prune)
        if loss_type == "standard":
            return -torch.sum(p), p
        if loss_type == "grover":
            return -torch.sum(QuantumCoNetAR.grover_acc(p, grover_n)), p
        if loss_type == "bestk":
            return -torch.sum(1.0 - (1.0 - p.clamp(max=1 - 1e-7)) ** int(best_k)), p
        raise ValueError(f"unknown loss_type {loss_type}")

    # ── independent reference: per-prefix probs for the DFS enumerator ───
    @torch.no_grad()
    def step_probs_single(self, Q: int, A: int, moves: List[int],
                          xs: List[int], step: int) -> List[float]:
        """Move probabilities for ONE prefix (reference path for validation).

        Uses the same weights but none of the tree bookkeeping: a batch of
        one through prefix_angles / angles_to_probs.
        """
        dev = self.device
        t = len(moves)
        ang = self.prefix_angles(
            torch.tensor([Q], device=dev), torch.tensor([A], device=dev),
            torch.tensor([moves], dtype=torch.long, device=dev).reshape(1, t),
            torch.tensor([xs], dtype=torch.long, device=dev).reshape(1, t))
        pat = QuantumCoNetAR.generate_unique_coin_state(
            self.N, self.K, Q, A, max_length=self.M)
        probs = self.angles_to_probs(
            ang, torch.tensor([pat[step]], dtype=torch.long, device=dev))
        return probs[0].detach().cpu().double().tolist()

    @torch.no_grad()
    def enumerate_success_prob(self, Q: int, A: int,
                               threshold: float = 0.0) -> float:
        """Brute-force DFS enumeration (float64 accumulation), the validation
        reference for success_probs -- same absorption semantics as the
        tabular enumerator in quconet_rl_training_ar._compute_qa_ipr."""
        total = 0.0
        stack = [(Q, 0, [], [], 1.0)]
        while stack:
            nd, st, mv, xh, cp = stack.pop()
            if nd == A:
                total += cp
                continue
            if st >= self.M:
                continue
            probs = self.step_probs_single(Q, A, mv, xh, st)
            for k in range(self.K):
                new_p = cp * probs[k]
                if threshold and new_p < threshold:
                    continue
                nxt = int(self.nbr[nd, k])
                stack.append((nxt, st + 1, mv + [k], xh + [nxt], new_p))
        return total
