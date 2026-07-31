"""Full-Hilbert-space verification of train-AR / infer-coherently + AA.

Builds the ACTUAL inference circuit for a trained QuCoNet-AR instance and
checks the two claims of the manuscript's Realization section numerically:

  1. |psi> = A |Q, c_in>, evolved coherently (no intermediate measurement)
     over H_pos (x) H_c1 (x) ... (x) H_cM  [dim N * K^M = 120 * 3^8 = 787,320],
     has good-subspace weight <psi|P_G|psi> EQUAL to the classical absorbed
     success probability p from exact path enumeration.  P_G is diagonal in
     the coin (which-path) registers: a coin string is good iff its decoded
     walk from Q first reaches the target within M steps (first-arrival
     prefix property) -- absorption without breaking unitarity.
  2. One amplification round G = A R_0 A^dag R_G rotates the state so that
     ||P_G G|psi>||^2 = sin^2(3 arcsin sqrt(p)) exactly (and k rounds give
     sin^2((2k+1) theta)), i.e. the AA accuracy map used throughout the
     paper is realized by the explicit circuit, which-path registers and all.

The walk step is unitary: coin C_x = exp(iH_x) acts on register s
conditioned on position (input channel = register content), then the shift
moves the position conditioned on register s.  For the sliding puzzle each
shift column snm[:,k] is a permutation of nodes (V involution, L/R
3-cycles), so the shift is unitary with the register left as the action
record (checked at load).  Equivalent qubit count: ceil(log2 N) + M*ceil(log2 K)
= 7 + 16 = 23 (qudit space embeds in the qubit space; unitaries extend by
identity on the padding).

Usage:
  python circuit_verify.py [--ckpt <path>] [--rounds 1 2] [--pairs 96]
Defaults to the grover_n1 seed1 B=32 sliding-puzzle run.
"""
import os, sys, glob, json, math, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amplification_scaling as amp
import amplification_step1 as s1

ROOT = os.environ.get("QSR_ROOT", os.path.abspath(os.path.join(HERE, "..")))


def build_A(cm, snm, scm, N, K, M):
    """Return apply_A(psi) and apply_Adag(psi) for the M-step coherent walk.

    psi: complex tensor [N, K, K, ..., K] (M coin registers).
    Step s: coin op on register s (input channel = register content,
    amplitude <k|C_x|c> = C_x[c, k], matching the AR enumerator), then the
    flip-flop shift |x, k> -> |snm[x,k], scm[x,k]> -- a bijection on the
    (position, register) product space (asserted), hence unitary even
    though the per-slot columns snm[:,k] are not node permutations.
    """
    cmn = cm.numpy().astype(np.complex128)          # [N, K, K]  C_x[c_in, k]
    snm = snm.numpy().astype(np.int64)              # [N, K]
    scm = scm.numpy().astype(np.int64)              # [N, K]
    pairs = {(int(snm[x, k]), int(scm[x, k])) for x in range(N) for k in range(K)}
    assert len(pairs) == N * K, "shift (snm, scm) is not a bijection on N*K"

    # flattened index maps for the product-space permutation
    src = (np.arange(N)[:, None] * K + np.arange(K)[None, :]).reshape(-1)
    dst = (snm * K + scm).reshape(-1)
    inv_perm = np.empty(N * K, dtype=np.int64)
    inv_perm[dst] = src

    def step(psi, s, forward=True):
        # bring register s next to the position axis: [N, K, rest]
        psi = np.moveaxis(psi, 1 + s, 1)
        shp = psi.shape
        psi = psi.reshape(N, K, -1)
        if forward:
            # coin: out[x, k, r] = sum_c C_x[c, k] in[x, c, r]
            psi = np.einsum("xck,xcr->xkr", cmn, psi)
            # shift: product-space permutation
            flat = psi.reshape(N * K, -1)
            out = np.empty_like(flat)
            out[dst, :] = flat[src, :]
            out = out.reshape(N, K, -1)
        else:
            flat = psi.reshape(N * K, -1)
            tmp = np.empty_like(flat)
            tmp[src, :] = flat[dst, :]
            tmp = tmp.reshape(N, K, -1)
            # coin^dag: out[x, c, r] = sum_k conj(C_x[c, k]) tmp[x, k, r]
            out = np.einsum("xck,xkr->xcr", cmn.conj(), tmp)
        out = out.reshape(shp)
        return np.moveaxis(out, 1, 1 + s)

    def apply_A(psi):
        for s in range(M):
            psi = step(psi, s, forward=True)
        return psi

    def apply_Adag(psi):
        for s in reversed(range(M)):
            psi = step(psi, s, forward=False)
        return psi

    return apply_A, apply_Adag


def good_mask(snm, scm, N, K, M, Q, target):
    """Boolean [N] + [K]*M mask over FINAL basis states (x_M, r_1..r_M).

    Enumerate all K^M action sequences from Q, tracking the node and the
    record scm[x, k] each shift leaves in its register; mark the final
    basis state iff the path's FIRST arrival at the target occurs within
    M steps (absorption semantics).  Backward determinism of the unitary
    walk (each shift is a product-space bijection; the coin touches only
    the not-yet-frozen register) guarantees paths <-> final basis states
    are one-to-one, so this diagonal projector is exactly the absorbed
    good subspace -- no interference between good and bad paths.
    """
    snm = snm.numpy().astype(np.int64) if torch.is_tensor(snm) else snm
    scm = scm.numpy().astype(np.int64) if torch.is_tensor(scm) else scm
    mask = np.zeros((N,) + (K,) * M, dtype=bool)
    def dfs(x, depth, records, hit):
        hit = hit or (x == target)
        if depth == M:
            if hit:
                mask[(x,) + records] = True
            return
        for k in range(K):
            dfs(int(snm[x, k]), depth + 1, records + (int(scm[x, k]),), hit)
    dfs(Q, 0, (), False)
    return mask


def run_pair(applyA, applyAdag, snm, scm, N, K, M, Q, target, coins_in, rounds):
    psi0 = np.zeros((N,) + (K,) * M, dtype=np.complex128)
    psi0[(Q,) + tuple(coins_in)] = 1.0
    psi = applyA(psi0.copy())
    g = good_mask(snm, scm, N, K, M, Q, target)
    p = float(np.sum(np.abs(psi[g]) ** 2))

    out = {"p_circuit": p, "rounds": {}}
    phi = psi.copy()
    for r in range(1, max(rounds) + 1):
        # R_G: flip sign on good subspace
        phi = np.where(g, -phi, phi)
        # A R_0 A^dag:  R_0 = 2|init><init| - I
        phi = applyAdag(phi)
        a0 = phi[(Q,) + tuple(coins_in)]
        phi = -phi
        phi[(Q,) + tuple(coins_in)] += 2.0 * a0
        phi = applyA(phi)
        if r in rounds:
            out["rounds"][r] = float(np.sum(np.abs(phi[g]) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="quantum checkpoint (default: grover_n1 s1 B32)")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--pairs", type=int, default=96,
                    help="max QA pairs to verify (train first, then valid)")
    ap.add_argument("--out", default=os.path.join(HERE, "_sweep_out"))
    args = ap.parse_args()

    apd = amp._import_enumerators(ROOT)
    if args.ckpt is None:
        hits = glob.glob(os.path.join(
            ROOT, "from4090/grover_sweep/**/grover_n1_s1_B32_*best_model.pt"),
            recursive=True)
        args.ckpt = sorted(hits)[0]
    print(f"checkpoint: {os.path.relpath(args.ckpt, ROOT)}")
    cm, snm, scm, N, K, cfg = apd.load_quantum_checkpoint(args.ckpt)
    M = 8
    rj = args.ckpt.replace("_best_model.pt", "_results.json")
    tr, va, _ = s1.split_qa(rj)
    qa = [("train", q, a) for q, a in tr] + [("valid", q, a) for q, a in va]
    qa = qa[: args.pairs]

    applyA, applyAdag = build_A(cm, snm, scm, N, K, M)
    print(f"Hilbert space: N*K^M = {N}*{K}^{M} = {N * K**M:,} "
          f"(~{math.ceil(math.log2(N)) + M * math.ceil(math.log2(K))} qubits)")

    rows, dev_p, dev_r = [], 0.0, {r: 0.0 for r in args.rounds}
    for split, Q, A in qa:
        coins_in = apd.generate_unique_coin_state(N, K, Q, A, max_length=M)
        res = run_pair(applyA, applyAdag, snm, scm, N, K, M, Q, A, coins_in,
                       set(args.rounds))
        p_enum = float(apd.compute_quantum_path_diversity(
            cm, snm, scm, Q, A, N, K, M)[3])
        dev_p = max(dev_p, abs(res["p_circuit"] - p_enum))
        th = math.asin(math.sqrt(min(max(res["p_circuit"], 0.0), 1.0)))
        row = {"split": split, "Q": Q, "A": A,
               "p_enum": p_enum, "p_circuit": res["p_circuit"]}
        for r in args.rounds:
            pred = math.sin((2 * r + 1) * th) ** 2
            row[f"round{r}"] = res["rounds"][r]
            row[f"pred{r}"] = pred
            dev_r[r] = max(dev_r[r], abs(res["rounds"][r] - pred))
        rows.append(row)
        print(f"  {split:>5} Q={Q:>3} A={A:>3}: p_enum={p_enum:.10f} "
              f"p_circ={res['p_circuit']:.10f} | " +
              " ".join(f"r{r}={res['rounds'][r]:.6f}(pred {row[f'pred{r}']:.6f})"
                       for r in args.rounds))

    print("\n===== summary =====")
    print(f"pairs verified: {len(rows)}")
    print(f"max |p_circuit - p_enumeration|          : {dev_p:.3e}")
    for r in args.rounds:
        print(f"max |round-{r} success - sin^2({2*r+1} theta)| : {dev_r[r]:.3e}")
    for split in ("train", "valid"):
        sel = [x for x in rows if x["split"] == split]
        if sel:
            print(f"{split}: mean p = {np.mean([x['p_enum'] for x in sel]):.4f}, "
                  f"mean round-1 = {np.mean([x['round1'] for x in sel]):.4f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "circuit_verify.json"), "w") as f:
        json.dump({"ckpt": os.path.relpath(args.ckpt, ROOT), "rows": rows,
                   "max_dev_p": dev_p,
                   "max_dev_rounds": {str(r): dev_r[r] for r in args.rounds}},
                  f, indent=1)
    print(f"-> {os.path.join(args.out, 'circuit_verify.json')}")


if __name__ == "__main__":
    main()
