# Data archive mount

This directory holds the raw experiment archive (training checkpoints,
per-run `results.json`, and the graph/QA pools).  It is not tracked by git.

Populate it from the Zenodo deposit (DOI: to be added at submission), or
point the scripts at an existing copy elsewhere by setting

```bash
export QSR_ROOT=/path/to/dir-containing-from4090
```

(`QSR_ROOT` is the *parent* of `from4090/`.)

The directory name is kept from the original archive so that every script,
cache key, and archived path matches verbatim.

## Expected layout

| subdirectory            | size   | contents |
|-------------------------|--------|----------|
| `grover_sweep/`         | 175 MB | main N=120 sliding-puzzle sweep: Grover-n=1..4 coin walkers and one-shot / best-of-k classical families at 32 seeds x B; capped / entropy control families at 32 seeds at B=32; semi (measured-walk) best-of-k families at 32 seeds at B=32 and eight at B in {8,128} (B=32 semi records re-run under the converged min-loss selection; superseded 8-seed records archived alongside) (Grover n=1 low-B cells re-recorded at the converged min-loss checkpoint; superseded max-SR records archived alongside) |
| `grover_sweep_bigB/`    | 16 MB  | large-B capacity scan on the 768-pair distance-6 pools (knee fits) |
| `grover_sweep_ext/`     | 4.6 MB | B1280-pool extension uncensoring the n=5,6 knees |
| `grover_sweep_nscan/`   | 30 MB  | random-3-regular N-scan of the capacity knee (N=120..960) |
| `grover_sweep_rr/`      | 22 MB  | random-regular family, N=120 (held-out size scan) |
| `grover_sweep_rr_N240/` | 11 MB  | same, N=240 |
| `grover_sweep_rr_N480/` | 8.6 MB | same, N=480 |
| `grover_sweep_rr_N960/` | 0.8 MB | same, N=960 (classical anchor only; statevector wall) |
| `tcoin_merged/`         | 74 MB  | 32-seed transformer-coin (neural) runs incl. capped control, difficulty D7, size N480, capacity B-scan |
| `expr4/`                | 4.2 GB | legacy REINFORCE baselines (one-shot classical/quantum), graph/QA pools used by every sweep |

Run directories follow
`<family>_<pool>_seed<s>_B<B>_.../<label>_s<s>_B<B>_..._{results.json,best_model.pt}`;
`results.json` stores the config, the exact train/valid QA split, and the
training history, so every split is recoverable from the archive alone.
