import sys
import numpy as np
from strobe_test import simulate, is_locked, classify, strobe_points

TWO_PI = 2.0 * np.pi
F_N = 0.24
K = 0.5
HALF = K / TWO_PI

_cache = {}


def traj(f_P, n_snap, seed, n=1, m=1):
    key = (round(f_P, 6), n_snap, seed, n, m)
    if key not in _cache:
        _cache[key] = simulate(f_P, F_N, K, n=n, m=m, n_snap=n_snap,
                               noise=0.0, seed=seed, transient=80.0,
                               substeps=8)
    return _cache[key]


def run(f_P, n_snap=800, noise=0.0, seed=0, n=1, m=1, tol=0.15):
    ts, a1, a2, r_bar = traj(f_P, n_snap, seed, n, m)
    if noise > 0.0:
        rng = np.random.default_rng(10000 + seed)
        a1 = a1 + rng.normal(0, noise * r_bar, len(a1))
        a2 = a2 + rng.normal(0, noise * r_bar, len(a2))
    scale = float(np.sqrt(np.var(a1) + np.var(a2)))
    lab, npts = classify(ts, a1, a2, f_P, scale, tol=tol)
    return lab, npts, is_locked(f_P, F_N, K, n=n, m=m)


def expected(truth):
    return '1:1' if truth else 'quasi-periodic'


def section(name):
    print("=" * 72)
    print(name)
    print("=" * 72)


which = sys.argv[1]

if which == 'sweep':
    section("E2b. DETUNING SWEEP AFTER PERIODICITY GUARD (800 snaps, clean)")
    print(f"  {'f_P':>6} {'d/half':>8} {'truth':>6} {'classifier':>16} {'pts':>5} {'ok':>4}")
    for d in [-3.0, -1.5, -1.10, -1.02, -0.98, -0.90, -0.5, 0.0,
              0.5, 0.90, 0.98, 1.02, 1.10, 1.5, 3.0]:
        f_P = F_N + d * HALF
        if f_P <= 0.02:
            continue
        lab, npts, truth = run(f_P, n_snap=800)
        ok = 'yes' if lab == expected(truth) else ('--' if lab == 'undecidable' else 'NO')
        print(f"  {f_P:>6.3f} {d:>8.2f} {('LOCK' if truth else 'free'):>6}"
              f" {lab:>16} {npts:>5} {ok:>4}")

elif which == 'noise':
    section("E4b. EXACT-LABEL ACCURACY vs NOISE (800 snaps, 10 seeds)")
    cases = [("deep in band", 0.30), ("near edge in", 0.90),
             ("just outside", 1.10), ("far outside ", 3.00)]
    print(f"  {'case':>13} {'noise':>7} {'accuracy':>9}  commonest wrong label")
    for name, d in cases:
        f_P = F_N + d * HALF
        for noise in [0.0, 0.02, 0.05, 0.10, 0.20]:
            hits, wrong = 0, {}
            for s in range(10):
                lab, npts, truth = run(f_P, n_snap=800, noise=noise, seed=s)
                if lab == expected(truth):
                    hits += 1
                else:
                    wrong[lab] = wrong.get(lab, 0) + 1
            top = max(wrong, key=wrong.get) if wrong else '-'
            print(f"  {name:>13} {noise:>7.2f} {hits/10:>9.2f}  {top}")

elif which == 'length':
    section("E5. ACCURACY vs RECORD LENGTH (10 seeds, 2% noise)")
    f_in = F_N + 0.5 * HALF
    f_out = F_N + 2.0 * HALF
    print(f"  {'n_snap':>8} {'strobe pts':>11} {'acc in-band':>12} {'acc out':>9}")
    for n_snap in [70, 140, 200, 400, 800, 1600]:
        accs = []
        npts_seen = 0
        for f_P in (f_in, f_out):
            hits = 0
            for s in range(10):
                lab, npts, truth = run(f_P, n_snap=n_snap, noise=0.02, seed=s)
                npts_seen = npts
                if lab == expected(truth):
                    hits += 1
            accs.append(hits / 10)
        print(f"  {n_snap:>8} {npts_seen:>11} {accs[0]:>12.2f} {accs[1]:>9.2f}")

elif which == 'train':
    section("E6. EACH TRAINING FREQUENCY AT 800 SNAPSHOTS, 2% NOISE")
    print(f"  {'f_P':>6} {'pts':>5} {'truth':>6} {'classifier':>16} {'ok':>4}")
    for f_P in [0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        lab, npts, truth = run(f_P, n_snap=800, noise=0.02)
        ok = 'yes' if lab == expected(truth) else ('--' if lab == 'undecidable' else 'NO')
        print(f"  {f_P:>6.2f} {npts:>5} {('LOCK' if truth else 'free'):>6}"
              f" {lab:>16} {ok:>4}")

elif which == 'sanity':
    section("E0b. SANITY AFTER FIX (4000 snapshots, clean)")
    for f_P in [0.24, 0.30, 0.40]:
        lab, npts, truth = run(f_P, n_snap=4000)
        print(f"  1:1  f_P={f_P:.2f} truth={'LOCK' if truth else 'free':>4}"
              f"  -> {lab:<16} pts={npts}")
    for f_P in [0.48, 0.52, 0.70]:
        lab, npts, truth = run(f_P, n_snap=4000, n=1, m=2)
        print(f"  1:2  f_P={f_P:.2f} truth={'LOCK' if truth else 'free':>4}"
              f"  -> {lab:<16} pts={npts}")
