"""
Two jobs:

(1) Settle the axis ambiguity.  nx == ny == 60, so shape alone cannot tell
    whether the on-disk order is (nt, nx, ny) or (nt, ny, nx).  Physics can:
    the plate pitches about x = pitch_axis at y = 0, and the wake extends
    DOWNSTREAM in +x.  So the time-variance of the velocity field should be
    strongly asymmetric along the streamwise axis (peak near the plate, long
    tail downstream) and roughly symmetric about y = 0 across it.

(2) Compute the F/D analysis.  The baselines normalise by distance from a
    single GLOBAL training mean, not from each case's own mean.  So a forecast
    that outputs a case's own time-mean field, constant in time, already
    scores below 1.  This measures how far below.

Run:  python3 fd_analysis.py ../data
"""

import sys
import numpy as np
import h5py

sys.path.insert(0, '.')
from c22data import list_cases, load_case, load_grid, parse_case, to_state


def variance_field(data_dir, path, aName, fName):
    """Time-variance of the velocity magnitude, as stored (no transpose)."""
    with h5py.File(path, 'r') as f:
        ux = np.asarray(f[aName][fName]['ux'])   # (nt, d1, d2) as stored
        uy = np.asarray(f[aName][fName]['uy'])
    return ux.var(axis=0) + uy.var(axis=0)       # (d1, d2)


def settle_axes(data_dir):
    x, y, dt, pitch_axis = load_grid(data_dir)
    path = f'{data_dir}/Challenge2_2_train.h5'
    a, fr = 'a30', 'f0p25'

    print('=' * 68)
    print('AXIS CHECK')
    print('=' * 68)
    print(f'x range: [{x.min():.3f}, {x.max():.3f}]   '
          f'y range: [{y.min():.3f}, {y.max():.3f}]')
    print(f'pitch axis at x = {pitch_axis}')

    v = variance_field(data_dir, path, a, fr)
    print(f'\nvariance field shape as stored: {v.shape}')

    # Marginal profiles along each stored axis
    m0 = v.mean(axis=1)   # profile along stored axis 0
    m1 = v.mean(axis=0)   # profile along stored axis 1

    i0 = int(np.argmax(m0))
    i1 = int(np.argmax(m1))

    print(f'\nIf stored axis 0 is x (streamwise):')
    print(f'  peak variance at x = {x[i0]:.3f}, y = {y[i1]:.3f}')
    print(f'If stored axis 0 is y (cross-stream):')
    print(f'  peak variance at y = {y[i0]:.3f}, x = {x[i1]:.3f}')

    # Symmetry test: the cross-stream profile should be far more symmetric
    # about y = 0 than the streamwise profile is about its own centre.
    def asym(prof, coord):
        c = np.interp(0.0, coord, np.arange(len(coord))) if \
            (coord.min() < 0 < coord.max()) else len(coord) / 2
        n = int(min(c, len(prof) - c))
        if n < 3:
            return np.nan
        lo = prof[int(c) - n:int(c)]
        hi = prof[int(c):int(c) + n][::-1]
        return float(np.abs(lo - hi).sum() / (lo.sum() + hi.sum()))

    a0_as_y = asym(m0, y)
    a1_as_y = asym(m1, y)
    print(f'\nasymmetry about y=0 (lower is more symmetric):')
    print(f'  stored axis 0 treated as y: {a0_as_y:.4f}')
    print(f'  stored axis 1 treated as y: {a1_as_y:.4f}')

    if np.isnan(a0_as_y) or np.isnan(a1_as_y):
        print('  inconclusive')
    elif a1_as_y < a0_as_y:
        print('  => stored axis 1 is CROSS-STREAM (y), axis 0 is x')
        print('     on-disk order is (nt, nx, ny); c22data transpose (2,1,0) '
              'is CORRECT')
    else:
        print('  => stored axis 0 is CROSS-STREAM (y), axis 1 is x')
        print('     on-disk order is (nt, ny, nx); c22data transpose is WRONG, '
              'use (1, 2, 0)')

    # Downstream tail: streamwise profile should have most of its energy
    # downstream of the pitch axis.
    print(f'\nstreamwise energy split about x = {pitch_axis} '
          f'(expect most downstream):')
    ix = int(np.searchsorted(x, pitch_axis))
    for name, prof in (('axis 0', m0), ('axis 1', m1)):
        up, dn = prof[:ix].sum(), prof[ix:].sum()
        print(f'  {name} as x: upstream {up/(up+dn):.3f}, '
              f'downstream {dn/(up+dn):.3f}')


def fd_analysis(data_dir):
    path = f'{data_dir}/Challenge2_2_train.h5'
    cases = list_cases(path)

    print()
    print('=' * 68)
    print('F/D ANALYSIS')
    print('=' * 68)
    print('Pass 1: global training mean over all cases and snapshots')

    acc = None
    n = 0
    per_case_mean = {}
    for a, fr in cases:
        d = load_case(path, a, fr, with_kinematics=False)
        q = to_state(d['ux'], d['uy'])
        per_case_mean[(a, fr)] = q.mean(axis=1)
        s = q.sum(axis=1)
        acc = s if acc is None else acc + s
        n += q.shape[1]
    gmean = acc / n
    print(f'  {n} snapshots, state dim {gmean.size}')

    print('\nPass 2: per-case scores for a constant "own mean" forecast')
    print('  F = own fluctuation energy, D = |own mean - global mean|^2')
    print(f'\n  {"case":>12} {"F":>10} {"D":>10} {"D/F":>7} {"score":>7}')

    rows = []
    for a, fr in cases:
        d = load_case(path, a, fr, with_kinematics=False)
        q = to_state(d['ux'], d['uy'])
        cmean = per_case_mean[(a, fr)]

        # their formula, per timestep: spatial mean of squared error
        # over spatial mean of squared distance from GLOBAL mean.
        # Prediction = own mean, constant in time.
        num = ((q - cmean[:, None]) ** 2).mean(axis=0)
        den = ((q - gmean[:, None]) ** 2).mean(axis=0)
        score = float((num / den).mean())

        F = float(((q - cmean[:, None]) ** 2).mean())
        D = float(((cmean - gmean) ** 2).mean())
        a0, fp = parse_case(a, fr)
        rows.append((a0, fp, F, D, score))
        print(f'  {a+"/"+fr:>12} {F:>10.5f} {D:>10.5f} '
              f'{D/F:>7.2f} {score:>7.3f}')

    print('\n  score = NMSE of predicting the case mean and nothing else.')
    print('  Their reported baseline errors must be read against THIS, not 1.')

    sc = np.array([r[4] for r in rows])
    print(f'\n  mean {sc.mean():.3f}, min {sc.min():.3f}, max {sc.max():.3f}')

    for a0 in (25.0, 30.0):
        s = np.array([r[4] for r in rows if r[0] == a0])
        print(f'  alpha_0 = {a0:.0f}: mean {s.mean():.3f}')


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data'
    settle_axes(data_dir)
    fd_analysis(data_dir)
