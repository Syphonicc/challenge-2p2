"""
Pre-checks C1 and C2 from PREREGISTRATION.md.

Method
------
Build the global POD basis exactly as the baselines do (20 modes, mean removed,
all 16 training cases concatenated).  Project each case onto it.

For each case, split every POD coefficient into

    a_k(t) = FORCED(t) + FREE(t)

where FORCED is a least-squares fit of the pitching harmonics at the KNOWN
pitching frequency,

    sum_{m=1..M} [ c_m cos(2 pi m f_P t) + s_m sin(2 pi m f_P t) ] + const

and FREE is the residual.  This matters because the record is only 80 time
units, so the FFT bin spacing is 1/80 = 0.0125 while f_N ~ 0.24 sits 0.01 away
from f_P = 0.25.  A spectrum cannot resolve them; a fit at the exactly known
f_P can.

C1: at alpha_0 = 25 deg the base flow is stable, so FREE should be small and
    the FORCED amplitude across the eight frequencies should trace a resonance
    curve peaking near f_N.  Its width gives the damping rate.

C2: at alpha_0 = 30 deg the base flow is unstable.  FREE carries self-sustained
    shedding; its amplitude and dominant frequency are the quantities of
    interest.

Caveat, stated up front: the pitching amplitude is fixed at 5 deg, but alphadot
scales with f_P, so the effective forcing is NOT constant across frequency.
Raw response amplitude and response normalised by f_P are both reported; which
is the right denominator depends on whether the flow responds to position or
rate, which is not known in advance.

Run:  python3 prechecks.py ../data
"""

import os
import sys
import numpy as np
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c22data import list_cases, load_case, load_grid, parse_case, to_state

N_MODES = 20
N_HARM = 12         # pitching harmonics to fit


def build_pod(data_dir, cache='../data/processed/pod20.npz'):
    """Global POD basis, matching the baselines' construction."""
    if os.path.exists(cache):
        z = np.load(cache)
        print(f'loaded cached POD basis from {cache}')
        return z['U'], z['mean'], z['s']

    path = f'{data_dir}/Challenge2_2_train.h5'
    cases = list_cases(path)
    print(f'assembling {len(cases)} cases...')
    blocks = []
    for a, fr in cases:
        d = load_case(path, a, fr, with_kinematics=False)
        blocks.append(to_state(d['ux'], d['uy']))
    Q = np.hstack(blocks)
    print(f'  data matrix {Q.shape}')

    qmean = Q.mean(axis=1)
    Q -= qmean[:, None]

    print('computing truncated SVD...')
    U, s, _ = svds(Q, k=N_MODES)
    order = np.argsort(s)[::-1]        # svds returns ascending
    U, s = U[:, order], s[order]

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, U=U, mean=qmean, s=s)
    print(f'  cached to {cache}')
    return U, qmean, s


def harmonic_split(a, t, f_P, n_harm=N_HARM):
    """Least-squares split of coefficient series into forced and free parts.

    a: (n_modes, nt).  Returns (forced, free), both (n_modes, nt).
    """
    dt_ = t[1] - t[0]
    m_max = min(n_harm, int(np.floor(0.5 / (dt_ * f_P))))
    cols = [np.ones_like(t)]
    for m in range(1, m_max + 1):
        cols.append(np.cos(2 * np.pi * m * f_P * t))
        cols.append(np.sin(2 * np.pi * m * f_P * t))
    D = np.column_stack(cols)                      # (nt, 1+2M)
    coef, *_ = np.linalg.lstsq(D, a.T, rcond=None)  # (1+2M, n_modes)
    forced = (D @ coef).T
    return forced, a - forced


def dominant_freq(x, dt):
    """Dominant frequency of a (possibly multi-channel) residual, by summed
    power spectrum.  Returns (f_peak, resolution)."""
    nt = x.shape[-1]
    X = np.fft.rfft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    P = (np.abs(X) ** 2).sum(axis=0)
    f = np.fft.rfftfreq(nt, d=dt)
    P[0] = 0.0
    return float(f[np.argmax(P)]), float(f[1])


def main(data_dir):
    x, y, dt, pitch_axis = load_grid(data_dir)
    U, qmean, s = build_pod(data_dir)

    path = f'{data_dir}/Challenge2_2_train.h5'
    cases = list_cases(path)

    print()
    print('=' * 74)
    print('FORCED / FREE SPLIT PER CASE')
    print('=' * 74)
    print('  A_forced: rms amplitude of the pitching-harmonic part')
    print('  A_free  : rms amplitude of the residual')
    print('  f_free  : dominant frequency of the residual')
    print()
    print(f'  {"case":>11} {"A_forced":>9} {"A_free":>9} {"free/tot":>9} '
          f'{"f_free":>7} {"f_free/f_P":>10}')

    rows = []
    for a, fr in cases:
        a0, fP = parse_case(a, fr)
        d = load_case(path, a, fr, with_kinematics=False)
        q = to_state(d['ux'], d['uy'])
        coeff = U.T @ (q - qmean[:, None])          # (n_modes, nt)
        nt = coeff.shape[1]
        t = np.arange(nt) * dt

        forced, free = harmonic_split(coeff, t, fP)
        A_f = float(np.sqrt((forced ** 2).sum(axis=0).mean()))
        A_r = float(np.sqrt((free ** 2).sum(axis=0).mean()))
        frac = A_r ** 2 / (A_f ** 2 + A_r ** 2)
        f_free, df = dominant_freq(free, dt)

        rows.append((a0, fP, A_f, A_r, frac, f_free))
        print(f'  {a+"/"+fr:>11} {A_f:>9.4f} {A_r:>9.4f} {frac:>9.3f} '
              f'{f_free:>7.3f} {f_free/fP:>10.2f}')

    print(f'\n  FFT resolution {df:.4f}; f_free is only meaningful to +/- that.')

    print()
    print('=' * 74)
    print('C1 -- RESONANCE CURVE AT alpha_0 = 25 (base flow stable)')
    print('=' * 74)
    r25 = [r for r in rows if r[0] == 25.0]
    print(f'  {"f_P":>6} {"A_forced":>9} {"A/f_P":>9} {"free frac":>10}')
    for a0, fP, A_f, A_r, frac, ff in r25:
        print(f'  {fP:>6.2f} {A_f:>9.4f} {A_f/fP:>9.4f} {frac:>10.3f}')

    f = np.array([r[1] for r in r25])
    A = np.array([r[2] for r in r25])
    for label, resp in (('raw', A), ('per unit f_P', A / f)):
        i = int(np.argmax(resp))
        print(f'\n  {label}: peak at f_P = {f[i]:.2f}, amplitude {resp[i]:.4f}')
        half = resp[i] / 2
        above = f[resp >= half]
        if len(above) > 1:
            w = above.max() - above.min()
            print(f'    points at or above half-max span {w:.3f} in f_P')
            print(f'    -> crude damping estimate |sigma| ~ {np.pi*w:.3f}')
        else:
            print('    resonance narrower than the frequency sampling; '
                  'width not measurable from 8 points')

    print()
    print('=' * 74)
    print('C2 -- FREE COMPONENT AT alpha_0 = 30 (base flow unstable)')
    print('=' * 74)
    r30 = [r for r in rows if r[0] == 30.0]
    print(f'  {"f_P":>6} {"A_free":>9} {"free frac":>10} {"f_free":>7}')
    for a0, fP, A_f, A_r, frac, ff in r30:
        print(f'  {fP:>6.2f} {A_r:>9.4f} {frac:>10.3f} {ff:>7.3f}')

    print('\n  Cases where the free fraction is near zero are locked: all the')
    print('  energy sits at pitching harmonics.  A large free fraction with')
    print('  f_free incommensurate with f_P means a second, independent clock.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../data')
