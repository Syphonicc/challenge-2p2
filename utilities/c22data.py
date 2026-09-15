"""
Loader for Challenge 2.2 HDF5 files.

Mirrors the conventions in the supplied MATLAB baselines so that a Python port
can be validated field-by-field against their output.

Axis order
----------
MATLAB h5read returns [ny nx nt] for /aXX/fYY/ux.  HDF5 stores in C order and
MATLAB reads in Fortran order, so the on-disk dataset shape is (nt, ny, nx) and
h5py returns exactly that.  We transpose to (ny, nx, nt) to match MATLAB.

Verified 2026-09-14 from flow physics; see fd_analysis.py.  check_layout() tests it against the grid file
and will tell you if it is wrong.

State vector
------------
The baselines build, per snapshot,

    q = [ reshape(ux, nx*ny, 1) ; reshape(uy, nx*ny, 1) ]

where MATLAB's reshape on an [ny nx] slice is column-major, i.e. y varies
fastest.  numpy must use order='F' to match.  Total state dimension 2*nx*ny.
"""

import h5py
import numpy as np


def list_cases(path):
    """Return [(aName, fName), ...] in the order MATLAB's fieldnames would give.

    MATLAB h5info returns groups in the order HDF5 reports them, which for
    these files is alphabetical.  h5py iterates alphabetically by default, so
    the orders agree -- but the evaluation script requires exact case-order
    agreement with the truth file, so this is worth checking once against
    their DMD output rather than assuming.
    """
    out = []
    with h5py.File(path, 'r') as f:
        for a in sorted(f.keys()):
            if not isinstance(f[a], h5py.Group):
                continue          # skips /t, /x, /y in results files
            for fr in sorted(f[a].keys()):
                if isinstance(f[a][fr], h5py.Group):
                    out.append((a, fr))
    return out


def parse_case(aName, fName):
    """'a25', 'f0p05' -> (25.0, 0.05).

    The baselines parse these as aName(2:end) and ['0.', fName(4:end)],
    i.e. f0p05 -> 0.05 and f0p35 -> 0.35.
    """
    alpha0 = float(aName[1:])
    f_P = float('0.' + fName[3:].replace('p', ''))
    return alpha0, f_P


def load_grid(data_dir):
    """Return (x, y, dt, pitch_axis)."""
    with h5py.File(f'{data_dir}/Challenge2_2_grid.h5', 'r') as f:
        x = np.asarray(f['x']).squeeze()
        y = np.asarray(f['y']).squeeze()
    with h5py.File(f'{data_dir}/Challenge2_2_parameters.h5', 'r') as f:
        dt = float(np.asarray(f['dt']).squeeze())
        pitch_axis = float(np.asarray(f['pitch_axis']).squeeze())
    return x, y, dt, pitch_axis


def load_case(path, aName, fName, with_kinematics=True):
    """Load one case.

    Returns dict with:
        ux, uy          (ny, nx, nt)
        alpha, alphadot (nt_kin,)  -- may be longer than nt for test input
    """
    out = {}
    with h5py.File(path, 'r') as f:
        g = f[aName][fName]
        for v in ('ux', 'uy'):
            arr = np.asarray(g[v])              # (nt, ny, nx) as stored
            out[v] = np.transpose(arr, (1, 2, 0))   # -> (ny, nx, nt)
        if with_kinematics:
            for v in ('alpha', 'alphadot'):
                if v in g:
                    out[v] = np.asarray(g[v]).squeeze()
    return out


def to_state(ux, uy):
    """Stack one case into the baselines' state matrix, shape (2*nx*ny, nt).

    MATLAB: [reshape(ux, nx*ny, nt); reshape(uy, nx*ny, nt)] on [ny nx nt]
    arrays, which is column-major with y fastest.
    """
    ny, nx, nt = ux.shape
    a = ux.reshape(nx * ny, nt, order='F')
    b = uy.reshape(nx * ny, nt, order='F')
    return np.vstack([a, b])


def from_state(q, ny, nx):
    """Inverse of to_state.  q is (2*nx*ny, nt) -> (ux, uy) each (ny, nx, nt)."""
    half = q.shape[0] // 2
    nt = q.shape[1]
    ux = q[:half, :].reshape(ny, nx, nt, order='F')
    uy = q[half:, :].reshape(ny, nx, nt, order='F')
    return ux, uy


def check_layout(data_dir, train_file='Challenge2_2_train.h5'):
    """Verify the inferred axis order against the grid.  Prints findings.

    Returns True if the layout looks consistent, False otherwise.  If this
    says the shapes do not match, STOP and fix load_case before going further:
    a wrong transpose runs silently and produces plausible nonsense.
    """
    x, y, dt, pitch_axis = load_grid(data_dir)
    nx, ny = len(x), len(y)
    path = f'{data_dir}/{train_file}'
    cases = list_cases(path)

    print(f'grid: nx={nx}, ny={ny}, dt={dt}, pitch_axis={pitch_axis}')
    print(f'cases found: {len(cases)}')

    with h5py.File(path, 'r') as f:
        a, fr = cases[0]
        raw = f[a][fr]['ux'].shape
    print(f'raw on-disk shape of /{a}/{fr}/ux: {raw}')

    ok = True
    if len(raw) != 3:
        print('  UNEXPECTED: not 3-D')
        return False

    if raw[1] == nx and raw[2] == ny:
        print(f'  nt={raw[0]} (axis order settled by fd_analysis.py, not here)')
    elif raw[1] == ny and raw[2] == nx:
        print(f'  looks like (nt, ny, nx) instead -- transpose in load_case '
              f'is WRONG, use (1, 2, 0)')
        ok = False
    else:
        print('  shape matches neither convention; inspect by hand')
        ok = False

    if nx == ny:
        print('  WARNING: nx == ny, so this check cannot distinguish the two '
              'orderings.  Verify against a field plot instead.')

    d = load_case(path, *cases[0])
    print(f'loaded ux shape (ny, nx, nt): {d["ux"].shape}')
    for k in ('alpha', 'alphadot'):
        if k in d:
            print(f'  {k}: shape {d[k].shape}, '
                  f'range [{d[k].min():.4f}, {d[k].max():.4f}]')

    q = to_state(d['ux'], d['uy'])
    ux2, uy2 = from_state(q, ny, nx)
    rt = max(np.abs(ux2 - d['ux']).max(), np.abs(uy2 - d['uy']).max())
    print(f'  state round-trip error: {rt:.3e}  ({"ok" if rt == 0 else "BAD"})')

    print('\ncases:')
    for a, fr in cases:
        a0, fp = parse_case(a, fr)
        print(f'  {a}/{fr} -> alpha_0={a0}, f_P={fp}')

    return ok and rt == 0
