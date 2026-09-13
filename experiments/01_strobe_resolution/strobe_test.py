"""
Cheap pre-Gate-0 test for AIAA Challenge 2.2.

Question: can stroboscopic (Poincare) sampling of latent coefficients classify
lock-in regime (1:1 locked / 1:2 locked / quasi-periodic) at the record lengths
the challenge actually gives us?

  - training case: 800 snapshots, dt = 0.1  -> T = 80 time units
  - test history:   70 snapshots, dt = 0.1  -> T =  7 time units

Ground truth is known because we build the system: a Stuart-Landau amplitude
equation with an Adler phase equation at an n:m resonance.

    dr/dt     = sigma*r - ell*r^3
    dtheta/dt = omega + K*sin(n*phi_p - m*theta)

with phi_p = 2*pi*f_P*t the (known) pitching phase.

Let psi = m*theta - n*phi_p.  Then
    dpsi/dt = (m*omega - n*Omega_p) - m*K*sin(psi)
so the system is phase-locked at order n:m iff

    |m*omega - n*Omega_p| <= m*K            (Arnold tongue condition)

That gives an exact analytic label to compare the classifier against.

Observable: two "POD-like" coefficients carrying a forced part (slaved to the
known pitch phase) plus a free shedding part:

    a1 = A_f*cos(phi_p) + r*cos(theta) + noise
    a2 = A_f*sin(phi_p) + r*sin(theta) + noise

At strobe times (phi_p = 0 mod 2*pi) the forced part is constant, so strobe
points sit on a circle of radius r centred at (A_f, 0):
    1:1 locked        -> 1 cluster
    1:k locked        -> k clusters
    quasi-periodic    -> points spread around the circle
"""

import numpy as np

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------
def simulate(f_P, f_N, K, n=1, m=1, sigma=1.0, ell=1.0, A_f=1.0,
             n_snap=800, dt=0.1, noise=0.0, transient=200.0,
             substeps=20, seed=0):
    """Integrate the forced oscillator, return (t, a1, a2, r_bar)."""
    rng = np.random.default_rng(seed)
    omega = TWO_PI * f_N
    Omega_p = TWO_PI * f_P

    h = dt / substeps

    def deriv(state, t):
        r, th = state
        phi_p = Omega_p * t
        dr = sigma * r - ell * r ** 3
        dth = omega + K * np.sin(n * phi_p - m * th)
        return np.array([dr, dth])

    # start on the limit cycle, random initial phase
    state = np.array([np.sqrt(sigma / ell), rng.uniform(0, TWO_PI)])
    t = -transient
    while t < 0.0:                       # burn off transients
        k1 = deriv(state, t)
        k2 = deriv(state + 0.5 * h * k1, t + 0.5 * h)
        k3 = deriv(state + 0.5 * h * k2, t + 0.5 * h)
        k4 = deriv(state + h * k3, t + h)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h

    ts = np.empty(n_snap)
    rs = np.empty(n_snap)
    ths = np.empty(n_snap)
    for i in range(n_snap):
        ts[i], rs[i], ths[i] = t, state[0], state[1]
        for _ in range(substeps):
            k1 = deriv(state, t)
            k2 = deriv(state + 0.5 * h * k1, t + 0.5 * h)
            k3 = deriv(state + 0.5 * h * k2, t + 0.5 * h)
            k4 = deriv(state + h * k3, t + h)
            state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h

    phi_p = Omega_p * ts
    a1 = A_f * np.cos(phi_p) + rs * np.cos(ths)
    a2 = A_f * np.sin(phi_p) + rs * np.sin(ths)
    r_bar = float(np.mean(rs))
    if noise > 0.0:
        scale = noise * r_bar
        a1 = a1 + rng.normal(0.0, scale, n_snap)
        a2 = a2 + rng.normal(0.0, scale, n_snap)
    return ts, a1, a2, r_bar


def is_locked(f_P, f_N, K, n=1, m=1):
    """Analytic ground truth: inside the n:m Arnold tongue?"""
    return abs(m * TWO_PI * f_N - n * TWO_PI * f_P) <= m * K


# ----------------------------------------------------------------------
# stroboscopic classifier
# ----------------------------------------------------------------------
def strobe_points(ts, a1, a2, f_P):
    """Sample (a1, a2) at pitch phase 0 (mod 2pi), linear interpolation."""
    T_p = 1.0 / f_P
    t0, t1 = ts[0], ts[-1]
    k0 = int(np.ceil(t0 / T_p))
    k1 = int(np.floor(t1 / T_p))
    t_strobe = np.arange(k0, k1 + 1) * T_p
    if t_strobe.size == 0:
        return np.empty((0, 2))
    p1 = np.interp(t_strobe, ts, a1)
    p2 = np.interp(t_strobe, ts, a2)
    return np.column_stack([p1, p2])


def _kmeans(pts, k, n_init=8, seed=0):
    """Tiny k-means. Returns (labels, within-cluster RMS radius)."""
    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    for _ in range(n_init):
        idx = rng.choice(len(pts), size=k, replace=False)
        cent = pts[idx].copy()
        labels = np.zeros(len(pts), dtype=int)
        for _ in range(50):
            d = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                sel = labels == j
                if sel.any():
                    cent[j] = pts[sel].mean(0)
        d = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        rms = float(np.sqrt(d.min(1).mean()))
        if rms < best[1]:
            best = (labels, rms)
    return best


def classify(ts, a1, a2, f_P, r_scale, tol=0.15, k_max=4, min_pts_per_cluster=3):
    """
    Return one of: '1:1', '1:k' (k>=2), 'quasi-periodic', 'undecidable'.

    tol: within-cluster RMS radius, as a fraction of the shedding amplitude,
         below which a set of points counts as a single cluster.
    """
    pts = strobe_points(ts, a1, a2, f_P)
    n_pts = len(pts)
    if n_pts < 2 * min_pts_per_cluster:
        return 'undecidable', n_pts

    for k in range(1, k_max + 1):
        if n_pts < k * min_pts_per_cluster:
            break
        if k == 1:
            rms = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
            if rms < tol * r_scale:
                return '1:1', n_pts
            continue
        labels, rms = _kmeans(pts, k)
        if rms >= tol * r_scale:
            continue
        # PERIODICITY GUARD.  Genuine 1:k locking means the strobe sequence
        # revisits the same cluster every k samples.  A quasi-periodic orbit
        # that is drifting slowly just outside the tongue also produces tight
        # clusters (the points sit on a short arc), but does NOT repeat with
        # period k.  Without this check, near-edge unlocked cases get labelled
        # 1:3 / 1:4.
        if len(labels) <= k:
            continue
        same = float(np.mean(labels[:-k] == labels[k:]))
        if same > 0.9:
            return f'1:{k}', n_pts
    return 'quasi-periodic', n_pts
