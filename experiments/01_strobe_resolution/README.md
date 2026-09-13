# 01 — Can stroboscopic sampling classify lock-in regime at challenge record lengths?

Run before any challenge data was downloaded. Synthetic only.

## Question

Phase A of the plan classifies every training case as forced / 1:1 / 1:k /
quasi-periodic by stroboscopically sampling latent coefficients at fixed pitch
phase. That regime map is what chooses the forecaster architecture. Does the
method actually work at the record lengths available?

    training case: 800 snapshots, dt = 0.1  ->  T = 80 time units
    test history:   70 snapshots, dt = 0.1  ->  T =  7 time units

## Setup

Forced Stuart-Landau amplitude equation plus an Adler phase equation at an
n:m resonance, so ground truth is analytic:

    dr/dt     = sigma*r - ell*r^3
    dtheta/dt = omega + K*sin(n*phi_p - m*theta)

locked at order n:m iff |m*omega - n*Omega_p| <= m*K.

Observable is two POD-like coefficients: a forced part slaved to the known
pitch phase plus a free shedding part. f_N = 0.24 (the 30 deg value). K = 0.5
throughout, giving a 1:1 tongue half-width of 0.08 in f_P.

## Results

1. The naive version has a specific bug. Quasi-periodic cases just outside the
   tongue get labelled 1:3 or 1:4, because slowly drifting strobe points sit on
   a short arc that k-means chops into tight clusters. Fixed with a periodicity
   guard: genuine 1:k locking means the strobe sequence revisits the same
   cluster every k samples; a drifting arc does not.

2. After the fix, 13/15 correct across the detuning sweep. Everything within
   +/- 1 half-width of the band edge is right. The two residual failures are
   both `free` labelled 1:4, at moderate detuning with 12 and 28 strobe points
   -- small-sample false positives, not a dynamical effect.

3. Noise is not the binding constraint. Exact-label accuracy holds at 1.00 up
   to 10% additive noise on the latent coefficients, dropping to 0.90 in one
   cell at 20%.

4. The binding constraint is strobe-point count, roughly 15. Points ~ T*f_P,
   so at 800 snapshots f_P = 0.05 gives 3 points (undecidable) and f_P = 0.10
   gives 7 (defaults to quasi-periodic, which happens to be right -- a
   coincidence, not a detection). f_P >= 0.20 is fine.

5. On a 70-snapshot history the classifier returns `undecidable` in 10/10 runs
   in-band. It abstains rather than mislabelling, but the regime cannot be
   determined from a test case's history by this method, and the point-count
   argument suggests not by any strobe-based method.

## Consequences for the plan

- The Phase A regime map covers 6 of 8 training frequencies. f_P = 0.05 and
  0.10 are known gaps. Both are far below f_N so the physical prior that they
  are unlocked is strong, but it will not have been measured.

- Sec. 7 of the handover claims strobing "works on short records where a raw
  spectrum will not". True relative to spectra, but short has a floor of about
  15 pitch periods, not 800 snapshots. Restate.

- Any architecture that classifies regime at forecast time and switches
  behaviour accordingly is ruled out. This strengthens the oscillator model,
  which does not classify: it learns sigma, omega, K as functions of
  (alpha_0, f_P) and the tongue condition falls out. Regime becomes
  extrapolation in parameter space, not inference from 70 snapshots.

## Prediction vs outcome

Predicted beforehand: 800 points would separate regimes fine except close to
the band edge; 70 points would be marginal for 1:1 and fail for 1:2.

Both wrong. Near-edge performance at 800 was perfect. The real limit was low
f_P -- record length in pitch periods -- which had been framed as a detuning
problem. And 70 snapshots does not degrade gracefully; it abstains outright.

## What this does not establish

Clean two-oscillator synthetic, one tongue width, white additive noise where
real POD coefficient error is correlated in time, and a single oscillator where
near-body structure may need more. This can show the method broken; it cannot
fully show it sound.

## Running

    python3 run_fast.py sanity|sweep|train|length|noise

numpy only. `length` and `noise` take a couple of minutes; the rest are fast.
