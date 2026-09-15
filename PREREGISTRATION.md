# Pre-registration — Challenge 2.2

Written before any challenge data was downloaded. The git timestamp on this
file is the claim: every prediction below was recorded before the data was
seen.

This file is not edited after data arrives. Corrections, outcomes and
retractions go in a dated appendix at the bottom, so the original text stays
readable and wrong predictions stay visible.

---

## 0. Status at time of writing

- Challenge data: not downloaded. Deep Blue deposit not accessed.
- Challenge website: not yet read.
- Crop indices, results-file format, scoring details: unknown.
- Only prior work consulted: the challenge paper (arXiv:2601.06183) and
  earlier work of my own on rollout phase drift (arXiv:2608.07189).
- One synthetic experiment has been run (experiments/01_strobe_resolution),
  on a hand-built oscillator model, with no challenge data involved.

## 1. Central hypothesis

The out-of-sample failure reported for both baselines at
(alpha_0 = 30 deg, f_P = 0.27) is an **amplitude** failure, not a phase
failure. Neither baseline has a preferred oscillation amplitude: a linear
model grows or decays, and an MSE-trained rollout hedges toward the mean.
The real flow sits on a limit cycle with a preferred amplitude.

A model with a Stuart-Landau amplitude equation and an Adler phase equation,
with coefficients learned as functions of (alpha_0, f_P), should therefore
beat both baselines specifically on out-of-sample locked cases, and the
margin should be largest where the baselines' amplitude error is largest.

## 2. Predictions

Numbered so outcomes can be recorded against them.

**P1.** An autoregressive model that has learned the locking will show
tangential (timing) rollout error that **saturates** inside the lock-in band,
rather than growing. In my published cylinder work the growth was 11-17x.
Saturating-versus-growing is the signature, not the absolute value.

**P2.** Tangential error growth despite in-band forcing means the model failed
to learn the locking dynamics. NMSE alone cannot distinguish this case from a
model that learned it.

**P3.** Tangential error growth rate scales inversely with distance from the
lock-in band edge, and goes to zero at the band edge (saddle-node).

**P4.** Out-of-sample, a learned model will place the band boundary wrongly,
producing qualitatively wrong relative-phase dynamics for test cases near the
edge.

**P5.** At (30 deg, 0.27) specifically, the dominant error component is
**transverse** (amplitude), not tangential (phase). DMDc keeps approximately
correct phase while its fluctuations grow. **If the decomposition says
otherwise, the hypothesis in Sec. 1 is wrong and should be abandoned rather
than rescued.**

**P6** (from experiments/01_strobe_resolution, synthetic). The lock-in regime
of a test case **cannot** be determined from its 70-snapshot history by
stroboscopic sampling: that gives 0-4 strobe points against a requirement of
roughly 15. Any architecture that classifies regime at forecast time and
switches behaviour is therefore ruled out. Regime must be predicted
structurally from (alpha_0, f_P).

**P7** (mine, low confidence, offered by Claude and adopted after checking the
argument). POD beats a CNN-AE on reconstruction error at alpha_0 = 25 deg at
matched latent sizes (4, 8, 12, 20). At 30 deg the gap narrows or reverses for
small latents.

## 3. Falsifiable pre-checks, available from training data alone

Run these before trusting the oscillator model.

**C1.** At 25 deg, forced response amplitude across the eight training
frequencies should trace a resonance curve peaking near f_N. Its width gives
the damping rate.

**C2.** At 30 deg, natural shedding amplitude gives the growth rate.

**C3.** Interpolating C1 and C2 yields a critical angle alpha_c. The challenge
paper states the base flow is stable at 25 deg and unstable above 27 deg, so
alpha_c should land near 27 deg. **If it lands outside roughly 26-28 deg, the
cubic normal form is too crude here and the framing in Sec. 1 is in trouble.**

## 4. Abandonment conditions

Stated in advance so they are not negotiated away later.

- P5 fails: the (30 deg, 0.27) error is predominantly tangential.
- C3 fails: interpolated alpha_c falls outside 26-28 deg.
- The regime map shows most training cases are quasi-periodic rather than
  locked. The attractor is then a 2-torus, the tangential/transverse
  decomposition is confounded by a second phase direction, and both the
  architecture and the diagnostic need rethinking.
- Chaotic dynamics near the band edge.

Any of these means reporting the negative result, not reframing the
hypothesis to survive it. The challenge organisers explicitly encourage
negative results and every completed challenge yields a journal article, so
there is no incentive to rescue a dead hypothesis.

## 5. Analysis decisions fixed in advance

- **Ten seeds minimum**, not three. A Spearman of exactly -1.00 with stable
  leave-one-out at three seeds became -0.50 and non-monotone at ten.
- **Nulls measured, not assumed.** The isotropic null 1/n was wrong by
  4.6-33x for smooth error fields and correcting it reversed a conclusion.
  Measure the null **per parameter point**, not once for the system: different
  (alpha_0, f_P) give different orbit shapes.
- **kappa is confounded by trajectory geometry** (Spearman +0.90 with
  trajectory straightness across systems). Within-system comparisons only.
  Enrichment ratio against its own measured null is preferred.
- **Horizon is fixed at 130 steps** by the challenge. Measured usable horizon
  appears only in supporting analysis.
- **NMSE split** into projection error and temporal error, otherwise the
  out-of-sample comparison confounds basis failure with forecaster failure.
- The error decomposition appears in the submission as supporting analysis
  alongside NMSE, not as a replacement metric.

## 6. Success criterion

A model that is worse than the challenge LSTM baseline means the diagnostic is
attached to a bad model. **Matching or beating the baseline comes first**; the
decomposition is the second half of the paper, not the first.

## 7. Known unverified claims

Flagged so they are not cited until checked.

- The tangential share in arXiv:2608.07189 is recollected as 97-98%, but an
  earlier draft said 97-99%. **Verify against 07_audit/audit_numbers.py
  output before quoting.**
- An earlier note claimed prior work reports faster phase-error accumulation
  under out-of-distribution generalisation. No citation was ever found.
  **Find the source or drop the claim.**
- The training frequency set is treated as
  {0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5}. The challenge paper's prose
  lists seven but says eight; 0.35 appears in Fig. 9b and the Deep Blue file
  list. **Confirm with the point of contact.**

---

## Appendix — outcomes

(empty at time of writing)

**2026-09-14.** Timing note. This file was committed at 22:58, after the
challenge data files landed on disk at 22:27-22:47 the same evening. The data
had not been opened, read or inspected at the time of writing -- no HDF5 file
was accessed and no field was plotted -- but the git timestamp does not
demonstrate that, and the header claim above overstates what the record shows.
Recorded here rather than corrected in place.

**2026-09-15. C1: passed.** Global POD basis (20 modes, matching the
baselines), each case's coefficients split by least squares into pitching
harmonics at the known f_P and a residual. At alpha_0 = 25 the residual is
~0.007 against forced amplitudes of 2-7, i.e. numerically zero: the base flow
is stable and the response is purely forced, as the challenge paper states.
The forced amplitude across the eight frequencies traces a resonance curve
peaking at f_P = 0.25 (amplitude 7.18, falling to 1.97 at f_P = 0.50). Crude
damping estimate |sigma_25| ~ 0.31 from the half-max width, limited by having
only eight sample frequencies.

**2026-09-15. f_N measured independently.** At alpha_0 = 30 the five unlocked
cases all carry their free component at f_free = 0.225-0.250, regardless of
f_P, against an FFT resolution of 0.0125. This is f_N measured five times
independently, and it agrees with the C1 resonance peak. The two measurements
are unrelated, which is the substance of the result.

**2026-09-15. Regime map, from data rather than assumption.** At alpha_0 = 30:
f_P = 0.20, 0.25, 0.30 are fully locked (free fraction 0.000, residual three
orders of magnitude below the forced part); f_P = 0.05, 0.10, 0.35, 0.40, 0.50
are quasi-periodic (free fraction 0.43-0.87). All eight alpha_0 = 25 cases are
purely forced. So the lock-in band at 30 deg contains 0.20 and 0.30 and
excludes 0.10 and 0.35, centred near 0.24. The paper's locked cases at 0.25 and
0.27 both sit inside it, and {30 deg, 0.27} -- where both baselines fail -- is
therefore in-band.

Abandonment condition 3 (most cases quasi-periodic, attractor a 2-torus) did
NOT fire: 11 of 16 cases are locked or purely forced. But the 5 quasi-periodic
cases are genuine tori, and the tangential/transverse decomposition is
confounded there. It is valid on the other 11 and needs separate treatment on
those 5.

**2026-09-15. C3 is underdetermined as written -- a defect in the check, not a
result.** Obtaining alpha_c requires sigma at both angles. C1 gives sigma_25
from the resonance width. But the Stuart-Landau saturated amplitude at 30 deg
is r^2 = sigma_30 / ell, so C2 yields only the RATIO sigma_30 / ell. Without an
independent estimate of the cubic coefficient ell, alpha_c cannot be computed.
C3 as pre-registered cannot be run. This was not spotted when the file was
written. Any replacement check must be stated before it is run, in a new dated
entry, not by editing C3 above.

**2026-09-15. Method note.** Experiment 01 validated stroboscopic regime
classification on synthetic data. The method actually used here is a
least-squares fit of pitching harmonics at the known f_P, which resolves cases
strobing could not (f_P = 0.05 gives only 3 strobe points). Experiment 01 is
what established that record length was the binding constraint, which motivated
the fit-at-known-frequency approach, but the method it validated is not the one
in use.
