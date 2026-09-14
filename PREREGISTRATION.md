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
