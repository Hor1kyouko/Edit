# EXP-002 — Seed Robustness of Centered Non-Additivity

## [KEY FINDING]

On the fixed V03 image and exact EXP-001 prompt set, centered
reference-corrected non-additivity reproduced across all five AB carrier seeds.
All five Q1 curves were non-monotonic and reached their maximum at the same
actual schedule point, sigma `0.490105301`. This is evidence of seed robustness
for this specific image, prompt set, carrier family, and implementation. It is
not evidence of cross-image generality or an object-interaction mechanism.

## [WHAT IT MEANS]

The non-zero EXP-001 residual was not an isolated seed-42 event. Its normalized
magnitude and temporal shape were comparatively consistent across seeds
42--46, especially after the first probe. The mismatch between `dAB` and
`dA+dB` is mixed: the additive prediction is too large in norm, while its
direction also differs non-trivially from `dAB`.

## [IMPACT ON CURRENT HYPOTHESIS]

The narrow hypothesis that direct affine additivity fails reproducibly across
AB carrier seeds on this test case is strengthened. Any stronger hypothesis
about semantic interaction, object structure, spatial localization, causal
usefulness, or generalization remains unsupported. In particular, the
behavioral failure of condition B prevents a clean object-level interpretation.

## [SEED ROBUSTNESS]

The six target sigmas mapped without duplicates to actual sigmas
`1.000000`, `0.896748`, `0.759511`, `0.490105`, `0.259758`, and `0.098208`.

| Actual sigma | Q1 mean ± population std | Q1 range | Q2 mean ± std | cos(`dAB`,`dA+dB`) | `||dA+dB||/||dAB||` |
|---:|---:|---:|---:|---:|---:|
| 1.000000 | 0.551940 ± 0.085338 | 0.458008–0.661815 | 0.325184 ± 0.033695 | 0.919830 ± 0.010442 | 1.303658 ± 0.097282 |
| 0.896748 | 0.989122 ± 0.072711 | 0.917823–1.104544 | 0.508918 ± 0.020797 | 0.906387 ± 0.011976 | 1.800742 ± 0.069056 |
| 0.759511 | 1.171744 ± 0.048813 | 1.081527–1.216583 | 0.564335 ± 0.012322 | 0.910502 ± 0.006915 | 2.006895 ± 0.044882 |
| 0.490105 | 1.335825 ± 0.067504 | 1.223543–1.424660 | 0.608819 ± 0.015660 | 0.887890 ± 0.014000 | 2.142118 ± 0.060695 |
| 0.259758 | 1.020520 ± 0.035235 | 0.971528–1.062977 | 0.536999 ± 0.010256 | 0.904885 ± 0.003724 | 1.832392 ± 0.032388 |
| 0.098208 | 0.905409 ± 0.006027 | 0.897939–0.912775 | 0.507770 ± 0.001531 | 0.907898 ± 0.000782 | 1.710414 ± 0.007157 |

The Q1 coefficient of variation falls from about 15.5% at sigma 1.0 to below
1% at sigma 0.0982. All five seeds show the same rise to a mid-trajectory peak
followed by a decline; the registered trend classifier labels all five curves
`non_monotonic`. The absolute `||r_ref||` mean is largest at sigma 1.0
(`255.64 ± 19.58`) even though normalized Q1 peaks at sigma 0.4901. This
demonstrates why the normalized measure changes the interpretation relative to
raw residual magnitude.

## [DIRECTION VS MAGNITUDE]

Directional agreement remains positive and fairly high (`0.888–0.920`), but it
is not collinear. Meanwhile, the additive sum is `1.30–2.14` times the norm of
`dAB`, with the largest scale excess at the Q1 peak. The evidence therefore
supports a mixed mismatch that is predominantly scale-related through the
middle and late probes, with a persistent directional component. It does not
support calling the difference an interaction or semantic field.

## [MANIPULATION CHECK]

Manual inspection of the five labeled contact sheets produced:

| Condition | PASS | PARTIAL | FAIL | Main observation |
|---|---:|---:|---:|---|
| 0 | 5 | 0 | 0 | Preservation reference materially preserves the scene. |
| A | 5 | 0 | 0 | Mug becomes white and the beige vase is retained. |
| B | 0 | 4 | 1 | Blue systematically leaks from the vase edit into the mug. |
| AB | 3 | 2 | 0 | Both edits usually occur; two seeds produce a cyan/blue-gray rather than clearly white mug. |

Thus the prompts cause meaningful behavioral changes, but condition B is not a
clean isolated edit. Seed 43 is the strongest failure: its B output turns the
mug saturated blue, and its AB output makes the mug pale cyan. This limitation
does not invalidate the algebraic velocity measurements, but it blocks a clean
semantic claim that the measured residual reflects interaction between two
successfully isolated object edits.

## [TECHNICAL VALIDITY]

- The inference process exited successfully and all five seed summaries report
  `complete`.
- All 30 probes are valid and unblocked.
- Every shared-latent per-call maximum difference is exactly zero.
- All raw and effective order-1/order-2 differences are exactly zero for every
  condition and probe.
- All prediction tensors are finite; shape and non-text preparation invariants
  pass.
- Raw conditional and effective velocities are identical by construction in
  this embedded-guidance path; the same tensor is passed to Euler.
- The formal seed-42 rerun exactly matches the development run in schedule,
  latent hashes, metrics, and order checks.
- Full velocity tensors were not saved; 20 full-resolution manipulation images
  remain only in the ignored remote output directory.

## [FP32 NUMERICAL CHECK]

The relative L2 discrepancy between the legacy prediction-dtype residual and
the centered FP32 residual ranges from `0.003547` to `0.006985` (about
0.35%–0.70%). This correction is measurable but does not materially change the
seed-robustness or temporal-pattern conclusions. FP32 centered arithmetic
remains the required diagnostic definition.

## [KNOWN CONFOUNDERS]

1. A and B contain explicit preservation clauses, whereas AB does not have the
   same linguistic structure.
2. Condition B behaviorally fails strict mug preservation in every seed.
3. Only one source image, one object pair, one prompt set, one model revision,
   and the AB carrier family were tested.
4. The local model directory uses the revision provenance established during
   EXP-001; the run did not cryptographically re-resolve the Hugging Face
   revision.
5. No spatial, attention, mask, causal-intervention, or object-centric evidence
   was collected.

## [NEXT HIGHEST-INFORMATION EXPERIMENT]

Run a paired prompt-template asymmetry control before carrier-family or
cross-image expansion. Keep the source image, seeds, AB carrier states, model,
schedule, precision, and sigma probes fixed. Compare the current prompts with a
balanced atomic template in which A and B use only their edit clauses and AB is
the exact conjunction of those same clauses. Measure the change in Q1/Q2,
direction/scale decomposition, and especially B-to-mug color leakage.

This experiment directly tests the largest observed confound: whether the
stable residual and behavioral leakage are driven partly by unmatched
preservation-clause wording. It should remain a diagnostic experiment, not a
new object-centric method.

## [CONFIDENCE]

- Technical validity: high.
- Seed robustness on this exact test case: high.
- Predominantly scale-related mismatch through middle/late probes: moderate to
  high.
- Semantic or object-interaction interpretation: low / unsupported.
- Cross-image or cross-object generalization: untested.

## [PROPOSED STATE CHANGE]

Do not yet promote an object-centric or causal conclusion. Proposed text for a
future authorized `RESEARCH_STATE.md` update:

> EXP-002 provides technically valid evidence that centered
> reference-corrected non-additivity is reproducible across AB carrier seeds
> 42--46 for the V03 image and fixed prompt set. All seeds share a non-monotonic
> Q1 curve peaking near actual sigma 0.4901. The mismatch is primarily an
> additive-sum scale excess with a persistent directional component. Semantic
> interpretation remains blocked by systematic B-condition color leakage and
> prompt-template asymmetry. Next priority: paired prompt-template control.

## Execution checkpoint policy

For future long runs, query Codex usage before each new seed. If either relevant
window has only approximately 10%–5% remaining, finish the current seed and
pause before the next `_prepare_conditions` call. A reusable seed checkpoint
requires `status=complete`, six valid probe JSON files, and four non-empty
manipulation outputs. A partially completed seed must be rerun in full. Commit a
small `checkpoint_status.json`; do not upload full-resolution images or velocity
tensors merely to checkpoint the run.
