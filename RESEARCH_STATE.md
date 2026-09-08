# Research State

Last updated: 2026-09-08

## Project Goal

Develop a scientifically defensible image-editing contribution around Flow
Matching / Rectified Flow and complex multi-instruction editing.

## Current Research Question

At an identical latent state z_t and timestep t, how do the conditional velocity
predictions v_A(z_t,t), v_B(z_t,t), and v_AB(z_t,t) relate when every
non-instruction input is held fixed?

## Current Main Hypothesis

The joint conditional velocity may contain object-dominant components associated
with A and B plus a non-negligible interaction/residual term:

    r_ref = v_AB - v_A - v_B + v_0

Status: **untested**. The survey establishes measurement feasibility, not
additivity, object separability, or spatial localization.

## Active Hypotheses

1. Single-object instructions produce velocity changes concentrated in their
   corresponding object regions.
2. The A/B/AB relationship varies with timestep, latent state, prompt wording,
   and guidance formulation.
3. Joint instructions contain interaction terms not recoverable by simple
   addition.
4. Attention localization may correlate with, but is not assumed equivalent to,
   causal velocity localization.

All remain unvalidated.

## Supported Findings

These are engineering findings from the completed pipeline survey:

- None of the six inspected repositories exposes exact same-latent A/B/AB
  evaluation as a first-class experiment.
- All six can in principle be instrumented without retraining to evaluate
  multiple conditions at one explicit latent and timestep, provided caches,
  stochastic inputs, guidance, masks, and other controls are fixed.
- VeloEdit exposes a compact velocity callback and records its prediction before
  the Euler update.
- SAM-Flow exposes explicit source/target prediction helpers plus spatial
  attention and mask machinery.
- Native FlowEdit-style source and target branches commonly use different latent
  states; their native difference is not an exact same-latent conditional
  difference.
- MaskFlow and IDAttn have condition-dependent spatial controls that must be
  frozen to avoid confounding prompt and control-policy effects.

Evidence and code locations: papers/notes/PIPELINE_SURVEY.md.

## Tentative Findings

- Primary base candidate: **VeloEdit**.
- Secondary base candidate: **SAM-Flow**.
- SplitFlow is a useful simpler SD3 baseline after separating its prediction
  helper from native different-latent construction.

These are pipeline-selection judgments, not scientific findings.

## Rejected Hypotheses

None. No decomposition hypothesis has yet been tested.

## Current Method

Characterization before method design. Preserve native model prediction, add only
an opt-in diagnostic wrapper, evaluate all conditions on a cloned z_t at one t,
and record raw outputs before scheduler or spatial-intervention updates.

No object-centric velocity decomposition method has been implemented.

## Current Pipeline

Survey complete; diagnostic implementation not started.

| Candidate | Family | Same-latent feasibility | Main value |
|---|---|---:|---|
| VeloEdit | FLUX / Qwen-Image-Edit | High | Clean callback and velocity logging |
| SAM-Flow | FLUX / SD3 | High | Branch helpers and spatial diagnostics |
| SplitFlow | SD3 | High after wrapper | Simple FlowEdit-style baseline |
| Follow-Your-Shape | FLUX | Moderate | Inversion, K/V injection, temporal maps |
| MaskFlow | Qwen-Image-Edit | Moderate-high | Masked/effective-field analysis |
| IDAttn | FLUX Kontext | Moderate-high | Token/region attention control |

FlowDC was paper-verified, but no official public code repository was found.

## Main Baselines

1. Raw same-latent conditional velocities from the selected editor.
2. Reference-corrected affine composition v_A + v_B - v_0.
3. Uncorrected composition v_A + v_B, reported separately.
4. Native FlowEdit-style behavior as a different-latent operational baseline.

## Datasets / Benchmarks

Not selected. Begin with one controlled source image containing two clearly
separated objects and instructions A, B, and AB. Benchmark selection follows only
after the measurement protocol passes invariance checks.

## Evaluation

At selected early, middle, and late timesteps:

- verify identical latent tensors, timesteps, image conditions, seeds, masks,
  dtype, guidance, and cache state across conditions;
- save raw model-native outputs before scheduler advancement;
- report global and object-region cosine similarity, norm ratios, cross-region
  leakage, and residual norms;
- report both r_raw = v_AB - v_A - v_B and
  r_ref = v_AB - v_A - v_B + v_0;
- permute condition evaluation order to detect mutable-state effects.

## Known Failure Modes

- Prediction parameterization and timestep/sigma conventions differ by model.
- Raw conditional outputs and CFG-guided outputs can be inadvertently mixed.
- Image-conditioning tokens and generated tokens can have different roles.
- Inversion and forward-generation latents may have different distributions.
- Attention hooks, K/V injection, caches, counters, noise, and midpoint calls can
  make evaluation stateful.
- Condition-specific masks or boxes can leak the intended partition.
- A single prompt/image/seed can produce misleading apparent localization.
- Some inspected repositories lack a license file at the inspected commit.

## Open Questions

- Which prediction level should be primary: raw conditional, guided, or
  post-intervention effective field?
- Is v_0 required for an appropriate affine comparison under each guidance
  implementation?
- How stable are regional relationships across timestep, seed, prompt paraphrase,
  object layout, and latent-state source?
- Do attention/token regions predict velocity localization or merely correlate
  with it?

## Current Experiments

None. Only paper/repository verification and read-only code inspection are
complete.

## Next Experiments

Implement one minimal, opt-in VeloEdit diagnostic wrapper accepting precomputed
z_t and t and evaluating fixed conditions {0,A,B,AB} without a scheduler update.
First run static/CPU smoke and invariance checks; then run one small controlled
GPU case after review. Replicate in SAM-Flow only after the protocol is stable.
