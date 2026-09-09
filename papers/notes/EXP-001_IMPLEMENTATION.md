# EXP-001 Implementation Contract

Date: 2026-09-09

## Scope

EXP-001 is a project-owned diagnostic wrapper around the read-only
`external/VeloEdit/` FLUX.1-Kontext path. It characterizes model predictions;
it does not implement VeloEdit intervention, attention analysis, SAM-Flow
replication, regional decomposition, or a new object-centric method.

The AB instruction drives the carrier trajectory. At selected pre-update
early/middle/late steps, the runner clones the current AB latent and evaluates
four text conditions on that exact state and sigma:

1. `0`
2. `A`
3. `B`
4. `AB`

## Condition 0

The default text is exactly:

> Keep the image unchanged.

This is an **operational preservation-instruction reference** for the
FLUX.1-Kontext editing interface. It is not an empty prompt, an unconditional
branch, or a claim that the predicted velocity is zero. The inspected Kontext
interface does not define a canonical null-edit condition. The CLI therefore:

- prints the chosen condition-0 text at startup;
- records it verbatim in every result;
- allows an explicit override with `--prompt-0`.

Results using different condition-0 strings are different experimental
conditions and must not be pooled silently.

## Raw and effective prediction levels

In `external/VeloEdit/analyzers/flux.py`, the transformer receives generated
tokens concatenated with fixed condition-image tokens, normalized timestep,
prompt embeddings, and an embedded guidance value. Its returned tensor contains
predictions for generated and condition-image tokens. VeloEdit slices it to the
generated-token length.

EXP-001 names that generated-token slice:

`raw_conditional_model_velocity`

For FLUX.1-Kontext, guidance is distilled/embedded and supplied as an input to
the transformer. There is no external CFG arithmetic after the transformer.
Consequently the exact same generated-token tensor is passed to VeloEdit's Euler
update and is also named:

`effective_velocity`

The two result fields are retained separately for semantic clarity, but metadata
states `post_transformer_guidance_applied=false` and
`raw_equals_effective_by_construction=true`. This equality is specific to the
inspected FLUX path and must not be generalized to VeloEdit's Qwen path.

## Residuals

For both prediction levels, EXP-001 computes:

`r_raw = v_AB - v_A - v_B`

`r_ref = v_AB - v_A - v_B + v_0`

The primary diagnostic is effective-level `r_ref`. Both are algebraic
quantities. The implementation assigns neither one an interaction,
object-component, nor causal meaning.

## Condition-order invariance

Every probe executes:

- order 1: `0 -> A -> B -> AB`
- order 2: `AB -> B -> A -> 0`

For every condition and both prediction levels, it reports:

- maximum absolute difference;
- relative L2 difference, using
  `max(L2(order_1), L2(order_2), epsilon)` as denominator.

The default tolerances are `1e-5` for both metrics. If either metric exceeds
its tolerance for any condition or prediction level, the measurement is marked
blocked, a partial report is saved, and the carrier trajectory is not advanced
using that probe. The runner does not reset, average, reorder, or otherwise
compensate for the dependence.

## Pre-update invariants

- Generated latent `z_t` is cloned from the AB carrier.
- Sigma/timestep is identical across conditions.
- Source image, seed, working resolution, reference image latent, sigma schedule,
  model instance, weights, dtype, device, guidance scale, and scheduler
  configuration are fixed.
- Repeated VeloEdit prompt preparation must reproduce bitwise-identical initial
  generated latents, reference latents, and sigma schedules or the run aborts.
- Every prediction receives a cloned copy of the same latent, and in-place input
  mutation is checked.
- No Euler update occurs until both condition orders pass.
- VeloEdit intervention and attention instrumentation are disabled.

## Probe selection

With `N` effective Euler transitions:

- early = transition `0`;
- middle = transition `N // 2`;
- late = transition `N - 1`.

These are the latent states immediately before their respective updates.
Explicit integer indices may be supplied, but at least three distinct valid
indices are required.

## Usage

```powershell
python -m src.exp001.veloedit_flux `
  --image path/to/source.png `
  --prompt-a "Instruction affecting object A" `
  --prompt-b "Instruction affecting object B" `
  --prompt-ab "Combined A and B instructions" `
  --prompt-0 "Keep the image unchanged." `
  --probe-steps early,middle,late `
  --output-dir outputs/EXP-001
```

The real runner requires the dependencies listed in
`external/VeloEdit/requirements.txt`, access to
`black-forest-labs/FLUX.1-Kontext-dev`, and suitable GPU memory. It does not
download anything until the user runs it in such an environment.

Each probe writes `metrics.json`, order-separated raw/effective tensors, and
raw/reference-corrected residual tensors. Use `--no-save-tensors` for scalar
reports only. Generated outputs remain under the ignored `outputs/` directory.

## Current evidence status

This implementation has not run a real FLUX.1-Kontext forward pass. Mock/unit
tests validate the registered condition set, order definitions, probe selection,
residual algebra, and blocking behavior when PyTorch is available. No scientific
conclusion should be updated until real-model evidence is produced.
