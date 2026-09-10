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

## AutoDL network preflight

This AutoDL-specific step changes only network routing, not the registered
experimental variables.

Before accessing Hugging Face or GitHub resources:

```bash
source /etc/network_turbo
```

Use this acceleration only for academic access to:

- `github.com`
- `githubusercontent.com`
- `githubassets.com`
- `huggingface.co`

After the required metadata check, repository access, or model download
finishes, disable the proxy to avoid affecting other connections:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset HF_ENDPOINT
```

Verify that the proxy has been cleared before dependency installation or other
network operations:

```bash
env | grep -Ei '^(http|https|all)_proxy='
```

The command should produce no output. Clearing both lower- and upper-case
variants is required because inherited shell settings may otherwise route a
transfer through a local proxy even after `/etc/network_turbo` is disabled.
Clear `HF_ENDPOINT` as well unless a non-default Hugging Face endpoint is
deliberately required for the next operation.

## Usage

```powershell
python -m src.exp001.veloedit_flux `
  --image path/to/source.png `
  --prompt-a "Instruction affecting object A" `
  --prompt-b "Instruction affecting object B" `
  --prompt-ab "Combined A and B instructions" `
  --prompt-0 "Keep the image unchanged." `
  --probe-steps early,middle,late `
  --cpu-offload `
  --output-dir outputs/EXP-001
```

The real runner requires the dependencies listed in
`external/VeloEdit/requirements.txt`, access to
`black-forest-labs/FLUX.1-Kontext-dev`, and suitable GPU memory. The complete
BF16 pipeline exceeds a 32 GB GPU when loaded monolithically; `--cpu-offload`
uses sequential component offload without changing weights, dtype, sampling,
or measured tensors. The runner does not download model files when a complete
local `--model-path` is supplied and offline mode is enabled.

Each probe writes `metrics.json`, order-separated raw/effective tensors, and
raw/reference-corrected residual tensors. Use `--no-save-tensors` for scalar
reports only. Generated outputs remain under the ignored `outputs/` directory.

## Current evidence status

One real-model validation run completed for `v03_edit_instruction` under the
registered seed-42, BF16, 30-requested-step configuration. The effective
schedule contained 24 transitions, with probes at steps 0, 12, and 23. All
forward/reverse condition-order comparisons were exactly equal within the
registered metrics. The effective reference-corrected residual L2 norms were
246.387802, 89.399887, and 57.805454 at early, middle, and late respectively.

This is evidence that the measurement path is executable and order-invariant
for this run. The decreasing residual is a single-case observation, not an
established temporal law or evidence of a specific interaction mechanism.
Replication across seeds and instruction/image cases remains required.
