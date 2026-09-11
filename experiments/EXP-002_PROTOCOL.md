# EXP-002 — Seed Robustness of Centered Non-Additivity

## Registered question

Does the centered reference-corrected non-additivity measured in EXP-001 remain
qualitatively and quantitatively stable when only the AB carrier seed changes?

## Fixed protocol

- Seeds: `42, 43, 44, 45, 46`.
- Source image: `assets/exp001/v03_input.png`.
- Carrier: native `AB` trajectory only.
- Model: FLUX.1-Kontext-dev, revision
  `24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d`.
- VeloEdit inspected commit:
  `ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a`.
- Prediction dtype: BF16; centered diagnostic arithmetic: FP32.
- Guidance scale: `2.5`; requested steps: `30`; first-step alignment: `4`.
- Interventions, attention analysis, adapters, and new decomposition methods are
  disabled.
- Target sigmas: `1.00, 0.90, 0.75, 0.50, 0.25, 0.10`, each mapped to the nearest
  non-terminal point in the actual aligned schedule and explicitly recorded.

The exact four text conditions are:

- `0`: `Keep the image unchanged.`
- `A`: `change the black mug to a white mug while keeping the beige vase with flowers unchanged`
- `B`: `keep the black mug unchanged and change the beige vase with flowers to a blue vase`
- `AB`: `change the black mug to a white mug and change the beige vase with flowers to a blue vase`

Condition `0` is an explicit preservation instruction accepted by the
FLUX.1-Kontext text interface. It is not an empty prompt, unconditional branch,
or claimed zero-velocity condition.

The prompts intentionally retain a linguistic asymmetry: A and B contain
explicit keep-unchanged clauses, whereas AB does not. This is a registered
confound, not silently corrected in EXP-002.

## Primary measurement

At the identical `z_t` and timestep for all four conditions:

```text
dA  = fp32(v_A)  - fp32(v_0)
dB  = fp32(v_B)  - fp32(v_0)
dAB = fp32(v_AB) - fp32(v_0)
r_ref = dAB - dA - dB
Q1 = ||r_ref|| / (||dAB|| + eps)
Q2 = ||r_ref|| / (||dA|| + ||dB|| + eps)
```

The legacy BF16 expression `v_AB - v_A - v_B + v_0` is also computed before
casting, solely to quantify numerical discrepancy. Neither residual is assigned
interaction semantics.

Each probe repeats predictions in forward order `0,A,B,AB` and reverse order
`AB,B,A,0`. Any tolerance violation is a blocking measurement issue. The runner
does not average, reset, or otherwise compensate for order dependence.

## Execution

Run the development verification first:

```bash
python scripts/run_exp002.py \
  --model-path /root/autodl-tmp/models/FLUX.1-Kontext-dev \
  --cpu-offload \
  --seeds 42 \
  --output-dir outputs/EXP-002/dev_seed42
```

After inspecting the dev record, run the fixed five-seed protocol:

```bash
python scripts/run_exp002.py \
  --model-path /root/autodl-tmp/models/FLUX.1-Kontext-dev \
  --cpu-offload
```

Full velocity tensors are disabled by default. Each seed saves scalar probe
records plus independent final images for conditions 0/A/B/AB. Qualitative
manipulation labels remain `PENDING_HUMAN_REVIEW` until those images are
inspected; numerical rows are never silently filtered by that label.
