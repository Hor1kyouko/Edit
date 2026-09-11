# EXP-003 Protocol — Prompt-Controlled Non-Additivity and Minimal Cross-Image Replication

## Decision question

Is the seed-robust centered non-additivity from EXP-002 robust enough to
prompt-template control and minimal image variation to justify subsequent
spatial/object mechanism research?

EXP-003 does not introduce a new editing method. VeloEdit intervention,
attention analysis, masks, decomposition fitting, and residual injection remain
disabled.

## EXP-003A: paired prompt-template control

The source image, five seeds (42–46), FLUX.1-Kontext model/revision, inspected
VeloEdit commit, AB carrier, scheduler, guidance, BF16 forward, FP32 diagnostic
arithmetic, and preservation reference are identical to EXP-002.

Exact atomic clauses:

```text
0:  Keep the image unchanged.
A:  change the black mug to a white mug
B:  change the beige vase with flowers to a blue vase
AB: change the black mug to a white mug and change the beige vase with flowers to a blue vase
```

AB is the literal composition `A + " and " + B`; it is also identical to the
legacy AB instruction. Legacy scalar records and images are reused rather than
rerun.

The reduced probes are fixed to schedule indices 0, 9, 17, and 23, which map
under the EXP-002 schedule to actual sigmas approximately 1.00, 0.76, 0.49,
and 0.10. There is no new peak search.

Every probe retains same-latent, finite, shape/dtype, and non-text-state checks.
Order invariance is spot-checked for seed 42 at indices 0 and 17. Any failure
blocks scientific interpretation and escalates the configuration to full order
validation.

Primary quantities remain centered FP32 Q1, Q2, cosine alignment, additive-sum
scale ratio, cos(dA,dB), and relative component norms. No LS or NNLS is used.

Final 0/A/B/AB images are reviewed per seed with PASS/PARTIAL/FAIL. The AB image
is decoded from the unmodified AB carrier; 0/A/B are independent trajectories.

## Conditional gate to EXP-003B

EXP-003B runs only if EXP-003A corresponds to the preregistered Outcome A or C
and its atomic manipulation check is behaviorally valid. Outcome B, Outcome D,
or a blocking measurement issue stops execution after EXP-003A.

## EXP-003B: approved cases

Both cases use seeds 42–44 and the same four schedule probes. Results remain
separate per image and are never pooled into a global generalization statistic.

### pinkballoon

```text
0:  Keep the image unchanged.
A:  change the red balloon to a purple balloon
B:  change the blue suitcase to a yellow suitcase
AB: change the red balloon to a purple balloon and change the blue suitcase to a yellow suitcase
```

### elk

```text
0:  Keep the image unchanged.
A:  change the left elk's fur to white
B:  change the right elk's fur to dark brown
AB: change the left elk's fur to white and change the right elk's fur to dark brown
```

The elk case has a preregistered instance-binding risk because both targets
belong to the same category. A/B collateral changes to the other elk must be
reported and must not be silently interpreted as object interaction.

Images are read from the official SplitFlow checkout under `external/`, which
remains read-only and untracked. The run records exact image hashes and the
external repository commit; third-party images are not copied into this repo.

