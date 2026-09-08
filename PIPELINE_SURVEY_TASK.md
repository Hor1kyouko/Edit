# Task: Flow Editing Pipeline Survey

## Objective

Survey recent Flow Matching / Rectified Flow image editing methods and their official code repositories.

The purpose is to identify the best implementation base for experiments on:

**object-centric velocity field decomposition in complex image editing.**

Read `RESEARCH_CONTEXT.md` before starting this task.

Also follow all project-level instructions in:

- `AGENTS.md`
- `AGENT_RESEARCH_GUIDELINE.md`
- `STATE.md`

if these files exist.

---

## Phase 1 — Paper and Repository Verification

Investigate the following papers first:

1. FlowDC
2. SplitFlow
3. Follow-Your-Shape
4. VeloEdit
5. SAM-Flow
6. MaskFlow
7. Shifting the Breaking Point of Flow Matching for Multi-Instance Editing

For every paper:

- verify the exact paper title,
- verify publication venue / status,
- identify the official project page,
- identify the official GitHub repository,
- determine whether the repository is actually maintained by the authors,
- determine whether inference code is available,
- determine whether pretrained weights are available,
- record the license if available.

Prefer:

1. official author repository,
2. official project page,
3. conference / arXiv paper,

in that order.

Do not treat an unofficial reimplementation as the primary implementation unless no official repository exists.

Clearly label unofficial implementations.

---

## Phase 2 — Method / Pipeline Understanding

For every usable method, understand the actual inference pipeline.

Do NOT rely only on README descriptions.

Inspect the implementation.

Record:

### Base model

Examples:

- FLUX
- FLUX.1-dev
- SD3
- Stable Diffusion
- rectified-flow transformer
- custom Flow Matching model

Also record the major framework:

- Diffusers
- custom PyTorch
- ComfyUI
- other

### Inputs

Determine:

- source image,
- source prompt,
- target prompt,
- mask,
- bounding box,
- reference image,
- inversion latent,
- random noise,
- other conditioning.

### Inversion

Determine whether the method uses:

- no inversion,
- RF inversion,
- DDIM-like inversion,
- source trajectory reconstruction,
- approximate inversion,
- custom inversion.

Identify the exact implementation.

---

## Phase 3 — Velocity Prediction

This is one of the most important parts of the survey.

Locate the exact place where the model predicts:

\[
v_\theta(z_t,t,c)
\]

or its equivalent representation.

For each repository report:

- file path,
- class,
- function,
- main forward call,
- tensor returned,
- tensor shape if identifiable,
- whether the tensor is true velocity,
- noise prediction,
- flow prediction,
- residual,
- transformed scheduler output.

Do NOT call a tensor "velocity" unless verified from the paper or implementation.

---

## Phase 4 — Latent Update

Identify the actual sampling update:

\[
z_{t+\Delta t}
=
z_t + \Delta t \, v_t
\]

or its implementation equivalent.

Record:

- scheduler,
- timestep definition,
- sign convention,
- scaling,
- normalization,
- CFG handling.

If the repository uses a different parameterization, document the exact relationship between the model output and the effective flow / velocity update.

---

## Phase 5 — Source / Target Velocity Construction

For each method determine whether it is possible to independently compute:

\[
v_{\text{source}}(t)
\]

and:

\[
v_{\text{target}}(t).
\]

Also determine whether the same latent state \(z_t\) can be evaluated with different text conditions.

This is essential.

We eventually need controlled evaluation of:

\[
v_A(t),
\quad
v_B(t),
\quad
v_{AB}(t).
\]

Determine whether these quantities can be evaluated at exactly the same:

- latent state,
- timestep,
- noise realization,
- source image context.

Explicitly distinguish between:

- same latent state evaluation,
- same timestep but different trajectory states,
- independently sampled trajectories.

The first case is strongly preferred for controlled decomposition analysis.

---

## Phase 6 — Velocity Manipulation

For each method identify any operation involving:

- velocity difference,
- velocity addition,
- velocity interpolation,
- velocity projection,
- orthogonal decomposition,
- semantic decomposition,
- residual velocity,
- mask-based velocity modification,
- source anchoring,
- regional velocity control.

Write the actual mathematical operation where possible.

Link each equation to the corresponding implementation location.

For example, identify operations conceptually similar to:

\[
\Delta v
=
v_{\text{target}}
-
v_{\text{source}}
\]

or:

\[
v'
=
\alpha_A v_A
+
\alpha_B v_B
\]

or:

\[
v'
=
M \odot v_{\text{edit}}
+
(1-M)\odot v_{\text{source}}.
\]

Do not assume the repository follows these equations unless verified.

---

## Phase 7 — Spatial Control

Determine whether the method uses:

- hard mask,
- soft mask,
- dynamic mask,
- attention-derived region,
- segmentation,
- bounding box,
- token localization,
- latent-space mask,
- pixel-space mask.

Determine where the spatial control enters the pipeline:

- attention,
- velocity,
- latent update,
- source injection,
- conditioning,
- objective,
- post-processing.

Also record whether the spatial region changes with timestep.

---

## Phase 8 — Attention Access

Determine whether the repository already provides hooks for:

- cross-attention,
- self-attention,
- token maps,
- spatial attention visualization.

If not, estimate the modification required.

This may later be useful for comparing:

\[
\text{attention localization}
\]

against:

\[
\text{velocity localization}.
\]

Record:

- exact modules where attention is computed,
- whether attention tensors are directly available,
- expected tensor sizes,
- whether spatial resolution changes across layers.

---

## Phase 9 — Suitability for Object-wise Velocity Analysis

For each repository determine whether we could construct an experiment containing:

source prompt:

\[
P_s
\]

single edit A:

\[
P_A
\]

single edit B:

\[
P_B
\]

combined edit:

\[
P_{AB}.
\]

and extract:

\[
v_s(t),
\quad
v_A(t),
\quad
v_B(t),
\quad
v_{AB}(t).
\]

Most importantly, determine whether these quantities can be evaluated at a shared:

\[
z_t.
\]

Evaluate whether the implementation supports controlled comparisons such as:

\[
v_{AB}(z_t,t)
\]

versus:

\[
v_A(z_t,t)
\]

and:

\[
v_B(z_t,t)
\]

for the exact same latent tensor \(z_t\).

Also assess whether different prompt conditions can be evaluated without changing:

- source-image encoding,
- inversion state,
- random seed,
- latent normalization,
- scheduler state.

---

## Phase 10 — Minimum Required Modification

For each repository describe the smallest possible modification needed to save per-step velocity tensors.

Example format:

```text
Repository:
external/example/

File:
pipeline.py

Class:
ExamplePipeline

Function:
denoise_step()

Existing logic:
velocity = transformer(...)

Required modification:
Save velocity.detach().cpu() before scheduler update.

Estimated difficulty:
Low
```

Do NOT implement the modification during this survey.

Only identify it.

Also identify whether dumping per-step tensors would significantly increase memory usage.

If so, suggest lightweight alternatives such as:

selected timesteps only,
reduced precision,
spatial norm maps,
cosine statistics,
CPU offload after each step.

---

## Phase 11 — Comparison Table

Create a comparison table containing at least:

Method	Venue / Status	Base Model	Official Repo	Inversion	Velocity Accessible	Same-latent Multi-prompt Evaluation	Mask / Region	Velocity Composition	Attention Access	Modification Difficulty

Use ratings such as:

Excellent
Good
Moderate
Poor

only after explaining the reason.

Do not rank methods only by image quality.

The main criterion is suitability for velocity-field analysis.

---

## Phase 12 — Repository Selection

Recommend the best TWO repositories for the next experimental stage.

Priority criteria:

direct access to Flow Matching velocity,
controlled same-latent evaluation,
source / target editing support,
minimal modification,
reproducible inference,
recent and competitive editing performance,
compatibility with object masks / spatial analysis,
clean code structure,
manageable hardware requirements.

Do not simply choose the method with the best reported image quality.

The primary criterion is suitability for velocity-field research.

If none of the paper repositories are ideal, explicitly recommend using a simpler vanilla Flow Matching editing pipeline instead.

---

## Phase 13 — Initial A / B / AB Experimental Feasibility

For the top candidate repositories, assess whether the following experiment is feasible without retraining.

Given a source image and source prompt:

$$ P_s $$

construct:

$$ P_A, \quad P_B, \quad P_{AB}. $$

At selected timesteps, compute:

$$ v_A(t), \quad v_B(t), \quad v_{AB}(t) $$

at the same latent state.

Then estimate:

$$ \hat v_{AB} = \alpha_A v_A + \alpha_B v_B $$

using:

least squares,
NNLS,
cosine-based projection,

only as analysis tools.

Possible quantities to measure:

$$ R^2_{\text{recon}}(t) $$ $$ \frac{\|v_{AB}-\hat v_{AB}\|}{\|v_{AB}\|} $$ $$ \cos(v_A,v_B) $$ $$ \cos(v_A,v_{AB}) $$ $$ \cos(v_B,v_{AB}). $$

Do not implement these experiments yet.

Only determine whether the repository structure supports them cleanly.

---

## Repository Handling Rules

During the initial survey:

do not modify third-party repositories,
do not commit third-party source code into the main repository,
do not download large model checkpoints unless strictly necessary,
do not start expensive GPU inference,
do not implement the proposed research method,
do not rewrite external repositories,
do not silently patch third-party code.

If repositories need to be cloned, place them under:

external/

Treat external/ as read-only reference implementations.

The main repository should contain only:

project instructions,
research notes,
survey documents,
experiment scripts written for this project,
minimal patches if later required.

---

## Deliverables

Create:

papers/notes/PIPELINE_SURVEY.md

The document should contain:

Executive Summary
Verified Paper / Repository List
Per-paper Method Summary
Per-repository Pipeline Analysis
Exact Velocity Prediction Locations
Latent Update Equations and Code Locations
Source / Target Velocity Construction
Spatial / Mask Handling
Velocity Composition Operations
Attention Access
Suitability for A / B / AB Experiments
Same-Latent Evaluation Feasibility
Minimum Required Code Modifications
Comparison Table
Recommended Base Repositories
Open Technical Questions
Suggested Next Experimental Step

Also update:

STATE.md

with:

what was completed,
important findings,
unresolved issues,
selected candidate repositories,
recommended next action.
---

## Important Research Discipline

Separate every important claim into one of three categories:

Confirmed from paper

The statement is directly supported by the paper.

Confirmed from code

The statement was verified in the released implementation.

Research inference / hypothesis

The statement is an interpretation, extrapolation, or proposed hypothesis.

Never present an inferred behavior as if it were verified from code.

If paper and code disagree, explicitly document the discrepancy.

Do not force the original research hypothesis to be true.

If the repository survey suggests that the current object-wise decomposition hypothesis is technically or conceptually weak, report that clearly.

---

## Final Decision Rule

The survey should answer the following practical question:

Which existing Flow Matching image-editing implementation gives the cleanest and most scientifically controlled access to object-related velocity analysis?

The selected base repository should make it possible to later study:

$$ v_{\text{complex}} = \sum_i w_i(x,t)v_i(x,t) + v_{\text{interaction}}(x,t) + r(x,t) $$

without requiring unnecessary architectural changes or retraining.

The survey stage ends once the top one or two candidate pipelines are identified and their exact velocity extraction points are documented.