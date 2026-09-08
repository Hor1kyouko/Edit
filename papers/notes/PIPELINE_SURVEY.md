# Pipeline Survey: Controlled Velocity-Field Analysis for Flow-Based Image Editing

Date: 2026-09-08

## Executive Summary

[KEY FINDING]

No surveyed release currently exposes `v_A(z_t,t)`, `v_B(z_t,t)`, and `v_AB(z_t,t)` at one exact shared latent state as a supported first-class experiment. Six author-linked repositories were inspected, and all six expose a model forward point from which this controlled evaluation can be constructed without retraining. FlowDC is the closest conceptual precedent, but no official code repository was found in its official paper records.

[WHAT IT MEANS]

The central comparison must be added as a diagnostic wrapper, not inferred from velocities recorded along separate editing trajectories. Same timestep is insufficient: SplitFlow, Follow-Your-Shape, and SAM-Flow normally evaluate source and target conditions at different latent states. Treating those tensors as same-latent evidence would confound prompt conditioning with trajectory displacement.

[IMPACT ON CURRENT HYPOTHESIS]

The object-centric decomposability hypothesis remains **untested**, not supported or contradicted. FlowDC, Follow-Your-Shape, VeloEdit, and SAM-Flow provide evidence that velocity or velocity discrepancy contains useful semantic/spatial structure, but none establishes that independently prompted object velocities linearly explain a combined-edit velocity at a shared state.

[NEXT EXPERIMENT]

Implement a minimal, analysis-only shared-latent probe first in VeloEdit's FLUX.1-Kontext path, then replicate it in SAM-Flow's FLUX path. At selected timesteps, freeze one generated-token latent `z_t`, timestep/sigma, source-image tokens, seed, scheduler state, and model settings; batch or sequentially evaluate instructions A, B, and AB before any update. Save CPU-offloaded FP16 tensors or summary statistics only. Do not alter the sampling trajectory in the first validation.

[CONFIDENCE]

High for paper identity, repository provenance, code locations, model-output/update semantics, and the finding that native same-latent A/B/AB evaluation is absent. Moderate for modification difficulty and hardware estimates. No empirical confidence is assigned to the decomposition hypothesis because no model inference was run.

## Scope and Evidence Discipline

This survey began with official paper and repository verification, followed by direct inspection of shallow clones. No checkpoint, dataset, dependency, or GPU inference was downloaded or executed. Repositories under `external/` were not modified.

Evidence labels used below:

- **Confirmed from paper**: stated in an official proceedings paper or arXiv manuscript.
- **Confirmed from code**: directly observed in the inspected commit.
- **Research inference / hypothesis**: an engineering or scientific interpretation that has not yet been executed.

`AGENT_RESEARCH_GUIDELINE.md` was requested but does not exist in the repository. `RESEARCH_GUIDELINES.md` was read as the available methodology document. `STATE.md` also does not exist; the user-requested `RESEARCH_STATE.md` is updated with this survey.

## Verified Paper and Repository List

| Method | Exact title | Venue / status verified | Official project page | Official repository | Author-maintained evidence | Inference | Method weights | Code license |
|---|---|---|---|---|---|---|---|---|
| FlowDC | *FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing* | CVPR 2026, CVF Open Access | No distinct page found | **No official repository found** | N/A | Not released | Training-free in paper; FLUX.1-dev base required | N/A |
| SplitFlow | *SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing* | NeurIPS 2025 Main Conference | Proceedings page serves as official page | [Harvard-AI-and-Robotics-Lab/SplitFlow](https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow) | Proceedings and paper link this repository | Yes, SD3 | No method weights; uses SD3 Medium and Mistral-7B | README claims MIT, but the inspected commit has no `LICENSE` file |
| Follow-Your-Shape | *Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control* | ICLR 2026 | [follow-your-shape.github.io](https://follow-your-shape.github.io/) | [mayuelala/FollowYourShape](https://github.com/mayuelala/FollowYourShape) | Project page links code; corresponding-author account | Yes | Training-free; FLUX.1-dev and external ControlNet/base weights | Apache-2.0 |
| VeloEdit | *VeloEdit: Training-Free Consistent and Continuous Instruction-Based Image Editing via Velocity Field Decomposition* | arXiv:2603.13388, preprint | [xmulzq.github.io/VeloEdit](https://xmulzq.github.io/VeloEdit/) | [xmulzq/VeloEdit](https://github.com/xmulzq/VeloEdit) | arXiv links repository; author-named account | Yes, FLUX Kontext and Qwen Image Edit | No method weights; base-model weights required | **No license file found** |
| SAM-Flow | *SAM-Flow: Source-Anchored Masked Flow for Training-Free Image Editing* | arXiv:2606.06228, preprint | No distinct page found | [chwbob/Sam-Flow](https://github.com/chwbob/Sam-Flow) | arXiv directly links repository; repository identifies itself as official | Yes, FLUX and SD3 | No method weights; base-model weights required | MIT |
| MaskFlow | *MaskFlow: Precise, Consistent and Seamless Regional Image Editing* | arXiv:2608.06929v2, preprint | [reychiaro.github.io/MaskFlow](https://reychiaro.github.io/MaskFlow/) | [ReyChiaro/MaskFlow](https://github.com/ReyChiaro/MaskFlow) | Paper/project/author page link repository | Yes, Qwen-Image-Edit | Yes: S/SEC SFT LoRAs and 8/16-step DMD residuals on Hugging Face | MIT for code/adapters; base model separate |
| IDAttn | *Shifting the Breaking Point of Flow Matching for Multi-Instance Editing* | Accepted at ICML 2026 (arXiv comment) | No distinct page found | [Blowing-Up-Groundhogs/IDAttn](https://github.com/Blowing-Up-Groundhogs/IDAttn) | Repository identifies itself as official and matches paper method/authors | Yes, FLUX.1-Kontext-dev | Base model works; optional local LoRA supported, but no released LoRA link found | **No license file found** |

Primary verification sources: [FlowDC CVPR record](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_FlowDC_Flow-Based_Decoupling-Decay_for_Complex_Image_Editing_CVPR_2026_paper.html), [SplitFlow NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e124f1547f7ac87e33d348b827d4291b-Abstract-Conference.html), [Follow-Your-Shape ICLR record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b315c0b736711b56f33cbacfb6d5d67-Abstract-Conference.html), [VeloEdit arXiv](https://arxiv.org/abs/2603.13388), [SAM-Flow arXiv](https://arxiv.org/abs/2606.06228), [MaskFlow arXiv](https://arxiv.org/abs/2608.06929), and [IDAttn arXiv](https://arxiv.org/abs/2602.08749).

### Inspected repository snapshots

| Local reference | Inspected commit | Latest commit date at clone time |
|---|---|---|
| `external/SplitFlow/` | `c9ae45d2a6e386d6d00b91f19ec30dac9a20e786` | 2025-11-05 |
| `external/FollowYourShape/` | `ba35feac8541065a1c5ee005bf369853ff3fbc63` | 2026-04-10 |
| `external/VeloEdit/` | `ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a` | 2026-07-28 |
| `external/Sam-Flow/` | `8167978666e6df11e987a453934b89ec422bd3a4` | 2026-06-13 |
| `external/MaskFlow/` | `4baac7733c5e33c5ef5c85be36b70631a40441e1` | 2026-08-13 |
| `external/IDAttn/` | `79ce339b9e7c789fbbed4b9a79f92d5067310abd` | 2026-07-02 |

## Cross-Method Synthesis

### Shared assumptions

1. **Confirmed from paper/code:** Rectified/flow-matching transformer outputs can be used directly as an ODE vector field under the repository's sigma/timestep convention.
2. **Confirmed from paper/code:** Source preservation is not guaranteed by a target-conditioned global flow. Every method adds structure through differential flows, source anchoring, feature reuse, masks, or attention partitioning.
3. **Confirmed from code:** “Velocity” tensors frequently retain diffusion-era names such as `noise_pred`. Semantics must be determined from their training target and integration rule, not the variable name.
4. **Research inference:** A/B/AB decomposition can be tested only if all conditions see the exact same state. Existing editing trajectories are useful for generation but are not controlled causal comparisons of conditioning.

### Main mechanism families

- **Prompt/flow decomposition:** FlowDC and SplitFlow create multiple semantic trajectories and combine/project them.
- **Velocity discrepancy as localization:** Follow-Your-Shape and VeloEdit use disagreement between editing/preservation dynamics to infer spatial structure.
- **Explicit region anchoring:** SAM-Flow and MaskFlow impose spatial preservation at the latent/probability-path level.
- **Attention disentanglement:** IDAttn binds instruction text to user-specified regions during velocity estimation.

### Unresolved assumption shared by the literature

None of the surveyed papers provides a controlled test that compares independently defined A, B, and AB conditional model outputs at the same `z_t`. Parallel trajectories often share a source noise/path, but their target states diverge. Therefore, published “sub-flow” structure does not by itself establish intrinsic object-wise linear decomposability of the conditional velocity field.

## Per-Method and Per-Repository Pipeline Analysis

### 1. FlowDC

**Paper.** [Confirmed from paper] FlowDC uses FLUX.1-dev and inversion-free FlowEdit-style transport. A complex prompt is decomposed into cumulative prompts, parallel editing trajectories share one source trajectory, progressive vectors are orthogonalized, and the complex editing velocity is projected into the resulting subspace while its orthogonal component is decayed.

The paper defines:

\[
z_t^{src}=t\epsilon+(1-t)x_{src},\qquad
z_t^{tar_i}=z_t^{src}+z_t^{edit_i}-x_{src},
\]

\[
v_i^{edit}(t)=v_\theta(z_t^{tar_i},t,P_i)-v_\theta(z_t^{src},t,P_s).
\]

**Same-latent implication.** [Confirmed from paper] The source trajectory is shared, but each target prompt is evaluated at its own `z_t^{tar_i}`. Thus the published PVG tensors are same-timestep/different-target-state quantities, not `v_i(z_t,t)` at one shared state. [Research inference] The conceptual design could support a same-latent probe, but the missing release prevents code-verified modification planning.

**Repository status.** No official repository, inference code, weights, or code license was found through the CVF record, arXiv HTML, author/paper searches, or exact-title GitHub search. Unofficial paper notes were not treated as implementation evidence.

### 2. SplitFlow

**Base and inputs.** [Confirmed from code] Diffusers Stable Diffusion 3 Medium; source image latent, source prompt, negative prompt, a schedule of cumulative target prompts, a seed, and fresh per-step forward noise. It is inversion-free.

**Velocity prediction.** [`SplitFlow_utils.py:58-81`](https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow/blob/c9ae45d2a6e386d6d00b91f19ec30dac9a20e786/SplitFlow_utils.py#L58-L81), function `calc_v_sd3`, calls `pipe.transformer(...)[0]`. The returned tensor is split into source/target unconditional and conditional batches and CFG-combined. Shape is SD3 latent-shaped `[B,C,H,W]` (same as each model input). Although named `noise_pred`, [confirmed from code] it is used directly as the flow derivative.

**Trajectory construction and update.** [`SplitFlow_utils.py:144-251`](https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow/blob/c9ae45d2a6e386d6d00b91f19ec30dac9a20e786/SplitFlow_utils.py#L144-L251):

\[
z_t^{src}=(1-t)x_{src}+t\epsilon,\quad
z_t^{tar_i}=z_t^{edit_i}+z_t^{src}-x_{src},\quad
\Delta v_i=v_i^{tar}-v^{src},
\]

\[
z_{t'}^{edit_i}=z_t^{edit_i}+(t'-t)\Delta v_i.
\]

The code aggregates parallel trajectory latents at line 202 and performs a spatial soft weighting of sub-flow differences at lines 207-220. [Confirmed from code] Source and target forwards use `z_t^{src}` and `z_t^{tar_i}`, respectively, so native outputs are not same-latent.

**Spatial and attention access.** No explicit masks or attention hooks. `joint_attention_kwargs=None` at the transformer call. Diffusers SD3 attention can be instrumented, but this repository does not expose maps.

**A/B/AB feasibility.** [Research inference] Good. Encode `P_A`, `P_B`, and `P_AB` once, repeat the same latent four-way per condition (including fixed negative/source guidance choices), and call `calc_v_sd3` or a small generalized wrapper before any update. This is a low-complexity change. However, SplitFlow's native cumulative prompts and independently evolving target trajectories should not be reused as controlled A/B/AB samples.

**Minimum change.** Add a callback or returned dictionary immediately after line 72/79, accepting an externally supplied shared latent. Save selected steps only. No retraining required. Memory is approximately `3 × B × C × H × W × dtype_size` per sampled timestep, plus optional CFG intermediates.

### 3. Follow-Your-Shape

**Base and inputs.** [Confirmed from code] Custom PyTorch fork of FLUX.1-dev with source image, source and target prompts, RF inversion trajectory, optional ControlNet, and scheduled KV injection. Packed image tokens are `[B,L,C_packed]`; for standard FLUX VAE latents `C_packed=64` after 2×2 packing.

**Velocity prediction and solver.** [`flux/sampling.py:185-272`](https://github.com/mayuelala/FollowYourShape/blob/ba35feac8541065a1c5ee005bf369853ff3fbc63/src/flux/sampling.py#L185-L272) calls `model(...)->(pred,info)` at the current state and again at a midpoint. The update is a second-order midpoint/Taylor-equivalent step:

\[
z_{mid}=z_t+\frac{\Delta t}{2}v(z_t,t),\quad
z_{t'}=z_t+\Delta t\,v_t+\tfrac12\Delta t^2\frac{v_{mid}-v_t}{\Delta t/2}.
\]

During inversion, `(pred + pred_mid)/2` is cached as `inv_noise` even though it is an effective flow/velocity average.

**TDM and spatial control.** [`flux/sampling.py:317-408`](https://github.com/mayuelala/FollowYourShape/blob/ba35feac8541065a1c5ee005bf369853ff3fbc63/src/flux/sampling.py#L317-L408) compares cached inversion velocity with target-denoising velocity, reduces channel energy to tokens, aggregates over time, smooths, and Otsu-thresholds a dynamic edit map. [`flux/modules/layers.py:230-291`](https://github.com/mayuelala/FollowYourShape/blob/ba35feac8541065a1c5ee005bf369853ff3fbc63/src/flux/modules/layers.py#L230-L291) injects source inversion K/V outside the chosen edit-token indices in later single-stream blocks. This is feature/attention control, not direct velocity masking.

**Same-latent implication.** [Confirmed from code] TDM compares an inversion-path velocity and an editing-path velocity that generally correspond to different states, and midpoint evaluations add another state difference. [Research inference] Exact shared-state A/B/AB calls are feasible, but mutable `info`, cached inversion features, second-order evaluation, and optional ControlNet should be disabled or strictly frozen for the initial probe. Modification difficulty is moderate.

**Attention access.** Q/K/V are explicit in `modules/layers.py`, but `flux/math.py` uses fused scaled-dot-product attention and returns only attention output. K/V capture already exists; probability-map capture requires manual `QK^T` or an attention backend hook. Spatial resolution is the packed FLUX token grid and remains tokenized across transformer layers.

### 4. VeloEdit

**Base and inputs.** [Confirmed from code] Diffusers `FluxKontextPipeline` and `QwenImageEditPlusPipeline`; source image, instruction prompt, fixed source-image condition tokens, deterministic seed/noise, and optional LoRA. It does not invert the source image. FLUX and Qwen generated-token states are sequence-shaped `[B,L,C]`.

**Velocity prediction.** FLUX: [`analyzers/flux.py:231-259`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/analyzers/flux.py#L231-L259). Qwen: [`analyzers/qwen.py:326-378`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/analyzers/qwen.py#L326-L378). Both concatenate fixed condition-image tokens to generated tokens, call the transformer, slice the returned tensor back to generated tokens, and optionally apply CFG. The variable name is `noise_pred`, but [confirmed from code] it is the effective vector field passed directly to Euler integration.

**Latent update and existing extraction.** [`core/sampler.py:23-40`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/core/sampler.py#L23-L40):

\[
\Delta\sigma=\sigma_{next}-\sigma,\quad
\hat x_0=z_\sigma-\sigma v,\quad
z_{next}=z_\sigma+\Delta\sigma v.
\]

[`core/sampler.py:222-321`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/core/sampler.py#L222-L321) evaluates `v_pred_fn(z,sigma)`, optionally intervenes, saves every velocity to CPU at line 315, then updates the latent. This is the cleanest existing extraction point in the survey.

**Velocity manipulation.** [`core/intervention.py:11-18`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/core/intervention.py#L11-L18) defines a reference/preservation velocity:

\[
v_{ref}(z_t,t)=\frac{z_t-z_{ref}}{\sigma+\epsilon}.
\]

[`core/intervention.py:255-298`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/core/intervention.py#L255-L298) replaces high-similarity elements with `v_ref` and blends low-similarity elements as `a v_ref+(1-a)v_pred`. [`core/decomposer.py:16-78`](https://github.com/xmulzq/VeloEdit/blob/ae3fec526b6bc9e9c13fd0e08dc9827b0343ba9a/core/decomposer.py#L16-L78) projects `v_pred` onto the global flattened `v_ref` direction and calls the residual an edit component. This is preserve-versus-edit decomposition, not object-wise decomposition.

**Same-latent A/B/AB feasibility.** [Research inference] Excellent after a small wrapper. Prepare one image/noise/schedule, encode A/B/AB instructions while holding source-image condition tokens fixed, and call the three condition-specific `v_pred_fn` equivalents on the same `z` before the line-317 update. The current closure captures one prompt, so native CLI runs are separate trajectories; they are deterministic but still not guaranteed to share later `z_t`.

**Attention access.** None exposed. FLUX passes empty `joint_attention_kwargs`; Qwen passes empty `attention_kwargs`. Adding attention capture is a separate moderate change and should not be mixed into the first velocity validation.

**Minimum change.** Generalize the closure to accept pre-encoded conditioning (or create three closures sharing the same fixed image latents), evaluate selected timesteps, and retain the existing CPU offload. Estimated difficulty: low. Existing all-step/all-tensor saving can be memory/disk-heavy; selected steps, FP16, norms, regional energies, and online cosine statistics are safer.

### 5. SAM-Flow

**Base and inputs.** [Confirmed from code] Diffusers FLUX.1-dev and SD3 Medium implementations; source image, source prompt, target prompt, source/target/unchanged tokens, seed/noise, and dynamic attention-derived regions. It is inversion-free and training-free.

**Velocity prediction.** FLUX: [`sam_flow_flux.py:126-139`](https://github.com/chwbob/Sam-Flow/blob/8167978666e6df11e987a453934b89ec422bd3a4/sam_flow/sam_flow_flux.py#L126-L139), `calc_v_flux`, calls `pipe.transformer(...)[0]`. SD3: [`sam_flow_sd3.py:479-519`](https://github.com/chwbob/Sam-Flow/blob/8167978666e6df11e987a453934b89ec422bd3a4/sam_flow/sam_flow_sd3.py#L479-L519), `transformer_forward`/`calc_v_sd3`, calls conditional and unconditional transformer forwards and applies CFG. Outputs match the input generated latent: FLUX `[B,L,64]` under standard packing; SD3 `[B,C,H,W]`.

**Source/target construction.** [`sam_flow_flux.py:566-595`](https://github.com/chwbob/Sam-Flow/blob/8167978666e6df11e987a453934b89ec422bd3a4/sam_flow/sam_flow_flux.py#L566-L595):

\[
z_t^{src}=(1-t)x_{src}+t\epsilon,\quad
z_t^{tar}=z_t^{model}+z_t^{src}-x_{src},\quad
\Delta v=v^{tar}(z_t^{tar},P_t)-v^{src}(z_t^{src},P_s),
\]

\[
z_{edit,next}=z_{edit}+(\sigma_{next}-\sigma)\Delta v.
\]

The SD3 path is equivalent at `sam_flow_sd3.py:812-841`.

**Spatial control.** [Confirmed from code] The repository monkey-patches scaled-dot-product attention during a scout pass, averages selected layers, extracts token maps, builds a time-varying soft/core/ring mask, and accumulates regions over time. FLUX attention capture is at [`sam_flow_flux.py:150-207`](https://github.com/chwbob/Sam-Flow/blob/8167978666e6df11e987a453934b89ec422bd3a4/sam_flow/sam_flow_flux.py#L150-L207). The main update first integrates the full differential velocity, then anchors/model-blends latent states with accumulated masks at [`sam_flow_flux.py:594-618`](https://github.com/chwbob/Sam-Flow/blob/8167978666e6df11e987a453934b89ec422bd3a4/sam_flow/sam_flow_flux.py#L594-L618). Therefore, the released code implements the paper's regional differential-flow idea through masked latent anchoring after the full differential update, rather than a literal `M*Δv` assignment at the prediction line.

**Same-latent A/B/AB feasibility.** [Research inference] Very good with a diagnostic-only branch. `calc_v_flux` and `calc_v_sd3` accept arbitrary latent and conditioning. Pre-encode A/B/AB, select one shared `z_probe` (preferably the main `z_t^{model}` or a clearly documented source-path state), and call each helper without updating. The native source/target tensors are different states and must not be labeled same-latent.

**Minimum change.** Insert a callback before lines 566-592 that receives one explicitly named `z_probe`, evaluates the three prompt embeddings, and saves selected tensors/statistics. Keep scout attention and mask creation frozen or off during the first causal validation. Estimated difficulty: low-to-moderate; attention maps and explicit source/target prompt handling are already present.

### 6. MaskFlow

**Base and inputs.** [Confirmed from code] Custom pipeline around Qwen-Image-Edit-2511 using Diffusers-compatible components; source image, aligned user mask, instruction prompt, random noise, mandatory standard MaskFlow LoRA for reported behavior, optional DMD LoRA, and optional Soft-Poisson refinement. It uses no inversion.

**True velocity semantics.** [`schedulers/flow_matching.py:77-103`](https://github.com/ReyChiaro/MaskFlow/blob/4baac7733c5e33c5ef5c85be36b70631a40441e1/schedulers/flow_matching.py#L77-L103) defines

\[
x_t=(1-\sigma)x_0+\sigma\epsilon,\quad v=\epsilon-x_0,\quad
x_{next}=x_t+(\sigma_{next}-\sigma)v.
\]

Thus [confirmed from code] the trained target and effective prediction are velocity, not epsilon-noise prediction. The mask-aware scheduler changes the probability path/target outside the edit mask and re-anchors the next state; see [`schedulers/mask_flow.py:18-101`](https://github.com/ReyChiaro/MaskFlow/blob/4baac7733c5e33c5ef5c85be36b70631a40441e1/schedulers/mask_flow.py#L18-L101).

**Prediction and CFG.** [`qwenimage_edit_plus.py:321-343`](https://github.com/ReyChiaro/MaskFlow/blob/4baac7733c5e33c5ef5c85be36b70631a40441e1/pipelines/qwenimage/qwenimage_edit_plus.py#L321-L343) calls the transformer and slices generated tokens. [`qwenimage_mask_flow.py:671-784`](https://github.com/ReyChiaro/MaskFlow/blob/4baac7733c5e33c5ef5c85be36b70631a40441e1/pipelines/qwenimage/qwenimage_mask_flow.py#L671-L784) forms positive-mask/null-text/null-mask branches and combines text and mask CFG residuals. [`qwenimage_mask_flow.py:797-869`](https://github.com/ReyChiaro/MaskFlow/blob/4baac7733c5e33c5ef5c85be36b70631a40441e1/pipelines/qwenimage/qwenimage_mask_flow.py#L797-L869) applies optional Soft-Poisson refinement and scheduler anchoring.

**Same-latent A/B/AB feasibility.** [Research inference] Technically good: `denoise_cfg_branch(branch,xt,timestep)` already accepts a shared `xt`. Scientifically, prompt comparison is confounded unless the exact same source encoding and mask conditioning are held fixed. For two spatially distinct edits, using one fixed union mask tests prompt compositionality under a shared region prior; using different A/B masks tests a different variable and cannot isolate text-conditioned velocity decomposition.

**Attention access and minimum change.** No attention tensor hooks are exposed; `attention_kwargs={}` is passed. Add an analysis callback after per-prompt branch prediction and before CFG/Poisson/scheduler transformation. Save the raw positive conditional velocity and the effective post-CFG velocity separately. Difficulty: moderate because raw model, CFG, Soft-Poisson, and scheduler-anchored effective fields are distinct quantities. Hardware and checkpoint burden are high relative to VeloEdit/SAM-Flow.

### 7. IDAttn / Multi-Instance Editing

**Base and inputs.** [Confirmed from code] Custom Diffusers-derived FLUX.1-Kontext-dev pipeline; source image, global/multiple instance instructions, normalized bounding boxes, optional LoRA, and random generated latents. No inversion is used.

**Velocity prediction/update.** [`pipeline_flux_kontext.py:1531-1571`](https://github.com/Blowing-Up-Groundhogs/IDAttn/blob/79ce339b9e7c789fbbed4b9a79f92d5067310abd/kontext/pipeline_flux_kontext.py#L1531-L1571) concatenates generated and source-image condition tokens, calls `self.transformer(...)[0]`, slices generated tokens, applies optional true CFG, and passes the result to `FlowMatchEulerDiscreteScheduler.step`. The result is the effective flow-model output despite the `noise_pred` name. Shape is generated packed-token shape `[B,L,C]`.

**Spatial control and attention.** [`attention_processor_APITA.py:245-300`](https://github.com/Blowing-Up-Groundhogs/IDAttn/blob/79ce339b9e7c789fbbed4b9a79f92d5067310abd/kontext/attention/attention_processor_APITA.py#L245-L300) builds and caches hard/soft full joint-attention masks from instance text spans, image-token indices, and bounding-box position masks, then supplies them to the attention backend. Region control enters velocity estimation through attention, not post-hoc velocity masking. Q/K/V and masks are explicit, but attention probabilities are not returned.

**Same-latent A/B/AB feasibility.** [Research inference] Feasible but less clean. A/B/AB prompts change token spans and possibly bbox/mask structure; the APITA processor has class-level cached masks and a global counter. A controlled evaluator must clear caches between conditions, preserve identical generated/source-image latents and scheduler state, and define whether A/B use the same union bbox set as AB. Otherwise attention-mask differences become an uncontrolled variable. Difficulty: moderate-to-high.

**Minimum change.** Capture raw conditional and post-CFG model outputs immediately before scheduler step; add an explicit cache reset and externally supplied `latents`. Attention probabilities require modifying the processor/backend. No velocity composition is implemented.

## Exact Same-Latent Evaluation Decision

The required controlled quantity is:

\[
\{v_A,v_B,v_{AB}\}=\{f_\theta(z_t,t,c_A,s),f_\theta(z_t,t,c_B,s),f_\theta(z_t,t,c_{AB},s)\},
\]

where `s` denotes every fixed non-text condition: source-image encoding, generated-token packing, mask/bbox policy, guidance convention, model weights/adapters, dtype, timestep mapping, and attention mode.

| Method | Native same latent? | Feasible without retraining? | Main confound to remove |
|---|---:|---:|---|
| FlowDC | No (paper trajectories differ) | Conceptually yes; code unavailable | Target trajectory state |
| SplitFlow | No | Yes, low change | Cumulative prompt and target trajectory state |
| Follow-Your-Shape | No | Yes, moderate change | Inversion/edit states, midpoint, injected features |
| VeloEdit | No first-class API; deterministic single-prompt closures | **Yes, low change** | Prompt-captured closure; later branch trajectories |
| SAM-Flow | No | **Yes, low-to-moderate change** | `z_src` versus `z_tar`; dynamic scout mask |
| MaskFlow | No multi-prompt API | Yes, moderate change | Mask/CFG/Poisson definition |
| IDAttn | No multi-prompt API | Yes, moderate-high change | Prompt-dependent APITA masks and caches |

Important interpretation rule: evaluating three independently sampled trajectories at the same nominal timestep is **not** acceptable. Deterministic seeds alone also do not guarantee same later states after condition-specific updates.

## Attention Access Summary

| Repository | Existing access | Spatial scale | Assessment |
|---|---|---|---|
| SplitFlow | None; Diffusers SD3 call only | Multi-resolution internal SD3 tokens | Moderate modification |
| Follow-Your-Shape | Explicit Q/K/V and stored inversion K/V; fused attention output only | FLUX packed token grid | Good for features, moderate for maps |
| VeloEdit | None | Model-internal | Moderate modification; defer initially |
| SAM-Flow | Existing SDPA hook and token maps over selected layers | FLUX/SD3 image-token grid | **Excellent** |
| MaskFlow | Conditioning masks but no attention capture | Qwen image-token grid across blocks | Moderate modification |
| IDAttn | Explicit Q/K/V and hard/soft attention masks; no returned probabilities | Text + generated image + context-image tokens | Excellent for imposed masks, moderate for probabilities |

## Comparison Table

Ratings refer to suitability for controlled velocity analysis, not output quality.

| Method | Venue / status | Base model / framework | Inversion | Velocity accessible | Same-latent multi-prompt | Mask / region | Velocity composition | Attention access | Modification difficulty |
|---|---|---|---|---|---|---|---|---|---|
| FlowDC | CVPR 2026 | FLUX.1-dev; unreleased code | None / FlowEdit | Paper only: Good conceptually | Poor today: no code | None explicit | Excellent in paper: progressive basis + orthogonal decay | None claimed | Impossible to verify |
| SplitFlow | NeurIPS 2025 | SD3 Medium / Diffusers | None | Good: direct helper | Good after wrapper | None | Good: differential sub-flows + aggregation | Poor | Low |
| Follow-Your-Shape | ICLR 2026 | FLUX.1-dev / custom PyTorch | RF inversion + second-order solver | Good | Moderate after isolation | Velocity-derived dynamic TDM | Difference for localization; no A/B sum | Good K/V, moderate maps | Moderate |
| VeloEdit | arXiv preprint | FLUX Kontext + Qwen / Diffusers | None | **Excellent; already saved** | **Excellent after wrapper** | Dynamic velocity-similarity or GT mask | Preserve projection/residual and interpolation | Poor | **Low** |
| SAM-Flow | arXiv preprint | FLUX.1-dev + SD3 / Diffusers | None / FlowEdit | **Excellent; paired helpers** | **Excellent after diagnostic branch** | Dynamic token-attention soft/core/ring maps | Target-source differential velocity + latent anchoring | **Excellent** | Low-moderate |
| MaskFlow | arXiv preprint | Qwen-Image-Edit-2511 / custom + Diffusers | None | Excellent raw/effective paths | Good with fixed union mask | User mask in path, objective, CFG, scheduler, pixel blend | CFG residuals + Soft-Poisson field refinement | Moderate-poor | Moderate |
| IDAttn | ICML 2026 accepted | FLUX.1-Kontext-dev / custom Diffusers | None | Good | Moderate after cache-safe wrapper | Bounding-box APITA attention masks | None | Good masks/QKV | Moderate-high |

## Recommended Base Repositories

### 1. VeloEdit — primary base

**Why selected.** [Confirmed from code] It already separates model-specific preparation from a generic deterministic Euler sampler, exposes a `v_pred_fn(z,sigma)`, stores each model output before integration, supports two current instruction editors, and CPU-offloads the recorded trajectory. The exact shared-latent experiment requires the smallest conceptual change: make conditioning an explicit argument and evaluate A/B/AB before one update.

**Scientific advantage.** It minimizes intervention by starting from a plain instruction-conditioned model output and permits raw velocity analysis before any VeloEdit replacement/blending. This gives clean causal attribution.

**Limitations.** It lacks source-prompt conditioning, attention hooks, and an explicit code license. FLUX Kontext/Qwen are large models; prompt embeddings and image-conditioning construction differ between backbones. The current code's name `noise_pred` can mislead downstream notes, so analysis artifacts must call it effective model velocity only after documenting scheduler convention.

### 2. SAM-Flow — replication and spatial-analysis base

**Why selected.** [Confirmed from code] It explicitly implements source and target prompt forwards for both FLUX and SD3, exposes the raw transformer outputs in small helpers, and already captures token-level attention maps and builds time-varying spatial masks. This directly supports the later comparison between attention localization and velocity localization.

**Scientific advantage.** A second implementation with a different editing formulation tests whether any observed decomposability is backbone/pipeline-specific. Source-prompt access is cleaner than in instruction-only VeloEdit.

**Limitations.** Native differential velocities compare different latent states; dynamic scout masks and source anchoring must be excluded or frozen in the first shared-latent diagnostic. The global monkey-patch of `torch.nn.functional.scaled_dot_product_attention` is brittle and should not be active during pure velocity extraction unless attention is the controlled variable.

### Why the others are not the first base

- FlowDC is the strongest conceptual match but has no verifiable code.
- SplitFlow is a useful SD3 baseline, but its small release couples cumulative prompt generation to independently evolving states and has no attention/spatial analysis support.
- Follow-Your-Shape entangles inversion, second-order midpoint states, TDM construction, KV injection, and ControlNet, making causal isolation harder.
- MaskFlow is excellent for studying mask-conditioned transport but changes the probability path and requires method LoRA weights; it would answer a different question unless the mask is rigorously fixed.
- IDAttn is valuable for multi-instance failure analysis but explicitly changes attention connectivity per instruction/region; it is better as a later mechanistic comparator than as the cleanest velocity baseline.

## Initial A/B/AB Experiment Feasibility

For VeloEdit and SAM-Flow, no retraining is required. A defensible first experiment should:

1. Choose one source image and define semantically independent instructions `P_A`, `P_B`, and `P_AB` without changing formatting conventions.
2. Encode the source image once and cache the exact condition-image tensor/IDs.
3. Generate one initial noise tensor and one scheduler/sigma sequence.
4. Follow one **reference trajectory** only (recommended: AB, plus a source/no-edit reference in a sensitivity check).
5. Before updating at selected timesteps, copy the same `z_t` and evaluate A, B, and AB with identical non-text conditions.
6. Save raw conditional and effective post-guidance velocities separately; do not mix pre-CFG and post-CFG tensors.
7. Fit scalar global coefficients by least squares, NNLS, and cosine projection only as analysis tools:

\[
\hat v_{AB}=\alpha_Av_A+\alpha_Bv_B.
\]

8. Report `R^2`, normalized residual, pairwise cosine, regional energy/locality, and leakage at each selected timestep.
9. Repeat with the A trajectory and B trajectory as `z_t` carriers. If results change materially, decomposability is state-dependent and cannot be summarized by timestep alone.

The first probe should not apply VeloEdit interventions, SAM-Flow masks/anchors, attention manipulation, or the proposed new decomposition method. It should characterize the pretrained conditional vector field.

## Open Technical Questions and Risks

1. **Condition definition:** FLUX.1-Kontext and Qwen image editors consume instructions plus image context, whereas FlowEdit-style methods use source and target descriptive prompts. `v_A` is not automatically comparable across these semantics.
2. **Carrier-state dependence:** Results may differ strongly depending on whether shared `z_t` comes from source, A, B, AB, or an unconditional trajectory. One carrier is insufficient for a general conclusion.
3. **CFG semantics:** Guidance changes velocity magnitude and direction nonlinearly through conditional/unconditional mixing and, in Qwen code, norm rescaling. Raw conditional and post-CFG fields must be distinguished.
4. **Timestep convention:** Repositories mix normalized `t`, training timestep `t/1000`, and shifted sigma. Comparisons must store both model input timestep and integrator sigma.
5. **Representation:** Packed token channels are not pixel RGB directions. Spatial maps require documented unpacking and VAE scale; channelwise locality has no direct semantic interpretation.
6. **Mask/bbox confounding:** Changing masks between A, B, and AB changes conditioning and possibly the transport path. Fixed union-mask and per-object-mask experiments answer different questions.
7. **Attention caching:** IDAttn uses class-level mask caches/counters. Cross-prompt probes can silently reuse invalid masks unless reset.
8. **Solver stage:** Follow-Your-Shape has current and midpoint velocities. Comparing one method's raw current-step output with another's solver-averaged field would be invalid.
9. **Raw versus effective field:** MaskFlow's Soft-Poisson output and VeloEdit/SAM interventions are transformed fields. Both raw model velocity and final integrated field should be named and stored separately.
10. **Memory:** A full `[B,L,C]` or `[B,C,H,W]` tensor for A/B/AB at every step can be large. Start with 4-6 timesteps, BF16/FP16 CPU offload, and online norms/cosines; retain full tensors only for selected cases.
11. **Licensing/reproducibility:** VeloEdit and IDAttn have no code license at inspected commits; SplitFlow claims MIT but lacks the referenced license file. This may constrain redistribution of modifications even though local research inspection is possible.
12. **Model access/hardware:** FLUX.1-dev/Kontext and SD3 Medium may require gated access; high-resolution transformer inference remains GPU-memory intensive even without training.
13. **FlowDC reproducibility:** The closest paper-level comparator cannot be code-validated until an official release appears.

## Recommended Next Experimental Step

Create a minimal project-owned analysis script for **VeloEdit FLUX.1-Kontext only** that performs no intervention and does not modify `external/VeloEdit/`:

- load the official repository as a reference dependency;
- cache one source-image encoding, one seed/noise tensor, and one shifted sigma schedule;
- encode `P_A`, `P_B`, and `P_AB` once;
- use the AB trajectory as the initial carrier, probing the exact same `z_t` at approximately early/mid/late sigmas (for example 0.9, 0.7, 0.5, 0.3, 0.1, mapped to actual schedule entries);
- save post-guidance velocities to CPU in BF16/FP16 plus metadata and compute online scalar similarities;
- assert bitwise identity of the latent, timestep, image-context tensor, scheduler configuration, and model weights passed to all three calls;
- run one small image/instruction case before any benchmark-scale study.

Then replicate the identical protocol in SAM-Flow FLUX using its source/target prompt interface and with mask/attention anchoring disabled. Agreement across the two formulations would raise confidence that the measured structure belongs to the conditional flow field rather than one editor's intervention logic.

This survey ends here. No object-centric decomposition method was implemented.
