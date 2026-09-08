# Research Context

## 1. Research Area

This project studies image editing based on Flow Matching / Rectified Flow models.

The current research focus is:

**object-centric velocity field decomposition for complex image editing.**

The goal is not to assume that a complex editing velocity field can be strictly separated by hard spatial masks.

Instead, we want to investigate whether complex editing velocity fields exhibit:

- object-dominant structure,
- spatially structured dynamics,
- compositionality,
- timestep-dependent locality,
- cross-object interaction components.

---

## 2. Core Scientific Question

Given a complex edit involving multiple objects or sub-instructions, does the resulting velocity field contain intrinsic object-wise structure?

A working formulation is:

\[
v_{\text{complex}}(x,t)
=
\sum_i w_i(x,t) v_i(x,t)
+
v_{\text{interaction}}(x,t)
+
r(x,t)
\]

where:

- \(v_i\) represents a velocity component associated with object or edit \(i\),
- \(w_i(x,t)\) is a soft and potentially timestep-dependent spatial support,
- \(v_{\text{interaction}}\) represents interaction, relation, or context coupling between objects,
- \(r\) represents unexplained residual.

We DO NOT assume:

\[
v_{\text{complex}}
=
v_A + v_B
\]

is always strictly valid.

We want to experimentally determine when such approximations hold and when they fail.

---

## 3. Main Research Questions

### RQ1. Intrinsic object-wise decomposability

Does a pretrained Flow Matching image editor naturally produce object-dominant velocity components?

For a complex edit AB:

\[
v_{AB}
\]

compare it against independently constructed:

\[
v_A,\quad v_B.
\]

Measure whether:

\[
v_{AB}
\approx
\alpha_A v_A + \alpha_B v_B.
\]

---

### RQ2. Spatial locality

Does an object-specific editing direction have stronger velocity energy inside the corresponding object region?

For object mask \(M_A\):

\[
E_A^{inside}
=
\|M_A \odot v_A\|
\]

versus:

\[
E_A^{outside}
=
\|(1-M_A)\odot v_A\|.
\]

Strict locality is NOT assumed.

We are interested in object-dominant rather than perfectly object-local velocity.

---

### RQ3. Cross-object leakage

For edits A and B, quantify whether:

\[
v_A
\]

produces substantial energy or semantic effect in object B's region and vice versa.

This may reveal velocity entanglement in complex editing.

---

### RQ4. Timestep-dependent decomposability

Object-wise structure may change over the Flow Matching trajectory.

Study:

\[
D_{\text{object}}(t),
\quad
R^2_{\text{recon}}(t),
\quad
\text{locality}(t),
\quad
\text{residual}(t).
\]

Potential hypothesis:

- early steps: stronger global / semantic coupling,
- middle steps: object structure becomes clearer,
- late steps: more localized appearance refinement.

This is a hypothesis only and must be verified experimentally.

---

### RQ5. Meaning of the residual

Given:

\[
r
=
v_{AB}
-
\alpha_A v_A
-
\alpha_B v_B,
\]

do NOT automatically interpret \(r\) as fitting error.

Investigate whether it contains:

- object-object interaction,
- spatial relation information,
- boundary effects,
- illumination / context coupling,
- occlusion effects,
- model approximation noise.

A more expressive model may be:

\[
v_{\text{complex}}
=
\sum_i v_i^{obj}
+
\sum_{i<j}v_{ij}^{interaction}
+
r.
\]

---

## 4. Important Related Directions

The following recent work is highly relevant.

### SplitFlow

Relevant because it decomposes complex editing at the semantic / prompt level and combines multiple flow components.

Key distinction:

SplitFlow studies semantic decomposition.

Our question is whether velocity fields contain intrinsic object-wise / spatial structure.

---

### FlowDC

Relevant because it studies complex editing through sub-edit decomposition and geometric velocity projection / decomposition.

Key distinction:

FlowDC does not systematically study whether sub-edit velocity components spatially correspond to individual objects.

---

### Follow-Your-Shape

Relevant because velocity differences along editing / inversion trajectories contain spatial localization information.

Important implication:

velocity fields contain non-trivial spatial information.

---

### VeloEdit

Relevant because it separates source-preserving and editing velocity effects and uses velocity discrepancy to identify editable regions.

Key distinction:

its regional decomposition is mainly edit-versus-preserve rather than multi-object decomposition.

---

### SAM-Flow

Relevant because it applies differential velocity only within dynamically estimated editing regions and anchors the source elsewhere.

Key distinction:

SAM-Flow uses a mask to control velocity.

Our research asks whether object structure can be discovered or characterized from the velocity itself.

---

### MaskFlow

Relevant because regional constraints are incorporated into the Flow Matching formulation itself.

Important implication:

different image regions may require different transport behavior.

---

### Multi-instance Flow Matching Editing

Relevant because multi-instance editing exposes semantic interference and entangled velocity estimation.

Important question for this project:

whether such entanglement can be characterized directly in velocity space.

---

## 5. Current Research Position

Do NOT start by proposing a new decomposition algorithm.

The current priority is:

**characterization before method design.**

First determine whether object-wise velocity structure exists.

Then determine:

1. when it exists,
2. when it fails,
3. whether failures are caused by interaction terms,
4. whether this structure can improve complex editing.

---

## 6. Initial Experimental Quantities

Useful metrics include:

### Reconstruction

\[
R^2_{\text{recon}}(t)
\]

and normalized residual:

\[
\frac{
\|v_{AB} - \hat v_{AB}\|
}{
\|v_{AB}\|
}.
\]

---

### Locality

For object A:

\[
L_A(t)
=
\frac{
\|M_A \odot v_A(t)\|
}{
\|v_A(t)\|
}.
\]

---

### Cross-object leakage

\[
C_{A\rightarrow B}(t)
=
\frac{
\|M_B \odot v_A(t)\|
}{
\|v_A(t)\|
}.
\]

---

### Direction similarity

\[
\cos(v_{AB}, v_A)
\]

\[
\cos(v_{AB}, v_B)
\]

and region-wise versions of the same quantity.

---

### Residual ratio

\[
R_r(t)
=
\frac{\|r(t)\|}{\|v_{AB}(t)\|}.
\]

---

## 7. Experimental Categories

Eventually evaluate at least:

1. single-object attribute editing,
2. two-object independent attribute editing,
3. object replacement,
4. geometric / structural editing,
5. spatial relation editing,
6. overlapping or occluded objects,
7. edits involving shared lighting / reflections / context.

These categories should be treated separately because decomposability may differ significantly.

---

## 8. Current Priority

The immediate task is NOT implementation.

The immediate task is to survey existing Flow Matching image editing repositories and determine which implementation is best suited for controlled velocity analysis.

The preferred base pipeline should allow clean access to:

\[
v_{\text{source}}(t),
v_{\text{target}}(t),
v_A(t),
v_B(t),
v_{AB}(t)
\]

under controlled latent / noise conditions.

The implementation should ideally require minimal modification.