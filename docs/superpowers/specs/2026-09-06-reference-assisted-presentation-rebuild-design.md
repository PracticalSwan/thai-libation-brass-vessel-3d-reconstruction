# Reference-Assisted Presentation Rebuild Design

Date: 2026-09-06

## Goal

Produce a complete, presentation-quality Thai libation brass vessel model without misrepresenting invented geometry as direct photogrammetry. Preserve all Steps 1-17 evidence and stop before manual Blender cleaning/sculpting.

## Problem boundary

The Step 17 asset is a truthful 73-view local reconstruction, but its 21 connected components and missing regions make it unsuitable as the final visual model. Directly filling those gaps would blur the distinction between measured photogrammetry and inferred geometry.

The rebuild therefore creates a separate asset under `reconstruction/reference_assisted/`.

## Inputs

Authoritative inputs are:

- immutable photographs under `IMG20260826122949/`;
- reviewed segmentation masks and `ml_dataset/manifest.csv`;
- existing visual/shape measurements for orientation and form review;
- Blender 5.2 for deterministic generation, material setup, rendering, and export.

The existing local dense/Poisson/textured outputs are comparison evidence only and are never source geometry for this phase.

## Method

1. Verify hashes of eight reviewed normal-side masks.
2. Extract a centered half-width silhouette profile from each mask.
3. Aggregate side profiles with a median and small symmetric smoothing filter.
4. Measure a warm-metal color reference from masked source photographs while excluding extreme highlights/shadows.
5. Build explainable radial profiles for the receiving bowl, water vessel, and lid, scaling their radius from the measured external width/height ratio.
6. Revolve those profiles around the vertical axis to create closed main forms.
7. Add separate closed geometry for the finial, bands, repeated visible/inferred ornament, and chain links.
8. Use a photograph-referenced metallic brass material with procedural roughness/bump instead of baking view-dependent highlights into albedo.
9. Export an editable `.blend` and portable `.glb`.
10. Render fixed beauty views plus a primary-form orthographic silhouette.
11. Validate all mesh objects for manifold edges and compare the primary silhouette with the aggregate reviewed-mask silhouette.

## Inference disclosure

The complete model deliberately uses inference where direct image evidence is incomplete:

- rotational symmetry completes unseen backsides of the manufactured round forms;
- repeated decoration is propagated around hidden surfaces from visible motif patterns;
- physical scale remains relative because the capture set contains no defensible real-world measurement;
- procedural material microstructure approximates brass appearance but is not a scanned PBR texture.

These inferences belong only to the presentation asset. They do not alter or replace measured reconstruction evidence.

## Acceptance gates

The pre-clean phase is accepted only when:

- the integrated generator finishes successfully;
- the `.blend`, `.glb`, and four beauty renders exist and are non-empty;
- every generated mesh object has zero non-manifold edges;
- the primary-form reference silhouette IoU is at least `0.70` after common height normalization;
- final beauty renders are visually inspected from front, quarter, side, and top-oblique views;
- `manual_cleanup_started=false` remains explicit in machine-readable reports;
- the complete project test suite passes.

## Stop boundary

No manual sculpting, hand retopology, Boolean cleanup, manual hole filling, manual deletion, or subjective mesh repair occurs in this phase. Those operations require a later explicitly authorized Blender-cleaning phase.
