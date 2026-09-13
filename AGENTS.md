# Project Agent Instructions

## Scope and objective

This repository is the CSX4213 Computer Vision project for reconstructing a real Thai brass libation vessel from photographs.

Current phase: **V4 dense reconstruction/post-fusion acceptance.** V1, V2, and V3 were visually rejected and their reconstruction/model artifacts have been removed. Do not restore or continue those versions unless the user explicitly asks for historical recovery.

The immutable incoming V4 source set is `CSX4213_Project_V4_Images/` (688 JPEGs). The approved planning documents classify it as 158 uncoated appearance/reference images, 107 empty-board/background images, and 423 object-bearing geometry images. Do not rename, move, recompress, rotate, overwrite, or otherwise modify those files. The current objective is to finish the already-running V4 pipeline from the newest verified state: correct/re-evaluate dense acceptance evidence -> accept the best real mask-constrained fused cloud when the essential visual/structural gates pass -> one preserved raw Poisson mesh -> Blender conservative cleanup/material/export -> final `.blend` and `.glb` -> fresh GLB re-import verification.

## Protected material

- `IMG20260826122949/` is historical raw evidence. Keep it immutable unless the user explicitly authorizes its deletion.
- `preprocessing/` and existing analysis material are historical/reusable references; never mix their old derived images into V4.
- Preserve reusable source such as OpenCV geometry utilities, ALIKED/LightGlue code, COLMAP/PLY helpers, and ML/segmentation utilities until V4 replacements are working.
- `.codegraph/` is local code-intelligence state. Maintain it when needed, but never publish its database, daemon state, sockets, or logs.
- Never publish private checkpoints from `analysis/ml/checkpoints/`.

## V4 capture boundary

The final incoming V4 images already exist in `CSX4213_Project_V4_Images/`. Treat that directory as an immutable source boundary. Store manifests and derived V4 products under `capture_v4/` and reconstruction products under `reconstruction/v4/`; do not duplicate or relocate the raw set unless a later implementation need is evidence-backed.

The authoritative V4 geometry capture is **fixed-camera turntable/object-rotation photogrammetry on a white background**. During each capture pass the camera stays completely stationary and the vessel rotates through one full 360-degree circle. The camera may be repositioned only between complete circles.

The final media contains six object-bearing geometry passes spanning lower/horizontal through upper/steep-high coverage; no recapture is possible or required by this project state. The audited EXIF shows one 3072x4080 OPPO Reno12 F rear-lens state with 3.98 mm / 26 mm-equivalent focal length and digital zoom 1 throughout, while the geometry/empty sequences use consistent ISO 100, 1/100 s, and manual-exposure/WB metadata. Use the actual sequence model in the approved spec/plan rather than inventing idealized ring counts or angles.

Because the white background is stationary while the object rotates, **background features are invalid reconstruction features**. Vessel segmentation/masking is mandatory before ALIKED/LightGlue and sparse SfM, and dense reconstruction must also exclude the stationary background. Do not rely on simple white-threshold segmentation because the temporary matte coating may be similar in color to the background.

For geometry capture, use the removable Caring Easy dry-shampoo coating to suppress brass specularity and add non-repeating random black marker spots preferably on the coating, not bare brass. Vary spacing, avoid grids/periodic patterns and oversized blobs, and distribute features across the bowl, globe, neck, lid/finial, pedestal, and transition areas. These marks are texture for correspondence only; never turn them into fabricated geometry.

The final source includes 107 empty-board/background frames and 158 uncoated appearance/reference frames. G8 and G9 contain same-setup rotating empty-board tails that are approved as negative refinement evidence near the base; use G6 only when camera/framing compatibility is proven. Use the uncoated set for final brass material work and never use the coated/marker geometry frames as the final material source.

## V4 reconstruction route

Use one strong primary pipeline. Do not schedule algorithm bake-offs or alternate reconstruction stacks.

1. **Ingest/minimal QA**: decode stills/video, extract frames where needed, inspect camera metadata, resolve orientation, and reject only unreadable or catastrophic blur/exposure failures. FFmpeg and ExifTool are preferred ingest/metadata tools when available.
2. **Object isolation**: use pretrained Grounding DINO-T to obtain the vessel ROI and SAM 2.1 Hiera-small for the full-resolution mask. Use SAM 2 temporal propagation for ordered rings/video when useful. An empty-background reference is an additional deterministic refinement cue, not a competing segmentation method.
3. **Conservative preprocessing + learned matching**: preserve image geometry, then use only ALIKED-N16Rot + LightGlue. Discard every learned keypoint outside the object feature mask. Build acquisition-aware pairs from nearby angular views, wider local neighbors, orbit closure, and corresponding phases across elevation rings.
4. **Sparse reconstruction**: import learned correspondences into COLMAP, run geometric verification, then pyCOLMAP incremental SfM + bundle adjustment. Share intrinsics only for images with the same real camera/lens/settings/resolution contract. Determine the actual camera model from V4 metadata/runtime, with `SIMPLE_RADIAL` only as the initial likely model.
5. **Dense reconstruction**: undistort the accepted sparse model and its masks consistently, run CUDA COLMAP PatchMatch with geometric consistency, then **geometric** stereo fusion with the object masks so stationary background cannot enter the fused cloud. The only resource fallback is reducing dense max image size from 2000 to 1600 on CUDA OOM.
6. **Mesh**: use Poisson reconstruction only. Preserve the fused cloud and raw Poisson mesh unchanged until direct visual inspection; remove only unsupported floaters/noise afterward.
7. **Blender finalization**: inspect the real raw mesh first; then clean, smooth/relax conservatively, repair only defensible holes, use controlled voxel-remesh/shrinkwrap only when needed, create a practical production mesh, UV/detail bake, build brass PBR from uncoated references, and export an editable Blender master plus working GLB.

## Current dense acceptance direction

- The accepted sparse reconstruction has already reached full selected-view registration and the active work is dense/post-fusion acceptance. Do not restart ingest, segmentation, matching, or sparse SfM unless new evidence identifies an upstream defect.
- Before any new PatchMatch rerun, correct the cross-camera depth-continuity metric in `v4_postfusion.py::_depth_pair_consistency`: after reprojecting a reference point into the source camera, compare the sampled source depth map against the **source-camera Z returned by `_project_points`**, not the original reference-camera depth value. Add a focused regression with distinct camera poses, then re-run the ring audit on the existing true3 geometric depth maps and corrected masks. Metric re-evaluation alone does not justify PatchMatch recomputation.
- COLMAP StereoFusion mask resolution must follow `<image_name>.png`; for a source named `foo.jpg`, the fusion mask is `foo.jpg.png`. Preserve the corrected 372/372 resolvable-mask contract.
- `mean_consistent_fraction_at_1pct >= 0.50` is an implementation diagnostic, not an authoritative project requirement. Do not let that numeric threshold alone permanently block progress when the corrected evidence and direct visual inspection show a real, finite/rank-3, recognizable vessel cloud without dominant board/background contamination.
- Use corrected cross-ring depth evidence to localize genuine discontinuities. If a real geometry break is visible, repair only the smallest affected transition/chunks. Do not restart the full 2000-pixel dense reconstruction merely to chase a diagnostic score.
- Once the corrected-mask dense cloud passes the essential contamination/identity gate, advance immediately to **one** preserved raw Poisson mesh and inspect fused cloud + raw Poisson from front/quarter/side/top-oblique. If that raw geometry is viable, proceed directly to Blender; do not add more audit infrastructure unless it protects a concrete failure mode.
- Preserve every accepted dense candidate/workspace/map/hash and the raw Poisson once created. Never overwrite prior evidence merely to produce a cleaner report.

## Time policy

The user wants speed. Skip broad parameter sweeps, broad historical regression suites, duplicate reports, and professor-facing documentation unless explicitly requested.

Keep only essential gates that prevent wasted compute or a broken final asset:

- new V4 media decode correctly;
- masks preserve the complete vessel;
- sparse virtual cameras form coherent orbital/ring trajectories with cross-ring connectivity;
- the real raw dense cloud/Poisson mesh is visually plausible before Blender cleanup;
- the final Blender master opens correctly;
- the exported GLB cleanly re-imports.

## CodeGraph navigation

- At the start of architecture-sensitive V4 work, call the available code-intelligence status surface first.
- After a large cleanup/refactor or when the index is stale, rebuild from the repository root with `codegraph index --force --quiet .`.
- Use CodeGraph structural search first for symbols, callers/callees, dependency paths, change impact, reusable helpers, and blast radius.
- Use direct guarded file reads/searches for exact literals, generated paths, configuration values, and content that may be stale or unsupported by the index.
- Before modifying an existing V4-reusable module, inspect its structural callers/tests and then its exact source. CodeGraph is navigation evidence, not runtime proof.
- `.codegraph/` must remain Git-ignored.
- Do not use Codex CLI to run CodeGraph or any other task.

## Core engineering rules

- Prefer the smallest coherent implementation and reuse existing code before adding dependencies.
- Preserve camera/lens-model consistency; do not mix raw distorted pixels with pinhole renders.
- Do not fabricate geometry, poses, metrics, screenshots, or reconstruction results.
- Show the real raw reconstruction before manual Blender beautification.
- No synthetic/reference-modeled geometry is allowed before the raw reconstruction is inspected and accepted as a viable basis.
- No Codex CLI agent/AI execution is permitted. Use the exposed ChatGPT integrations/tools.
- Git milestone authorization: the user explicitly authorizes a focused commit and push to the current configured upstream branch whenever a **major verified V4 milestone** is reached/completed (for example: dense acceptance, raw Poisson acceptance, Blender model acceptance, final export/re-import completion, or another comparably large completed step). Before each milestone commit, inspect branch/upstream/status and the intended diff, stage only milestone-related files, exclude secrets/temp/unrelated user-owned work, use a descriptive commit message, push normally, and verify the remote state. Do not create noisy commits for minor edits/checkpoints. Do not create branches/tags/releases, rewrite history, or force-push unless separately authorized.

## Session startup

1. Read this file and `CLAUDE.md`.
2. Read `LESSONS.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md` for substantive continuation.
3. Inspect current workspace/Git state and preserve unrelated user-owned changes.
4. Verify CodeGraph is current and use it for structural navigation before changing reusable V4 source.
5. Read the approved V4 spec/plan and continue from the **newest verified checkpoint**, not from stale unchecked boxes. At the current checkpoint, dense metric correction/re-evaluation precedes any new PatchMatch work; once dense is visually acceptable, move to Poisson and Blender without reopening completed upstream stages.

## Completion standard

V4 is complete only when the fixed-camera turntable capture has produced a visually acceptable full sparse+dense reconstruction, one preserved raw Poisson mesh, a cleaned scan-derived Blender model, a working canonical `.blend`, and a final GLB that cleanly re-imports for inspection.
