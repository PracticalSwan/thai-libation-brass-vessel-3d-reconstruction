# Thai Libation Brass Vessel 3D Reconstruction

Computer Vision coursework project for reconstructing a real Thai brass libation vessel from photographs.

## Current status — V4 best-defensible reconstruction complete

V1 and V2 were rejected by the professor, and V3 was visually rejected after Blender inspection. V4 is now completed end-to-end under the project’s completion-first policy: every final artifact remains derived from the project capture/reconstruction evidence, strict failures remain explicit, and Blender was not used to fabricate missing anatomy.

The immutable V4 source is `CSX4213_Project_V4_Images/`: **688 JPEGs = 158 uncoated appearance/reference + 107 empty-board/background + 423 coated/marked object-bearing geometry images**. Historical photographs in `IMG20260826122949/` remain preserved as historical evidence and are not V4 reconstruction input.

## Final verified V4 state

- Sparse: best-defensible 372-view v47 lineage in `best_defensible_sparse_v2.json`, source-model SHA-256 `b1c4f142…23922e`; strict mask-projection failures remain explicit.
- Dense: selected 372-view/2000px CUDA geometric PatchMatch cloud SHA-256 `4a596f56…c7d73`, with exact one-reference-one-write provenance. The bounded 3072px `geo_g12` recovery completed 37/37 views but did not improve the unresolved upper anatomy, so it remains comparison evidence.
- Poisson: selected depth-13/trim-5 scan-derived shell SHA-256 `33fe1f6e…d6941`, dominant face fraction `0.9925223`, second-largest `0.0007196`. The narrow-finial and major-hole anatomy checks remain failed and are not hidden.
- Blender: versioned final master `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v6/Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5_FINAL.blend`, SHA-256 `622a676a…3a273`. Raw Poisson and CleanHigh remain preserved and hidden; LOD0 remains `151,547` vertices / `294,713` faces with unchanged geometry/transforms.
- Appearance: reproducible BaseColor/Roughness from the complete **158-image uncoated project set**, scan-derived 2048px AO and CleanHigh→LOD0 tangent normal/detail, metallic brass response, and `photographic_projection_verified=false`. No coated-scan vertex colors, hand-picked reference, or artist-authored texture are used in the final export.
- Export: versioned GLB SHA-256 `15ca1f76…e43de`. A fresh factory-empty Blender 5.2 re-import contains exactly one final mesh, valid UV/material/textures/normals, no camera/light/debug/source objects, and no exported vertex colors.
- Visual QA: eight authoring views and eight fresh-reimport views are materially equivalent (mean 8-bit pixel MAE ~`0.000637`, minimum PSNR ~`78.56 dB`). No duplicate/z-fighting remains. The scan-derived upper neck/lid/finial defects remain visible and documented; strict anatomical acceptance is **not** claimed.
- Canonical outputs `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend` and `.glb` are promoted from the verified v6 bytes. The previous canonical hashes and rollback Git commit are recorded in `canonical_promotion_report.json`.

The final deliverable is therefore the strongest defensible genuine CV reconstruction recoverable from the captured evidence, not a manually repaired or reference-assisted idealization.

## V4 target

Produce one complete working 3D asset from the new photographs using one fixed pipeline, with a deliberate execution handoff after Poisson:

```text
new immutable fixed-camera turntable capture
-> FFmpeg/OpenCV ingest + ExifTool metadata + minimal QA
-> Grounding DINO-T + SAM 2.1 full-resolution vessel masks
-> conservative geometry-preserving preprocessing
-> ALIKED-N16Rot + LightGlue feature matching
-> COLMAP geometric verification
-> pyCOLMAP incremental SfM + bundle adjustment
-> image + mask undistortion
-> CUDA COLMAP PatchMatch geometric consistency
-> mask-aware geometric stereo fusion
-> Poisson raw mesh
-> PRE-BLENDER FREEZE + HASHED HANDOFF
   [Codex/local executor stops here]
-> Blender raw-mesh inspection
-> conservative cleanup / production mesh / UV / detail bake
-> brass PBR material from uncoated references
-> editable Blender master + final GLB
-> clean GLB re-import verification
   [ChatGPT + Blender MCP owns these stages]
```

V4 intentionally uses this one route. It does not schedule SIFT-vs-ALIKED, VGGT-vs-COLMAP, dense-method, or meshing bake-offs. A resource-only retry such as reducing PatchMatch from 2000 px to 1600 px is allowed when the same chosen algorithm hits VRAM limits.

## Authoritative V4 capture

The final acquisition is a **white-background fixed-camera turntable/object-rotation capture**. The audited set contains six object-bearing passes spanning lower/horizontal through upper/steep-high coverage, plus rotating empty-board sequences and a separate uncoated appearance/reference set. The geometry/empty sequences report one locked 3072x4080 OPPO Reno12 F rear-lens state at 3.98 mm / 26 mm-equivalent, digital zoom 1, ISO 100, 1/100 s, with manual-exposure/WB metadata.

The dry-shampoo coating reduced but did not eliminate brass reflections. Random black markers provide useful non-periodic texture across most of the bowl, globe, neck, pedestal, and transitions; finial/rim/interior/base-contact regions remain higher risk.

The white cloth background has visible stationary folds/seams, while the wooden board rotates with the vessel and is highly feature-rich. Board leakage is therefore a first-class SfM/MVS failure mode. Grounding DINO-T + SAM 2.1 is the authoritative isolation route; G8/G9 same-setup empty-board tails are used as negative refinement evidence near the base. Simple white-threshold segmentation is not the V4 plan, and no recapture is expected or requested.

## Fast execution policy

For the current V4 repair, continue from the newest verified checkpoint in the audited V4 implementation plan. After a restart, inspect persisted compute evidence before rerunning anything. Skip broad historical regression suites, algorithm comparisons, large reports, and parameter sweeps. Keep only the checks that prevent wasting reconstruction time:

1. V4 media decode correctly;
2. masks preserve the complete vessel;
3. sparse virtual camera rings/points are coherent and cross-elevation registration is connected;
4. the real raw dense cloud/Poisson mesh is visually plausible before cleanup and the pre-Blender handoff is hash-bound;
5. **post-Blender:** the final Blender master opens correctly;
6. **post-Blender:** the exported GLB cleanly re-imports.

## Retained reusable code

The project keeps capture-independent OpenCV QA/geometry utilities, ALIKED/LightGlue feature and database helpers, pyCOLMAP/COLMAP orchestration, PLY/mesh helpers, and segmentation utilities for V4. Legacy V3/Step-specific assumptions must be generalized rather than reused blindly.
