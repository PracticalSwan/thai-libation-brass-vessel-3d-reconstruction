# Thai Libation Brass Vessel 3D Reconstruction

Computer Vision coursework project for reconstructing a real Thai brass libation vessel from photographs.

## Current status — V4 scan-derived final model complete

V1 and V2 were rejected by the professor. V3 was also visually rejected after a full 266-view learned-feature photometric reconstruction was inspected in Blender. Their active reconstruction/model artifacts have been removed.

The final V4 source is immutable `CSX4213_Project_V4_Images/`: **688 JPEGs = 158 uncoated appearance/reference + 107 empty-board/background + 423 coated/marked object-bearing geometry images**. The fixed-camera turntable pipeline reached a complete, scan-derived Blender master and GLB. Historical photographs in `IMG20260826122949/` remain preserved as evidence/reference only and are not V4 reconstruction input.

## Verified V4 completion

- The corrected `_depth_pair_consistency` compares sampled source geometric depth with the reprojected source-camera Z. The distinct-camera-pose regression and the focused post-fusion suite pass (`7 passed`); the existing true3 maps were re-audited with the corrected 372/372 `<image_name>.png` masks and PatchMatch was not rerun.
- All 28 selected cross-ring measurements were finite/measured across seven transitions. The 0.50-at-1% value remains a diagnostic (four transitions are below it), while the authoritative corrected contamination/identity evidence passes: no dominant board slab, cloth/background structure, or pedestal-board webbing, and vessel identity confirmed.
- The accepted fused candidate is `reconstruction/v4/dense/fused_workspace_tiled6_g8g9_priority_true3_maskfix.ply` (SHA-256 `df05019e2e56d1c54351f4b2cee161cbcfb7b782a3c6b0d4920d6e303f39d7d8`, 1,426,482 finite rank-3 colored points). One preserved raw Poisson mesh is `reconstruction/v4/mesh/poisson_raw.ply` (SHA-256 `55a4e92c9491cced508360d1645355ed447785f175df0556b400528fa9522941`, 887,770 vertices / 1,669,931 faces).
- Blender keeps `SM_V4_Poisson_Raw` and its rollback copy, a hidden scan-derived `SM_V4_Scan_CleanHigh`, and active `SM_V4_Vessel_LOD0` (172,851 vertices / 299,324 faces). Cleanup removed only 622 detached components satisfying both `<=100` faces and `<=0.0035` maximum world dimension; voxel/remesh alternatives that lost measured extent were rejected. LOD0 has one finite angle-based UV map and a packed 1024² AO detail bake.
- `MAT_V4_Brass` uses a robust median from uncoated reference `IMG20260912132007.jpg` (source SHA-256 `393C8FDCFBB025EA3D58649AB2E6314BF2D6E7185C3ACEA6744B8055AC959F3B`), with the AO map driving restrained roughness variation. The canonical files are [`Thai_Libation_Vessel_V4_FINAL.blend`](reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend) and [`Thai_Libation_Vessel_V4_FINAL.glb`](reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb); their fresh factory-startup GLB re-import passes mesh/material/texture/bounds/normals/export-scope checks and produces four inspected views under `reconstruction/v4/previews/final_glb_reimport_v1/`.

The geometry remains an honest scan result rather than a reference-modeled replacement: several bowl/stem regions are porous or missing because the accepted dense evidence does not support them. Absolute physical scale is also unverified; the saved master records a consistent normalized scale and these limitations.

## V4 target

Produce one complete working 3D asset from the new photographs using one fixed pipeline, then finish the accepted real reconstruction in Blender:

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
-> Blender raw-mesh inspection
-> conservative cleanup / production mesh / UV / detail bake
-> brass PBR material from uncoated references
-> editable Blender master + final GLB
-> clean GLB re-import verification
```

V4 intentionally uses this one route. It does not schedule SIFT-vs-ALIKED, VGGT-vs-COLMAP, dense-method, or meshing bake-offs. A resource-only retry such as reducing PatchMatch from 2000 px to 1600 px is allowed when the same chosen algorithm hits VRAM limits.

## Authoritative V4 capture

The final acquisition is a **white-background fixed-camera turntable/object-rotation capture**. The audited set contains six object-bearing passes spanning lower/horizontal through upper/steep-high coverage, plus rotating empty-board sequences and a separate uncoated appearance/reference set. The geometry/empty sequences report one locked 3072x4080 OPPO Reno12 F rear-lens state at 3.98 mm / 26 mm-equivalent, digital zoom 1, ISO 100, 1/100 s, with manual-exposure/WB metadata.

The dry-shampoo coating reduced but did not eliminate brass reflections. Random black markers provide useful non-periodic texture across most of the bowl, globe, neck, pedestal, and transitions; finial/rim/interior/base-contact regions remain higher risk.

The white cloth background has visible stationary folds/seams, while the wooden board rotates with the vessel and is highly feature-rich. Board leakage is therefore a first-class SfM/MVS failure mode. Grounding DINO-T + SAM 2.1 is the authoritative isolation route; G8/G9 same-setup empty-board tails are used as negative refinement evidence near the base. Simple white-threshold segmentation is not the V4 plan, and no recapture is expected or requested.

## Fast execution policy

When V4 implementation starts, begin from the first unchecked task in the audited V4 implementation plan. Skip broad historical regression suites, algorithm comparisons, large reports, and parameter sweeps. Keep only the checks that prevent wasting reconstruction time:

1. V4 media decode correctly;
2. masks preserve the complete vessel;
3. sparse virtual camera rings/points are coherent and cross-elevation registration is connected;
4. the real raw dense cloud/Poisson mesh is visually plausible before cleanup;
5. the final Blender master opens correctly;
6. the exported GLB cleanly re-imports.

## Retained reusable code

The project keeps capture-independent OpenCV QA/geometry utilities, ALIKED/LightGlue feature and database helpers, pyCOLMAP/COLMAP orchestration, PLY/mesh helpers, and segmentation utilities for V4. Legacy V3/Step-specific assumptions must be generalized rather than reused blindly.
