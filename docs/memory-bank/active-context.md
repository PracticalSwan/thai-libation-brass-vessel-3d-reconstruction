# Active Context

Updated: 2026-09-13

## Current focus

**V4 is the only active direction.** V1/V2 were rejected by the professor and V3 was visually rejected after direct Blender inspection. Their active reconstruction/model artifacts were removed; do not restore them unless historical recovery is explicitly requested.

The final V4 photographs are present and cannot be retaken. **Implementation is complete through dense/post-fusion acceptance, raw Poisson, conservative Blender cleanup, LOD0/UV/AO, brass lookdev, canonical export, and fresh GLB re-import.**

Latest verified continuation facts:

```text
selected geometry / sparse registered: 372 / 372
accepted sparse model: one connected model
sparse points: 65,560
observations: 460,628
mean track length: ~7.026
mean reprojection error: ~1.223 px
camera model: SIMPLE_RADIAL
cross-ring connections: 1005 in the healthy source graph
corrected COLMAP fusion-mask resolution: 372 / 372, zero legacy fallback/missing
corrected-mask fused SHA-256: df05019e2e56d1c54351f4b2cee161cbcfb7b782a3c6b0d4920d6e303f39d7d8
raw Poisson: 887,770 vertices / 1,669,931 faces; SHA-256 55a4e92c9491cced508360d1645355ed447785f175df0556b400528fa9522941
clean high: 883,016 vertices / 1,662,911 faces after strict detached-noise cull
LOD0: 172,851 vertices / 299,324 faces; one finite angle-based UV map
canonical master: reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend
canonical GLB: reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb
```

The corrected-mask post-fusion report found `board_slab_detected=false`, `cloth_or_background_structure_detected=false`, `pedestal_board_webbing_detected=false`, and `vessel_identity_confirmed=true`. The source-camera-Z correction was regression-tested and the existing true3 maps were re-audited without PatchMatch recomputation. All 28 selected cross-ring measurements are finite/measured; the 0.50-at-1% value is diagnostic only (four transitions remain below it), so no geometry transition rerun was justified.

## Final V4 assets and limits

The accepted fused cloud is finite, rank-3, and recognizable in four semantic views. One raw Poisson mesh is preserved unchanged and passed the four-view raw gate. Blender retains the raw and rollback objects, a hidden scan-derived clean-high object, and active `SM_V4_Vessel_LOD0`. Cleanup removed only 622 detached components that met both the 100-face and 0.0035-world-dimension limits; coarser voxel/remesh tests were rejected after measured extent loss.

`MAT_V4_Brass` is built from the uncoated reference set, using `IMG20260912132007.jpg` (SHA-256 `393C8FDCFBB025EA3D58649AB2E6314BF2D6E7185C3ACEA6744B8055AC959F3B`) and a packed 1024² AO detail map. Fresh factory-startup GLB import passed the mesh/material/texture/bounds/normals/export-scope gate and produced four re-import previews under `reconstruction/v4/previews/final_glb_reimport_v1/`.

The final geometry is intentionally honest: porous/missing bowl and stem regions remain where the dense evidence is absent, no reference-modeled geometry was substituted, and absolute physical scale is unverified (normalized scale only).

Canonical immutable source:

```text
CSX4213_Project_V4_Images/
```

Do not rename, move, recompress, rotate, overwrite, or delete those files. Historical `IMG20260826122949/` remains immutable evidence/reference and is not V4 geometry input.

## Final V4 media audit

All 688 JPEGs were visually reviewed and inspected with ExifTool/decoded-image QA.

```text
688 total
158 uncoated appearance/reference
107 empty-board/background
423 coated/marked object-bearing geometry
3072 x 4080, EXIF orientation 1 for all files
OPPO Reno12 F rear 26mm-equivalent camera, 3.98 mm, f/1.8, digital zoom 1 for all files
```

Geometry/empty captures use consistent reported settings: ISO 100, 1/100 s, same lens/focal state and manual exposure/WB metadata. The 36 auto-exposure/auto-WB frames are early uncoated detail references, not geometry.

Six object-bearing source passes were found:

```text
geo_g7   142 images  14:19:12-14:24:04
geo_g8    72 images  14:33:36-14:37:18
geo_g9    55 images  14:44:13-14:46:29
geo_g10   59 images  14:51:20-14:53:47
geo_g11   58 images  15:01:36-15:03:48
geo_g12   37 images  15:05:14-15:06:27
```

G7 has roughly twice a normal ring count plus visual recurrence evidence, so implementation must detect revolution wrap(s) before assigning phase rather than treating 142 views as one circle or blindly splitting at 71.

Known empty-board sweeps:

```text
empty_g6  51 images  13:35:28-13:36:52
empty_g8  31 images  14:37:26-14:38:25
empty_g9  25 images  14:46:36-14:47:19
```

The G8/G9 empty tails are same-setup negative evidence for board-leakage refinement. G6 may be associated with a geometry setup only after camera/framing compatibility is proven.

No byte-identical duplicate exists. Same-second `_01` pairs include near-duplicates; preserve all source files but select only unique/useful geometry views in the manifest.

## Dominant real-media risks

The white cloth backdrop has visible folds/seams and is stationary, so it must never support SfM/MVS.

The wooden square board is the larger risk: it is richly textured **and rotates with the vessel**, so leaked board pixels can create false but geometrically consistent correspondence tracks and dense structure. V4 masking, sparse diagnostics, and dense fusion must explicitly prove that the vessel rather than the board supports the reconstruction.

The dry-shampoo treatment reduced but did not remove brass specularity. Black random markers are dense on the outer bowl, globe/shoulder, pedestal/base, and much of the neck. Higher-risk/sparser regions are the finial, polished transition bands, bowl rim/interior, and base contact. High-angle geometry exists and supplies top/interior/finial evidence.

The 158 uncoated images are reserved for material/lookdev. Broad clean rings are the primary material reference; close-detail references contain some clipping and occasional tripod/green-sheet intrusion and must be sampled selectively.

## Approved V4 pipeline

```text
immutable final source
-> deterministic 688-file manifest / role / logical-ring / phase audit
-> Grounding DINO-T + SAM 2.1 masks
-> phase-compatible empty-board negative refinement where available
-> mask-filtered ALIKED-N16Rot + LightGlue
-> circular + cross-ring phase-aware pair schedule
-> COLMAP geometric verification
-> pyCOLMAP incremental SfM + bundle adjustment
-> vessel-centered/board-free sparse gate
-> image + mask undistortion
-> real bounded CUDA PatchMatch smoke
-> full CUDA geometric PatchMatch + mask-aware geometric fusion
-> Poisson raw mesh
-> mandatory raw visual gate
-> conservative Blender scan cleanup
-> production mesh / UV / useful detail bake
-> brass PBR from uncoated references
-> canonical .blend + .glb
-> fresh GLB re-import verification
```

No SIFT comparison, alternate 3D stack, dense-method comparison, mesher bake-off, or synthetic replacement is planned.

## Camera and matching implications

Actual EXIF strongly supports one shared geometry intrinsic group by default: same device/lens/focal/zoom/resolution/orientation. Start with `SIMPLE_RADIAL`, but split/change only if calibration evidence requires it.

Cross-ring start angles are not measured. Implementation must estimate circular phase offsets from vessel-only learned match/inlier support and pair by normalized phase rather than raw index. Unequal ring counts and G7 repeated coverage make fixed-index sequential matching invalid.

## Tools verified during planning

```text
CodeGraph 1.6.0 provider/index available and current
ExifTool 13.59
FFmpeg 9.0.1
COLMAP 4.2.0 commit be5e291 with CUDA
```

Project handoff also previously verified Python 3.14.2, PyTorch 2.13.0+cu130 with CUDA, NVIDIA GeForce RTX 5050 Laptop GPU, and pyCOLMAP 4.2.0. Implementation refreshes these identities before use.

## Completion checkpoint

The dense metric fix, distinct-pose regression, corrected true3 ring audit, raw Poisson, four-view raw inspection, scan-preserving cleanup, LOD0/UV/AO, brass material, canonical Blender save/export, and fresh GLB re-import are complete. Do not reopen ingest, segmentation, matching, sparse SfM, PatchMatch, or rejected V1/V2/V3 geometry unless a new user request supplies concrete contrary evidence.

Canonical planning files:

```text
docs/superpowers/specs/2026-09-10-v4-fast-end-to-end-reconstruction-design.md
docs/superpowers/plans/2026-09-10-v4-fast-end-to-end-reconstruction.md
```
