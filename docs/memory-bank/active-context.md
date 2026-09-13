# Active Context

Updated: 2026-09-13

## Current focus

**V4 is the only active direction.** V1/V2 were rejected by the professor and V3 was visually rejected after direct Blender inspection. Their active reconstruction/model artifacts were removed; do not restore them unless historical recovery is explicitly requested.

The final V4 photographs are present and cannot be retaken. **Implementation has progressed through ingest, isolation, learned matching, sparse reconstruction, and dense reconstruction. The active checkpoint is dense/post-fusion acceptance, immediately before raw Poisson meshing.**

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
```

The corrected-mask post-fusion report found `board_slab_detected=false`, `cloth_or_background_structure_detected=false`, `pedestal_board_webbing_detected=false`, and `vessel_identity_confirmed=true`. The remaining reported failure is driven by cross-ring continuity evidence that is currently invalidated by a coordinate-frame bug in `_depth_pair_consistency` plus an implementation-only 0.50-at-1% threshold.

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

## Immediate implementation action

Read `AGENTS.md`, this file, `progress.md`, the V4 spec, and the V4 implementation plan. Preserve the large intentional cleanup diff and all current V4 dense candidates/maps/hashes. **Do not restart Task 1 or any completed upstream stage.** First correct `_depth_pair_consistency` so source depth is compared against reprojected source-camera Z, add a distinct-pose regression, and re-run the ring audit on the existing true3 depth maps with the corrected 372/372 mask set. Treat the 0.50-at-1% score as diagnostic rather than an authoritative blocker. If corrected evidence and direct cloud inspection show a recognizable finite/rank-3 vessel without dominant board/background contamination, accept the best fused candidate and proceed immediately to one preserved raw Poisson mesh and the four-view raw visual gate. If viable, continue directly into Blender finalization, material/UV, canonical `.blend`/`.glb`, and fresh GLB re-import verification.

Canonical planning files:

```text
docs/superpowers/specs/2026-09-10-v4-fast-end-to-end-reconstruction-design.md
docs/superpowers/plans/2026-09-10-v4-fast-end-to-end-reconstruction.md
```
