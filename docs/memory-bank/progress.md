# Progress

Updated: 2026-09-13

## Current project state

The project is at **V4 dense/post-fusion acceptance; raw Poisson and Blender finalization remain**.

- V1: rejected by professor; removed from active project.
- V2: rejected by professor; removed from active project.
- V3: rejected by user after direct Blender inspection; V3 artifacts and V3-only implementation files removed.
- The large existing cleanup diff is intentional and must not be restored.
- Historical Git history remains available only if recovery is explicitly requested.

Verified V4 implementation progress now includes:

```text
372 selected geometry views
372 / 372 sparse registration in one connected model
65,560 sparse points
460,628 observations
~7.026 mean track length
~1.223 px mean reprojection error
SIMPLE_RADIAL sparse camera model
completed 2000-pixel CUDA geometric PatchMatch workspaces/candidates preserved
corrected StereoFusion mask contract: 372 / 372 `<image_name>.png` masks resolved
corrected-mask fused SHA-256: df05019e2e56d1c54351f4b2cee161cbcfb7b782a3c6b0d4920d6e303f39d7d8
```

The corrected-mask post-fusion evidence currently reports no detected dominant board slab, cloth/background structure, or pedestal-board webbing and confirms vessel identity. Raw Poisson has not yet been authorized by the current software gate because the ring-transition audit still uses a defective cross-camera depth comparison and an implementation-only hard threshold.

Canonical immutable V4 source:

```text
CSX4213_Project_V4_Images/
```

## Completed V4 planning/media work

The complete final source set was audited before implementation:

```text
688 JPEGs total
158 uncoated appearance/reference
107 empty-board/background
423 coated/marked object-bearing geometry
all 3072x4080, orientation 1
same OPPO Reno12 F rear 26mm-equivalent lens, 3.98 mm, f/1.8, digital zoom 1
```

All images were visually reviewed through bounded contact sheets and numerically inspected for metadata/technical anomalies. No corrupt source or catastrophic object-bearing blur/cutoff was found. Geometry highlight clipping is limited despite residual brass specularity.

Object-bearing capture passes:

```text
geo_g7   142
geo_g8    72
geo_g9    55
geo_g10   59
geo_g11   58
geo_g12   37
```

Empty-board sweeps:

```text
empty_g6  51
empty_g8  31
empty_g9  25
```

Exact mixed-sequence boundaries were verified:

```text
G8 geometry through IMG20260912143718.jpg; empty starts IMG20260912143726.jpg
G9 geometry through IMG20260912144629.jpg; empty starts IMG20260912144636.jpg
```

G7 shows repeated phase/revolution evidence and must be split/deduplicated from computed wrap evidence during ingest rather than treated as one 142-view circle.

No SHA-identical duplicates were found. Same-second `_01` pairs include perceptually redundant views and will be retained as source but selectively excluded from reconstruction when they add no phase evidence.

## Actual-media reconstruction implications

The dominant failure risk is the rotating wooden board. Its texture rotates with the vessel, so segmentation leakage can create false but internally consistent SfM/MVS support. The updated V4 plan therefore requires:

- complete vessel-only masks before ALIKED;
- same-setup G8/G9 empty-board sweeps as negative refinement evidence near the foot/base;
- explicit board-contamination checks before sparse acceptance and after dense fusion;
- aligned vessel masks during geometric fusion.

The stationary white cloth also has strong folds/seams and must be excluded.

Dry-shampoo coating did not fully suppress reflection, but dense random black markers are present over most major surfaces. Higher-risk regions are finial/ridged tip, polished transition bands, bowl rim/interior, and base contact. High-angle/top coverage exists.

The 158 uncoated images provide final brass appearance evidence. Broad clean reference rings are preferred; close/detail frames with clipped reflections, green-sheet/tripod intrusion, or unrelated background are filtered during material sampling.

## Approved V4 implementation

The canonical updated design and execution plan are:

```text
docs/superpowers/specs/2026-09-10-v4-fast-end-to-end-reconstruction-design.md
docs/superpowers/plans/2026-09-10-v4-fast-end-to-end-reconstruction.md
```

Implementation order is:

```text
paths/tools/restart state
-> authoritative 688-file manifest + logical rings/phases
-> Grounding DINO-T + SAM 2.1 masks + board suppression
-> ALIKED-N16Rot resource preflight + vessel-only feature cache
-> cross-ring phase-offset estimation + deterministic circular pair schedule
-> LightGlue + COLMAP geometric verification database
-> pyCOLMAP incremental SfM/BA + board-free sparse gate
-> image/mask undistortion + real CUDA PatchMatch smoke
-> production CUDA geometric PatchMatch + masked geometric fusion
-> Poisson raw mesh + mandatory raw visual gate
-> conservative Blender scan cleanup
-> production mesh/UV/detail bake
-> brass PBR from clean references
-> canonical .blend/.glb + fresh GLB re-import
-> focused verification/cleanup/docs
```

No alternate reconstruction stack or synthetic replacement is planned.

## Reusable V4 components retained

- historical immutable raw capture and analysis references;
- OpenCV image/geometry utilities;
- ALIKED/LightGlue matching and COLMAP database helpers;
- pyCOLMAP/COLMAP orchestration and PLY helpers;
- segmentation/ML utilities;
- private SmallSeg checkpoint under `analysis/ml/checkpoints/`, which remains local and is not the V4 primary segmenter;
- local CodeGraph index state.

Legacy V4-facing assumptions still scheduled for narrow generalization include `expected_images=288`, old >=274 registration gates, Step-13 labels/contracts, exactly-one-camera acceptance logic, and the old `reconstruction/local_dense` path contract.

## Planning/tool state

Fresh planning-session verification:

```text
CodeGraph 1.6.0 provider/index current
ExifTool 13.59
FFmpeg 9.0.1 full build
COLMAP 4.2.0 commit be5e291 with CUDA
```

Earlier project verification also established Python 3.14.2, PyTorch 2.13.0+cu130 with CUDA, NVIDIA GeForce RTX 5050 Laptop GPU, and pyCOLMAP 4.2.0. Implementation refreshes these before use and performs a real bounded CUDA PatchMatch smoke before production dense reconstruction.

## Completion/verification policy

Essential gates are: 688-file manifest reconciliation; full vessel masks with no board/background leakage; coherent cross-connected vessel-centered sparse model; real CUDA PatchMatch smoke; plausible fused cloud/raw Poisson before Blender; valid canonical `.blend`; clean GLB re-import.

### Current completion direction — 2026-09-13

Do not repeat completed upstream stages or restart the full dense run. Correct the ring depth audit first: `_depth_pair_consistency` must compare the source geometric depth map to the reprojected **source-camera Z** rather than to reference-camera depth. Add a distinct-camera-pose regression and re-evaluate existing true3 maps with the corrected 372/372 masks. The current `mean_consistent_fraction_at_1pct >= 0.50` rule is diagnostic only and must not override the authoritative visual/contamination gate. If corrected evidence shows no real transition break and the cloud is recognizable, finite/rank-3, and free of dominant board/background contamination, create one canonical `poisson_raw.ply`, preserve it unchanged, inspect it from front/quarter/side/top-oblique, and—if viable—finish Blender cleanup, LOD0/UV/detail bake, brass material from uncoated references, canonical `.blend`/`.glb`, and fresh GLB re-import. Only a concrete real geometry defect justifies localized dense recomputation.

This documentation update changes direction/state only. It does not itself modify V4 reconstruction code, dense outputs, Blender state, Git history, or external services.
