# Changelog

## Unreleased

### Current V4 dataset cleanup and publication — 2026-09-16

- Removed the superseded pre-V4 raw-image dataset and its legacy preprocessing, analysis, segmentation-dataset, checkpoint, pipeline-code, test, and documentation trees.
- Kept `CSX4213_Project_V4_Images/` as the only authoritative raw source set: 688 immutable JPEG photographs.
- Kept the current processed image sets under `capture_v4/derived/`: 372 MVS images, 372 vessel masks, and 372 feature masks.
- Added Git LFS rules for the current raw and processed image trees so the exact media used by the V4 pipeline can be versioned without mixing in obsolete captures.
- Retained only the current V4 reconstruction lineage, current V4 engineering documentation, representative reconstruction evidence, and reusable code still required by `run_v4.py` / `v4_*`.
- Updated path protection, contributor instructions, documentation, and tests to remove stale pre-V4 references.

### V4 canonical reconstruction baseline — 2026-09-13

- Corrected `_depth_pair_consistency` to compare source geometric depth against reprojected source-camera Z and added a distinct-camera-pose regression.
- Preserved the finite/rank-3 masked fused candidate (`1,426,482` colored points) with measured cross-ring evidence and explicit residual limitations.
- Generated the selected Poisson surface and retained representative raw-surface evidence before Blender processing.
- Built scan-derived Blender clean-high and LOD0 assets with UVs, packed AO, and uncoated-reference brass PBR appearance.
- Saved the canonical Blender master and GLB and verified a fresh factory-startup GLB re-import with finite geometry, normals, UV/material state, packed texture, matching bounds, and no debug export.
- Reconstruction limitations remain explicit where captured dense evidence is absent; no synthetic/reference-modeled vessel anatomy is accepted.
