# Changelog

## Unreleased

### V4 final model — 2026-09-13

- Corrected `_depth_pair_consistency` to compare source geometric depth against reprojected source-camera Z and added a distinct-camera-pose regression; the focused post-fusion suite passes (`7 passed`). Existing true3 maps were re-audited with the corrected 372/372 COLMAP masks without rerunning PatchMatch.
- Accepted and preserved the finite/rank-3 masked fused candidate (`1,426,482` colored points) after corrected contamination/identity evidence; all 28 selected cross-ring measurements are finite/measured. The 0.50-at-1% value remains diagnostic only.
- Generated one preserved raw Poisson mesh (`887,770` vertices / `1,669,931` faces), passed the four-view raw visual gate, and rejected coarser remesh alternatives that lost measured extent.
- Built scan-derived Blender clean-high and LOD0 assets, removing only 622 strict detached-noise components; LOD0 has `172,851` vertices / `299,324` faces, explicit-seam UVs, and a packed 1024² AO bake.
- Added uncoated-reference brass PBR (`MAT_V4_Brass`), saved the canonical Blender master and GLB, and verified a fresh factory-startup GLB re-import with one final mesh, finite normals/positions, packed texture, matching bounds, and no debug export.
- Final limitation remains explicit: porous/missing bowl and stem regions are retained where accepted dense evidence is absent; no synthetic/reference-modeled geometry was added and absolute physical scale is unverified.

### V4 reset — 2026-09-10

- User rejected the V3 sparse/dense/mesh visual result and directed the project to move to V4 using a new capture.
- Removed `reconstruction/learned_dense_v3/`, the retained old learned-reconstruction output used by V3, V3-only runners/render helpers, and the V3 implementation-plan file.
- Removed the V3 review objects from the unsaved Blender scene; no rejected V3 geometry was saved as a Blender master.
- Preserved historical raw photographs, preprocessing/analysis reference, reusable OpenCV/ML/ALIKED-LightGlue/COLMAP/PLY code, and project lessons.
- Current project state is waiting for a new V4 capture. Reconstruction should start immediately after the new images are provided.

### V3 final visual experiment — rejected

- Primary learned sparse result: ALIKED-N16Rot + LightGlue, 266/288 registered images.
- Full photometric depth/normal set reached 266/266 views.
- Full reduced-resolution photometric fusion produced about 80,015 points.
- Raw Poisson mesh contained 340,687 vertices and 657,693 faces and was visually unacceptable in Blender.
- The result showed that strong sparse registration did not overcome the old reflective/low-texture acquisition problems; further old-capture parameter sweeps were abandoned.

### Earlier project reset

- Professor rejected V1 and V2 and recommended retaking the object photographs.
- Rejected V1/V2 reference-assisted/final Blender implementation and the old SIFT/local-dense generated outputs were removed from the active project.

## Retained technical lessons

- Mild geometry-preserving preprocessing previously improved measured feature correspondences over raw input, but V4 preprocessing must be chosen from the new capture rather than copied blindly.
- ALIKED-N16Rot + LightGlue is retained as the primary learned local-feature matcher; SIFT/RANSAC remains a diagnostic/fallback.
- For polished brass, acquisition quality, stable visual features, and controlled reflections matter more than repeated reconstruction parameter sweeps.
