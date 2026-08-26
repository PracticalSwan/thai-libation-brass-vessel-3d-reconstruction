# Active Context

Updated: 2026-08-27

## Current focus

Preprocessing is complete and verified. The next phase may begin pyCOLMAP feature extraction and matching using only `preprocessing/pycolmap_input/images/`, but this task intentionally stopped before any pyCOLMAP command or reconstruction stage.

## Verified preprocessing state

- Raw source: `IMG20260826122949/`, 297 immutable JPEG files at 3072 x 4080.
- Final decisions: 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT`.
- Rejected images: indices 289-297 only, because they are the separate hand-held/flipped sequence with object movement and hand occlusion.
- Selected set: all 288 `ACCEPT` + `WARN` images.
- Selected variant: PREPROCESSED, using a geometry-preserving 15% LAB-luminance CLAHE blend.
- Matching evidence: 2,483 PREPROCESSED versus 2,376 RAW fundamental-matrix RANSAC inliers over ten representative neighboring pairs; PREPROCESSED was non-worse on 9 of 10. Matching used decoded quality-95 JPEG bytes identical to the final export encoding.
- Final input: `preprocessing/pycolmap_input/images/`, 288 readable files at unchanged 3072 x 4080 geometry.
- Integrity: no duplicate selected-output hashes and zero raw SHA-256/size mismatches across all 297 originals.
- Visual review: all ten before/after previews and all four sheets containing every WARN/REJECT case were inspected.

## Next action

Plan and run the pyCOLMAP stage separately. Reconfirm the installed pyCOLMAP API/environment, use the final 288-image directory as the only input, and preserve the preprocessing reports as the provenance boundary.
