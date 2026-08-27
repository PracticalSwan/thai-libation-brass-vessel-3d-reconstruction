# Active Context

Updated: 2026-08-27

## Current focus

Preprocessing is complete and verified. A geometry-detection and machine-learning extension is now designed and planned, but not implemented. The next authorized implementation should complete Phase A geometry/ML analysis and presentation evidence without modifying `preprocessing/pycolmap_input/images/`; the later Phase B may then compare unmasked versus ML-mask-assisted pyCOLMAP reconstruction.

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

## Planned extension state

- Geometry 1: SIFT keypoints/candidate matches plus Fundamental Matrix/RANSAC inlier visualization.
- Geometry 2: epipolar-line visualization and geometric residuals from the same two-view geometry.
- Geometry 3: Canny edges, contours, ellipse fitting where valid, and a principal/symmetry axis for representative vessel views.
- ML: pretrained Meta SAM 2.1 vessel segmentation, starting with `sam2.1_hiera_small`, producing COLMAP-compatible binary masks without changing image geometry.
- Planned course-presentation evidence includes match/inlier, epipolar, shape-geometry, segmentation, mask-contact-sheet, and masked-keypoint figures.
- Design: `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`.
- Implementation plan: `docs/superpowers/plans/2026-08-27-geometry-ml-integration.md`.
- No geometry-extension scripts, SAM masks/model weights, or pyCOLMAP outputs exist yet.

## Next action

When implementation is explicitly authorized, execute Phase A of the geometry/ML plan and stop before pyCOLMAP. After Phase A is verified, a separately authorized Phase B may run controlled unmasked and SAM-mask-assisted pyCOLMAP sparse reconstruction and choose the reconstruction path from measured evidence.
