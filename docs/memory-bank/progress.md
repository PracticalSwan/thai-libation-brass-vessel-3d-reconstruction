# Progress

Updated: 2026-09-06

## Completed and verified

### Preprocessing

- Published and audited the 297-image real capture set while preserving raw-file immutability.
- Final quality decisions are 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT`; all 288 non-rejected images remain in the selected set.
- The final PREPROCESSED variant uses the geometry-preserving 15% LAB-luminance CLAHE blend.
- Ten neighboring-pair RAW/PREPROCESSED comparisons produced 2,483 verified PREPROCESSED Fundamental-Matrix RANSAC inliers versus 2,376 RAW; PREPROCESSED was non-worse on 9/10 pairs.
- `preprocessing/pycolmap_input/images/` contains the deterministic 288-image next-stage input set.
- Final integrity verification found 0 raw mismatches across all 297 photographs and verified all 288 selected outputs against `selection_manifest.csv`.

### Step 6 — classical geometry

- Implemented `analysis_common.py`, `geometry_detection.py`, `shape_geometry.py`, `run_geometry_analysis.py`, and `show_geometry_visuals.py`.
- Pair 165-166: 4,653 / 4,643 SIFT keypoints, 478 candidates, 300 RANSAC inliers, 0.628 inlier ratio, median Sampson error 0.1431 px².
- Supporting pair 255-256: 57 candidates, 18 RANSAC inliers.
- Classical shape analysis retained contour/centroid/bounding-box/PCA evidence for images 165 and 255, rejected the weak global ellipse for 165, and retained the valid ellipse for 255.
- Six Step 6 presentation figures and machine-readable reports were generated and visually verified.

### Steps 7 + 8 — custom CNN segmentation and SIFT feature-mask analysis

- Frozen 36 reviewed source-resolution vessel masks under `ml_dataset/masks/`.
- Sequence-aware split: 24 train / 6 validation / 6 held-out test.
- Held-out indices: 72, 142, 165, 200, 255, 288.
- Label manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Annotation method: `opencv_assisted_visually_reviewed_bounded_correction`; CNN predictions were never used as labels.
- Implemented `segmentation_data.py`, `cnn_segmentation.py`, `train_cnn_segmentation.py`, `ml_feature_analysis.py`, and `run_ml_analysis.py` plus focused tests.
- `SmallSegCNN` is a project-defined compact U-Net-like network trained from random initialization with no pretrained weights/backbone and **487,297** trainable parameters.
- Fixed training configuration: 384 x 288 `(H x W)`, BCE-with-logits + soft Dice loss, Adam lr 1e-3, batch 8, seed 4213, threshold 0.5, validation-only checkpoint selection.
- Measured training environment: Python 3.14.2, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 5050 Laptop GPU.
- Training completed 49 epochs in 332.112 s. Best epoch 39 reached validation Dice 0.968066 and IoU 0.938252. No training-only label expansion was needed.
- Frozen held-out aggregate: mean Dice 0.952521, median Dice 0.963377, mean IoU 0.910745, median IoU 0.929347, mean precision 0.930884, mean recall 0.976529.
- The held-out image-72 yellow-wall `background_false_positive` remains visible and unedited; index 200 retains a smaller `minor_boundary_error`. The other four test predictions are recorded as `ok`.
- Step 8 reuses `geometry_detection.extract_sift` and CNN-predicted masks: 28,673 total held-out SIFT keypoints, 27,431 inside predicted vessel masks, and 1,242 outside. Mean per-image vessel-feature fraction is 0.952693.
- Six ML presentation figures were generated from real outputs and visually inspected. The feature-mask figures retain the weak image-72 limitation instead of implying that high vessel-feature fractions prove reconstruction improvement.

### Step 9 — reconstruction readiness

- Implemented `reconstruction_masks.py`, `reconstruction_matching.py`, `camera_readiness.py`, and `run_reconstruction_readiness.py` with bounded stages `masks`, `benchmark`, `connectivity`, `camera`, and `summary`.
- Full-sequence inference produced 288 source-size raw CNN predictions and 288 deterministic cleanup masks. Cleanup changed 30 predictions; mean foreground fraction changed from 0.275260 to 0.274487.
- Frozen 20-pair x 3-mode geometry benchmark chose **unmasked** SIFT: 3,146 RANSAC inliers versus 2,841 for both masked modes. Both masked modes retained 90.31% of unmasked inliers and failed the fixed 95% qualification floor.
- Full 287-edge adjacent audit found 273 strong and 14 weak transitions. Fourteen local skip bridges were tested and none was strong, so the conservative recommended subset remains 288/288 images with zero exclusions.
- Camera/EXIF audit found one complete signature across all 288 selected filenames: OPPO Reno12 F, 3072 x 4080, orientation 1, 3.98 mm focal length, 26 mm 35-mm equivalent, digital zoom 1.0. The measured starting recommendation is one shared camera/intrinsics group.
- Four Step 9 presentation figures and machine-readable CSV/JSON evidence were generated and visually inspected. A zero-range camera-metadata plot defect found during review was fixed.
- `docs/geometry-ml/reconstruction-readiness.md` records the measured Step 9 method, results, limitations, and no-reconstruction boundary.

### Step 10 — sparse Structure from Motion

- Added `pycolmap>=4.2,<5`, `sparse_reconstruction.py`, `run_sparse_reconstruction.py`, and focused Step 10 tests.
- Final sparse-SIFT runtime used pyCOLMAP 4.2.0, CPU feature extraction, `max_image_size=1200`, native unmasked SIFT, one shared `SIMPLE_RADIAL` camera, and sequential matching.
- Baseline overlap 20: 1,255,153 SIFT features, 1,500 non-empty matched pairs, 902 verified pairs, 7 sparse models, 216 distinct images represented across those models. Largest model: 73/288 images, 6,099 points, 21,351 observations, mean track length 3.5007, mean reprojection error 1.2373 px.
- The single planned overlap-40 retry also produced 7 models; union coverage increased to 223 images but the largest model remained 73 images with 5,769 points.
- The frozen ranking rule selected the baseline 73-image component; it is exported under `reconstruction/sparse/best/` with `points3D.ply`.
- Both Step 10 figures were visually inspected. The selected local component has a coherent camera arc and plausible point cloud, but the full sequence remains fragmented, so `acceptance_met=false` and dense reconstruction was not started.
- `docs/geometry-ml/sparse-reconstruction.md` records the measured implementation, retry decision, fragmentation evidence, outputs, and boundary.

### Step 11 — sparse component bridging

- Refactored the Step 10 image-reader, feature-extraction, and incremental-mapping configuration into shared public helpers so Step 11 uses the same `SIMPLE_RADIAL`, CPU SIFT, and mapper contract.
- Implemented deterministic bridge candidate generation, pair-list safety, SQLite match summaries, qualification/selection, the fail-closed targeted gate, resumable feature/exhaustive databases, and durable CLI stages.
- Matched exactly 2,340 non-local diagnostic candidates: 780 around each of boundaries 73-74, 145-146, and 203-204.
- Boundaries 73-74 and 145-146 had zero geometrically verified candidates. Boundary 203-204 had 68 qualified candidates and 8 selected bridges, so the targeted mapper was skipped because all three boundaries did not qualify.
- Ran exactly one CPU exhaustive fallback with block size 50. Its database contains 14,900 non-empty match rows and 3,020 verified pair rows.
- Exhaustive mapping produced eight models with 224 distinct images across their disconnected union. The strongest single model registers 73/288 images with 3,443 points, 12,914 observations, mean track length 3.7508, and 1.1989 px mean reprojection error.
- The Step 11 candidate, sparse-model, and registration figures were visually inspected. They show the two empty bridge boundaries, the eight selected 203-204 bridges, a plausible local camera arc/point structure, and registration limited to indices 1-73.
- `reconstruction/bridging/best/`, `points3D.ply`, all component models, seven reports, and three figures preserve the measured result. `bridge_success=false`; dense reconstruction remains blocked.
- `docs/geometry-ml/sparse-component-bridging.md` records the method, interruption/resume provenance, measurements, visual review, acceptance decision, artifacts, and boundary.

### Step 12 — learned sparse recovery capability boundary

- Implemented the bounded native learned-recovery domain and staged runner with ALIKED-N16Rot + ALIKED-LightGlue first and LoMa-B + LoMa-L only as the conditional fallback.
- Preserved explicit learned matcher options on both sequential/imported paths, exact Step 11 candidate identity, strict per-image feature-cache layout fingerprints, strongest-single-model acceptance, bounded cleanup, and no learned exhaustive or dense API.
- Static Python 3.14.2 / pyCOLMAP 4.2.0 capability evidence exposed both approved frontends with valid CPU option objects and `max_image_size=1600`.
- The real ALIKED smoke on selected indices 1-2 failed before feature extraction with `RuntimeError: ALIKED feature extraction requires ONNX support.`
- The capability gate stopped the 288-image ALIKED extraction, all 2,340 learned boundary matches, learned sparse mapping, and every LoMa runtime call. No dependency replacement, pyCOLMAP rebuild, or external learned stack was introduced.
- Final artifacts truthfully record ALIKED `blocked`, LoMa `not_run`, no selected learned model, `metric_acceptance_met=false`, `visual_plausibility_status=failed`, `learned_recovery_success=false`, and `dense_reconstruction_started=false`.
- The only applicable Step 12 figure compares the authoritative Step 10/11 strongest-single-model SIFT baselines and was visually inspected. `docs/geometry-ml/learned-sparse-recovery.md` records the measured capability boundary.

### Step 13 — external learned global recovery

- Added one pinned official CVG LightGlue dependency plus `external_learned_recovery.py`, `run_external_learned_recovery.py`, focused tests, a frozen design/plan, and Step 13-local durable evidence.
- Real ALIKED-N16Rot + LightGlue capability ran on CUDA with 4,096 features per smoke image and 2,952 raw matches.
- Exact 2,340-pair learned diagnostics recovered all three critical boundaries: 778 qualified candidates at 73-74, 418 at 145-146, and 745 at 203-204; eight bridges were selected per boundary.
- Diagnostic reporting records 363,318 pre-verification LightGlue matches and 363,171 post-verification COLMAP match rows.
- The one approved full schedule contained 5,574 unique pairs and 5,269,937 raw LightGlue correspondences. Its verified COLMAP database contained 288 images, 1,129,555 imported keypoints, 5,531 non-empty match rows, and 5,264 verified pair rows.
- Three sparse models were produced. The strongest registers 266/288 images with 29,713 points, 106,480 observations, mean track length 3.5836, one `SIMPLE_RADIAL` camera, and 1.374824 px mean reprojection error.
- The model leaves selected indices 267-288 unregistered and therefore misses the frozen >=274 global gate by 8 images. `metric_acceptance_met=false`, `visual_plausibility_status=failed`, and `step13_success=false`.
- The 266-image model plus PLY is retained under `reconstruction/external_learned_recovery/best/` as evidence, but the frozen local fallback selects Step 10 `reconstruction/sparse/best` because it ties Step 11 at 73 images and has more sparse points (6,099 vs 3,443).
- Four real Step 13 figures were generated and visually inspected. Transient learned features, diagnostic/mapping databases, pair lists, mapping components, and the Step 13 `work/` directory were removed after finalization.
- Sparse recovery is closed; no Step 13 retry, second matcher, parameter sweep, learned exhaustive matcher, dense MVS, mesh, texture, or Blender output was run.
- `docs/geometry-ml/external-learned-global-recovery.md` records the complete measured Step 13 outcome.

### Steps 14-17 — local dense reconstruction, mesh, and photo texture

- Implemented restartable `prepare`, `stereo`, `mesh`, `texture`, `finalize`, and tested `all` orchestration with frozen Step 10 source protection and bounded attempt ledgers.
- Prepared exactly 73 registered views. Official COLMAP 4.2.0 CUDA PatchMatch completed 73 depth/normal maps in 1,811.67 seconds; no PatchMatch fallback ran.
- Geometric fusion produced 391,899 finite colored points in 101.83 seconds after one recorded cache-only correction from 1 GiB to 4 GiB reused the completed maps. The dense/sparse ratio is 64.2563 and expanded-box coverage is 99.9980%.
- Poisson primary completed with 1,088,150 vertices / 2,040,189 faces. The one 69,153-face Delaunay alternative was rejected for giant unsupported sheets.
- The measured 0.5%-face component rule kept 18 components and 96.6387% of Poisson faces while removing 5,458 tiny components. QEM simplified the retained surface at ratio 0.253599 to the accepted 283,341-vertex / 499,999-face final mesh.
- One COLMAP texture attempt completed in 48.80 seconds. Exact coordinates and face topology were preserved; meaningful UV coverage is 73.0045% and the atlas is 4096 x 1902.
- Headless Blender 5.2.0 LTS reopened the mesh/UV/material/atlas contract and rendered three calibrated views. No `.blend` file or manual cleanup was produced.
- All Step 14-17 hard gates passed. `reconstruction/local_dense/reports/steps14_17_summary.json` records `steps14_17_success=true` and `blender_manual_cleanup_started=false`.

## Verification

- Final Steps 14-17 checks: **49 focused tests passed**, **106 Step 10-13 regression tests passed**, and **247 complete project tests passed**.
- Independent GLM review led to four verified safeguards: full Windows process-tree termination/restart checks, ownership guards for final and texture-attempt outputs, normalized UV-range validation, and explicit sampled-vertex evidence labels for large Step 16 previews.
- The four Steps 14-17 source modules and four focused test files compiled successfully.
- The real `--stage all` path passed without creating additional PatchMatch, mesher, simplifier, or texturer attempts.
- Final reopening verified the 73-image / 6,099-point Step 10 model, 391,899-point dense cloud, 283,341-vertex / 499,999-face final mesh, exact textured topology, 73.0045% meaningful UV coverage, 4096 x 1902 atlas, and three-view Blender report.
- Final integrity found 297/297 raw images unchanged, 288/288 selected inputs exact, all 73 local inputs exact, and 160 protected Step 10-13 files unchanged. `.codegraph/` and the private checkpoint remain present; no `.blend` or manual/sculpt output exists.
- Documentation links, whitespace, and Git diff checks passed after bounded cleanup of task-created caches, temporary preview inputs, duplicate Delaunay workspace copies, process state, intermediate per-camera renders, and accidental `NUL` residue.
- Fresh Step 11-focused suite after review: **32 passed**.
- Fresh complete project suite after Step 11 review: **141 passed**.
- Step 12 maintenance verification: **20 domain tests passed**, **14 runner tests passed**, and **49 Step 10/11 regression tests passed**.
- The runtime-blocker reporting regression first raised the Step 12 runner suite to **15 passed**; final review added a blocked-attempt frontend-identity regression, bringing the runner suite to **16 passed**.
- Final complete project verification after Step 12 review: **177 passed**; all four Step 12 source/test files compiled successfully.
- Final Step 13 verification: **21 focused tests passed** and **198 complete project tests passed**; all four Step 13 source/test files compiled successfully and compile-cache residue was removed.
- Final Step 13 source-integrity verification found 297/297 raw images unchanged and 288/288 selected images matching the frozen manifest; five protected Step 10/11 report hashes remained exactly unchanged.
- The retained Step 13 evidence model reopened at 266 images / 29,713 points / one `SIMPLE_RADIAL` camera / 1.374823762049934 px. The selected downstream Step 10 model reopened at 73 images / 6,099 points / 1.2373052447638215 px.
- Step 13 source scan found no learned exhaustive or dense/MVS API, transient `work/` state is absent, and all four Step 13 figures were visually inspected.
- The Step 10/11 selected models reopened with exact recorded metrics, and five protected Step 10/11 report hashes matched their pre-Step-12 snapshots.
- Final source verification again found 297/297 raw images and 288/288 selected images unchanged; Step 12 left no transient database, work directory, learned model/PLY, or compile cache.
- Syntax compilation succeeded for both Step 12 source modules and both Step 12 test files.
- Changed sparse Python modules completed `python -B -m py_compile` successfully.
- Fresh final source-integrity verification: 297/297 raw unchanged with zero mismatches; 288/288 selected images verified against `selection_manifest.csv`.
- The Step 10 selected sparse model re-opened with pyCOLMAP 4.2.0 and exactly matched the summary metrics: 73 registered images, 6,099 sparse points, one camera, 1.2373052447638215 px mean reprojection error.
- The Step 11 selected model re-opened and exactly matched its summary metrics: 73 registered images, 3,443 points, one `SIMPLE_RADIAL` camera, 1.1988826674412258 px mean reprojection error.
- Review fixes now reject stale feature/resume databases whose per-image names, camera IDs, keypoint rows, or descriptor rows differ despite matching aggregate counts; the zero-inlier candidate figure also uses a nonnegative scale.
- Frozen Step 10 report hashes remained exactly unchanged after Step 11.
- Both final Step 10 figures were visually inspected; they explicitly identify the 73-image result as the selected component.
- Transient baseline/retry COLMAP databases and task-created caches were removed after model/report export; intentional sparse models, PLY, reports, figures, source code, spec, and plan remain.
- Final model checkpoint remains preserved locally at `analysis/ml/checkpoints/best_small_seg_cnn.pt` and is not intended for repository publication.
- No dense MVS, mesh, texture, or Blender artifact was created by Step 11.

## Local tooling

- CodeGraph 1.6.0 is installed and the repository graph is initialized under `.codegraph/`; the maintenance review confirmed the project index is available and current.
- Use CodeGraph when dependency, call-path, architecture, or change-impact analysis materially helps, but prefer direct inspection for trivial edits. Preserve `.codegraph/` unless explicit maintenance requires otherwise.

## Next phase

Steps 14-17 are complete. The accepted downstream asset is `reconstruction/local_dense/texture/attempt_1/mesh.ply` plus `texture.png`; the final untextured mesh is `reconstruction/local_dense/mesh/final_mesh.ply`. Any manual Blender cleanup or final presentation packaging is a separate phase and must preserve the local 73-view, disconnected-geometry, missing-region, reflective-brass, seam, and 73.00% UV-coverage limitations. Sparse recovery remains closed, and the Step 13 266-image model remains evidence only.
