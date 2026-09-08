# Changelog

All notable verified project milestones are recorded here.

## Unreleased

### Added
- Public repository and shared contributor/AI-agent workflow.
- Real 297-image capture audit with readability, EXIF, quality metrics, hashes, duplicate checks, and contact-sheet review.
- Immutable raw-data policy and explicit cleanup/removal policy.
- Public versioning of the reviewed 297-image raw capture set and image-processing audit evidence, per user authorization.
- Final `quality_check.py`, `preprocess_images.py`, and `run_preprocessing.py` workflow with 21 focused tests.
- Dataset-relative `ACCEPT` / `WARN` / `REJECT` decisions: 207 / 81 / 9, retaining 288 images.
- Ten neighboring-pair RAW vs PREPROCESSED SIFT comparisons with fundamental-matrix RANSAC verification.
- PREPROCESSED reconstruction input selection from 2,483 verified inliers versus 2,376 for RAW, non-worse on 9 of 10 pairs, using the exact exported quality-95 JPEG encoding.
- Deterministic 288-image `preprocessing/pycolmap_input/images/` set and selection manifest.
- Ten before/after previews, four complete WARN/REJECT sheets, and SIFT inlier chart.
- Final raw integrity and output checks: 297/297 original hashes unchanged; 288/288 selected outputs readable at 3072 x 4080; no duplicate output hashes.
- Initial Geometry/ML planning split the work into Step 6 geometry detection/analysis and a combined Steps 7+8 segmentation + feature-mask phase, while keeping pyCOLMAP/reconstruction outside the current scope.
- Revised the Steps 7+8 architecture on 2026-09-01 from pretrained SAM inference to a small binary segmentation CNN trained from random initialization: initial 36 manual masks, sequence-aware 24/6/6 train/validation/test split, validation-based model selection, held-out Dice/IoU evaluation, and Step 8 reuse of Step 6 SIFT on CNN-predicted masks.
- Step 6 selected-input verifier and shared scale contract in `analysis_common.py`,
  including manifest-order lookup plus readability, dimensions, size, and
  SHA-256 checks for all 288 inputs before output creation.
- Step 6 SIFT/Fundamental Matrix implementation with BF-L2 ratio-test matches,
  strict RANSAC-mask handling, epipolar lines, and Sampson residual reports.
- Step 6 classical shape implementation with grayscale/Canny evidence,
  deterministic brass-color contour selection with Canny fallback, bounding
  box, centroid, PCA principal axis, and residual-gated optional ellipse fit.
- Deterministic `run_geometry_analysis.py` orchestration and
  `show_geometry_visuals.py` popup visualizer, backed by 31 focused Step 6
  tests and real-input smoke checks.
- Six visually inspected real Step 6 presentation figures and five
  machine-readable report files under `analysis/`, including complete source,
  configuration, runtime, measurement, and scope-exclusion provenance.
- Measured Step 6 evidence: pair 165-166 produced 478 candidates / 300 RANSAC
  inliers; pair 255-256 produced 57 / 18; primary-pair median Sampson error was
  0.1431 analysis pixels squared.
- Final Step 6 integrity proof: 288/288 selected images matched their manifest,
  and all 297 raw originals remained unchanged with zero size or SHA-256
  mismatches.
- Frozen Steps 7+8 segmentation dataset with 36 reviewed source-size masks and a sequence-aware 24 train / 6 validation / 6 held-out test split. The label manifest SHA-256 is `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Project-defined `SmallSegCNN` binary segmenter trained from random initialization with 487,297 trainable parameters and no pretrained weights/backbone.
- Measured baseline training on CUDA: 49 epochs, best epoch 39, validation Dice 0.9681 / IoU 0.9383, 332.1 s runtime on the NVIDIA GeForce RTX 5050 Laptop GPU.
- Frozen six-image held-out evaluation with mean Dice 0.9525 and mean IoU 0.9107; the image-72 yellow-wall `background_false_positive` remains visible and unedited in the evidence.
- Step 8 feature-mask analysis reusing `geometry_detection.extract_sift`: 28,673 held-out SIFT keypoints, 27,431 inside CNN-predicted vessel masks and 1,242 outside.
- Six ML presentation figures plus machine-readable training, held-out, summary, and masked-feature reports generated from real outputs and visually reviewed.
- Final Steps 7+8 verification: 13 ML-focused tests and 66 complete project tests passed; changed Python compiled; all 297 raw photographs and 288 selected images remained unchanged.
- Step 9 reconstruction-readiness design and four implementation plans covering full-sequence CNN inference, masked-vs-unmasked geometry benchmarking, full-sequence connectivity/subset analysis, and camera/EXIF readiness.
- Step 9A full-sequence inference: 288 frozen source-size CNN predictions plus 288 deterministic connected-component cleanup masks; cleanup changed 30 predictions while preserving connected failure cases such as the known index-72 yellow-wall false positive.
- Step 9B frozen 20-pair x 3-mode geometry benchmark: unmasked SIFT produced 3,146 RANSAC inliers versus 2,841 for both masked modes. Each masked mode retained only 90.31% of unmasked inliers and failed the fixed 95% qualification floor, so unmasked SIFT is the measured readiness baseline.
- Step 9C full-sequence connectivity audit: 287 adjacent edges, 273 strong and 14 weak; 14 local skip bridges tested, zero strong. Conservative subset remains all 288 selected images with zero exclusions.
- Step 9D raw-EXIF audit: one complete camera signature across all 288 selected filenames (OPPO Reno12 F, 3072 x 4080, orientation 1, 3.98 mm focal length, 26 mm 35-mm equivalent, digital zoom 1.0), supporting one shared camera/intrinsics group as the later SfM starting recommendation.
- Step 9 machine-readable evidence, 288-image inclusion manifest, and four visually inspected presentation figures under `analysis/` and `preprocessing/reconstruction_input_v1/`.
- Final Step 9 verification: 26 focused tests and 92 complete project tests passed; changed Step 9 Python compiled; 297/297 raw files remained hash-identical, 288/288 selected images matched their manifest, and all 576 generated Step 9 masks re-opened as hash-matching source-size binary PNGs.
- Step 10 sparse-SfM design, implementation plan, `sparse_reconstruction.py`, `run_sparse_reconstruction.py`, focused tests, and `pycolmap>=4.2,<5` dependency.
- Real pyCOLMAP 4.2.0 CPU sparse run over all 288 PREPROCESSED inputs using native unmasked SIFT, one shared `SIMPLE_RADIAL` camera, internal `max_image_size=1200`, sequential matching, and incremental mapping.
- Step 10 baseline overlap 20: 1,255,153 SIFT features, 1,500 non-empty matched pairs, 902 verified pairs, seven sparse models, 216-image union coverage; largest component 73 images / 6,099 points / 1.2373 px mean reprojection error.
- Single controlled overlap-40 retry: seven sparse models, 223-image union coverage; largest component remained 73 images / 5,769 points, so the frozen ranking retained the baseline component.
- Selected sparse COLMAP model plus PLY export under `reconstruction/sparse/best/`, machine-readable Step 10 reports, and two visually inspected sparse/registration figures.
- Step 10 records `acceptance_met=false`: the local sparse reconstruction is plausible, but the full sequence remains fragmented, so dense reconstruction was not started and no success claim was made for a global 288-image model.
- Final Step 10 verification: 11 focused tests and 103 complete project tests passed; changed Step 10 Python compiled; the selected sparse model re-opened with matching summary metrics; 297/297 raw files and 288/288 selected files remained verified; transient COLMAP databases/caches were removed while preserving the sparse models and evidence.
- Step 11 sparse-component-bridging design, implementation plan, shared Step 10 pyCOLMAP option/runtime helpers, `sparse_bridging.py`, `run_sparse_bridging.py`, and focused tests.
- Deterministic non-local diagnosis of exactly 2,340 pairs: 780 around each fixed boundary 73-74, 145-146, and 203-204. The first two boundaries produced zero geometrically verified candidates; 203-204 produced 68 qualified candidates and 8 selected bridges.
- Fail-closed targeted gate: targeted mapping was skipped because every boundary required at least one selected qualified bridge.
- Exactly one CPU exhaustive fallback with block size 50: 14,900 non-empty match rows, 3,020 geometrically verified rows, eight sparse models, and 224-image union coverage.
- Selected Step 11 sparse model plus PLY under `reconstruction/bridging/best/`, seven machine-readable reports, and three visually inspected candidate/sparse/registration figures.
- Step 11 records `bridge_success=false`: the strongest single model remains 73/288 images with 3,443 points and 1.1989 px mean reprojection error, so disconnected-model union coverage is not misreported as a global reconstruction and dense work remains blocked.
- Final Step 11 review verification: 32 focused tests and 141 complete project tests passed; changed sparse modules compiled; both Step 10 and Step 11 models re-opened with matching metrics; 297/297 raw files and 288/288 selected files remained verified.
- Post-implementation review hardened feature-cache and interrupted exhaustive-resume identity validation using exact image/camera/keypoint/descriptor layout, and fixed the zero-inlier bridge-candidate figure so its scale cannot display impossible negative inlier values. These fixes do not change the measured Step 11 reconstruction result.
- Step 12 learned sparse-recovery design and implementation plan, plus the recovered partial implementation in `learned_sparse_recovery.py`, `run_learned_sparse_recovery.py`, and focused tests. The bounded native frontends remain ALIKED-N16Rot + LightGlue first and LoMa-B + LoMa-L as the single conditional fallback, both CPU with `max_image_size=1600`.
- Step 12 maintenance review verified the already-reached domain/runner boundary only: 20 domain tests, 14 runner tests, and 49 Step 10/11 regression tests passed. No real learned extraction, matching, mapping, visual acceptance, final runtime report, or dense reconstruction was executed.
- Real Step 12 capability execution on Python 3.14.2 / pyCOLMAP 4.2.0. Static ALIKED/LoMa options passed, but the ALIKED-N16Rot smoke on selected indices 1-2 stopped with `RuntimeError: ALIKED feature extraction requires ONNX support.`
- Truthful Step 12 blocked-result artifacts under `reconstruction/learned_recovery/`: capability, ALIKED attempt, attempts table, final summary, and a visually inspected strongest-single-model SIFT comparison. No learned candidate table, learned sparse model, PLY, or model-only figure was fabricated.
- Measured Step 12 report `docs/geometry-ml/learned-sparse-recovery.md`; `learned_recovery_success=false` and `dense_reconstruction_started=false`.
- Final Step 12 verification after review: 20 domain tests, 16 runner tests, 49 Step 10/11 regressions, and 177 complete project tests passed; touched Python compiled; Step 10/11 models and five frozen report hashes remained unchanged; 297 raw and 288 selected images reverified; compile/transient residue was removed.
- Project instructions now record `.codegraph/` as installed CodeGraph state and allow automatic use of relevant skills/plugins plus GLM-only subagents when materially useful.
- Step 13 external learned global-recovery design, implementation plan, pinned official LightGlue dependency, external ALIKED/LightGlue adapter, COLMAP external-feature import path, staged runner, focused tests, durable reports, and four real review figures.
- Real Step 13 capability smoke on CUDA: 4,096 ALIKED-N16Rot features per smoke image and 2,952 LightGlue raw matches using the official package pinned to commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`.
- Exact 2,340-pair learned boundary diagnosis recovered every critical boundary: 778 qualified candidates at 73-74, 418 at 145-146, and 745 at 203-204, with eight selected bridges per boundary.
- The one permitted 5,574-pair full learned mapping attempt produced a strongest single model with 266/288 registered images, 29,713 points, 106,480 observations, one `SIMPLE_RADIAL` camera, and 1.374824 px mean reprojection error.
- Step 13 retained the 266-image model as measured evidence but recorded `step13_success=false` because the frozen >=274-image gate was missed by eight images. The verified Step 10 73-image / 6,099-point model is selected as the local downstream fallback and sparse recovery is closed.
- Final Step 13 verification: 21 focused tests and 198 complete project tests passed; four touched Python files compiled; 297 raw and 288 selected images reverified; five protected Step 10/11 report hashes remained unchanged; four real Step 13 figures were visually inspected; transient Step 13 work was removed.
- Steps 14-17 restartable local dense pipeline, PLY/component/UV validators, headless Blender renderer, focused tests, design/plan/handoff, durable reports, and real previews.
- Exact 73-view Step 10 dense workspace with verified poses/intrinsics and official COLMAP 4.2.0 CUDA plus Blender 5.2.0 LTS capability evidence.
- One preferred CUDA PatchMatch run completed 73 depth/normal maps; geometric fusion produced 391,899 finite colored points, 64.26 times the sparse source, with 99.9980% inside the expanded Step 10 bounds.
- Poisson primary mesh with 2,040,189 faces, one rejected Delaunay giant-shell alternative, deterministic 0.5%-face component filtering that retained 96.64% of faces, and one measured-ratio QEM simplification to 499,999 faces.
- One COLMAP photo-texture attempt with exact topology preservation, a 4096 x 1902 atlas, 73.00% meaningful UV coverage, and a three-view headless Blender validation render.
- Integrated `steps14_17_success=true` result with `blender_manual_cleanup_started=false` and a measured report at `docs/geometry-ml/local-dense-mesh-texture.md`.
- Final Steps 14-17 verification: 49 focused tests, 106 Step 10-13 regressions, and 247 complete project tests passed; changed Python compiled; the real `all` stage passed; 297 raw, 288 selected, 73 local, and 160 protected Step 10-13 files remained exact; final artifacts reopened; documentation and cleanup checks passed.
- Final V2 Plan 1 CV evidence/model-fit implementation with 16 canonical registered views, source-photo-refined geometry masks, reviewed component landmarks, Step 6 SIFT/RANSAC diagnostics, bounded ALIKED/LightGlue detail-alignment evidence, Step 13 camera reuse, component masks, deterministic profile fitting, and Blender-consumable `final_profiles.json`.
- Accepted V2 Gate B candidate: median whole-object silhouette IoU 0.901495, minimum reliable-view IoU 0.856102, median landmark error 0.011306 object height, p95 landmark error 0.038080, with candidate-bound bowl/globe/neck/lid/finial visual review passing.
- Final Plan 1 verification after review: changed modules compiled; 59 focused V1/V2 IO/reference/CV-fit/orchestrator tests passed; the real `analyze` stage regenerated accepted reference evidence; the real `cv-fit` stage reproduced the accepted Gate B metrics.
- Added downstream V2 CV-continuity requirements: distortion-consistent Step 13 camera comparisons, a non-canonical registered-view silhouette generalization audit, Step 6 classical ring/axis cross-checks, and `surface_evidence_coverage.json` provenance for direct versus inferred surface completion. Plan 5 now also requires bounded overlap-based photometric harmonization before robust multi-view texture fusion so exposure/white-balance variation does not become texture seams. The orchestrator blocks ornament and later stages until the Plan 2 coverage reports are explicitly accepted; the updated V1/V2 IO/reference/CV-fit/orchestrator suite passes **61/61** tests.
- Completed corrected Plan 2 validation: restored the accepted `30_BASE_GEOMETRY_ACCEPTED.blend` binding, made non-canonical classification fail closed, derived surface coverage from visibility, bound reliable Step-6 candidate projections, generated a true wireframe review, and accepted the audit at **77 OK / 63 camera failures / 24 mask failures / 0 usable model mismatches**.
- Completed Plans 3-5 and pre-export Plan 6: source-supported ornament families, practical topology/normal cleanup, shared UVs, 2048² AO/tangent-normal/curvature bakes, photo-informed texture fallback, polished warm-brass lookdev, representative source-camera/detail renders, wireframe/UV QA, and `Thai_Libation_Vessel_FINAL.blend` at the user-inspection gate.
- Corrected the source-visible globe/receiving-bowl relationship through two explicit user-review passes while keeping the frozen Plan-1 profile JSON unchanged. The downstream finalization profile now reshapes only the globe/shoulder radial envelope to a compact ellipsoid: globe max radius **0.15989** instead of **0.20073**, receiving-bowl/globe max-radius ratio **1.30**, and rolled-rim radial clearance about **0.0604**. All accepted Z levels, axis, neck junction, lower bowl profile, pedestal, lid, finial, and component positions remain fixed.
- Added a separate source-supported `SM_Globe_LowerSupport` plus two small construction rings so the narrow lower collar visible inside the bowl in all four close oblique reference photographs is represented as real geometry rather than empty space or a fused globe extension.
- Final pre-export inspection state passes zoomed high-oblique, top, globe-close, and quarter visual review; the focused Final-V2 Blender/CV/orchestrator suite passes **92/92**, and `final_validation_report.json` remains accepted with `export_performed=false`.

### Fixed
- Fail preprocessing before creating generated outputs when the configured expected raw-image count disagrees with the verified baseline.
- Treat a degenerate OpenCV fundamental-matrix fit as zero geometric inliers instead of crashing the matching experiment.
- Correct stale phase/test-count documentation and use the cache-free pytest command in the reproduction steps.
- Removed duplicated ALIKED capability interpretation from the Step 12 `all` orchestration path. The diagnose stage now owns capability-blocked report semantics, fixing the two stopped-session runner failures and preventing the top-level state machine from diverging from stage behavior.
- Hardened Step 12 learned-feature cache reuse with a SHA-256 fingerprint of the exact per-image database layout, so stale caches with unchanged aggregate feature totals are rebuilt rather than accepted.
- Carried the ALIKED native capability blocker into LoMa's `not_run` final-report reason instead of using the ambiguous phrase “fallback was not required”; that runtime-path regression raised the Step 12 runner suite to 15 tests.
- Preserved frontend/extractor/matcher identity in blocked and explicit `not_run` attempt reports; the final report-contract regression raises the Step 12 runner suite to 16 tests.
- Step 13 diagnostic reports now distinguish LightGlue correspondences before geometric verification (363,318) from COLMAP match rows after verification (363,171), preventing ambiguous raw-match reporting.
- Step 13 feature-cache validation now rejects any external learned feature set whose coordinate frame does not match the source image dimensions, making the original-pixel COLMAP import contract explicit and regression-tested.
- Steps 14-17 now persist visual rejection by artifact/preview hash, record failed mesh subprocesses before using the one alternative, and permit only measured resource/runtime dense fallback categories.
- Dense plausibility now checks the robust 0.1-99.9 percentile core plus at least 99% expanded-source-box coverage while preserving full bounds and full-cloud previews.
- Textured-asset validation now compares exact face connectivity, requires at least 50% meaningful UV coverage, isolates retry outputs, and invalidates cached renders when mesh, atlas, manifest, renderer, Blender, or preview hashes change.
- Headless Blender uses its verified 5.2 render-engine identifier, creates a world after factory reset, exits nonzero on Python errors, and must produce reopenable preview/report artifacts before the stage can pass.
- The `all` CLI path now calls the same tested gate-owning `run_sequence` helper used by orchestration tests.
- Windows interruption now terminates the complete launched subprocess tree, and restart protection checks the recorded parent plus surviving descendants.
- Final-mesh and texture-attempt paths now reject unrecorded existing content before any overwrite, while textured-asset validation rejects finite UVs outside the normalized atlas range.
- Step 16 machine reports and documentation now disclose when a large-mesh preview uses sampled vertices; Step 17 still imports and renders the exact final triangle surface in Blender.
- Windows process-tree cleanup now isolates 	askkill from host stdio with DEVNULL, preventing WinError 50 in MCP/headless hosts while preserving timeout and descendant termination.

- Final V2 reference-evidence acceptance is now fail-closed on an explicit visual review bound to the exact current geometry-mask and landmark contact-sheet hashes; regenerated reports no longer remain accepted while those reviews say `pending`.
- `run_final_model.py --stage all` now intentionally stops at pre-export `final-validate`, and the explicit `export` stage requires a user-approval sidecar before any GLB/export/re-import work can begin.

### Removed
- Course-presentation DOCX/PDF walkthrough artifacts after delivery, while retaining all measured preprocessing reports, contact sheets, previews, and reconstruction-input evidence.
- Three task-created per-camera Blender render intermediates after composing the final Step 17 triptych.

### Next
- Preserve the accepted pre-export Final V2 inspection asset and wait for explicit user approval before running any GLB export/re-import/promotion/publication step.
- If the user requests additional visual corrections, keep them bounded to source-supported defects and re-run only the affected downstream Blender stages; do not reopen frozen sparse/dense experiments without new blocking evidence.
- Keep the truthful Step 10 local-dense limitations, historical CNN/SfM evidence, Plan-2 audit classifications, and component-level photo-texture fallback visible in coursework reporting; V2 is CV-constrained + Blender-completed, not direct complete photogrammetry.
