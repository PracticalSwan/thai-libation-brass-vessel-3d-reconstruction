# Lessons Learned

Read this after `AGENTS.md` when starting substantive work. Keep process lessons here; keep current project state in `docs/memory-bank/`.

## 2026-08-27

- Do not over-engineer, over-complicate, or over-test. The coursework benefits from simple, explainable code and verification proportional to actual risk.
- Numerical image-quality thresholds are not authoritative by themselves. Validate them against the real capture distribution and visual coverage before rejecting frames.
- Polished brass naturally produces moving highlights; reflection alone is not a rejection reason.
- Raw smartphone images are immutable source evidence. Derived data must live outside the raw directory, and cleanup must never touch the originals.
- Contact-sheet review found that the final hand-held/flipped sequence changes object pose/background relation and should not be treated like the fixed-object SfM orbit without explicit justification.
- Choose a reconstruction input variant from the exact exported artifact's geometric correspondence evidence, not visual preference or an in-memory approximation. In the final ten-pair experiment, the mild quality-95 JPEG preprocessing produced more total verified inliers and was non-worse on 9 of 10 pairs.

## 2026-09-06

- Keep one owner for orchestration gates. Step 12's `all` stage duplicated ALIKED capability interpretation that already belonged to the diagnose stage, causing state-machine drift and two failing orchestration tests. Let each stage own its report semantics and let the top-level runner branch only on stage results.
- Cache provenance must identify the exact per-image database layout, not only aggregate feature counts. A learned-feature database can preserve the same total while changing image-level keypoint/descriptor rows; persist a deterministic layout fingerprint before reusing it.
- Learned pyCOLMAP matching must pass the chosen `FeatureMatchingOptions` explicitly to both sequential and imported-pair matching. Omitting either call can silently fall back to SIFT and invalidate the learned-recovery experiment.
- On Windows tests that rebuild SQLite files, explicitly close fixture connections before unlinking or replacing the database; transaction context management alone does not close the connection.
- Unit/orchestration tests validate the Step 12 control contract, not real learned reconstruction quality. Do not report ALIKED/LoMa recovery results until the native runtime stages are actually executed and measured.
- External learned matching can recover pairwise boundaries and still miss a frozen global-model acceptance target. Step 13 recovered all three critical boundaries and raised the strongest model from 73 to 266 images, but the >=274 gate still failed; do not move the threshold after seeing the result.
- Report imported learned correspondences before geometric verification separately from COLMAP match rows after verification; they are different quantities even when only a small fraction is removed.
- External feature caches must explicitly validate their coordinate frame against the source image dimensions before COLMAP import; checking only finite/in-bounds keypoints is not sufficient provenance.
- Dense full-coordinate minima and maxima are too sensitive to a few stereo outliers to serve as the only plausibility gate. Keep the full bounds and preview visible, then measure a robust percentile core and an explicit source-relative coverage fraction; this run had only 8 of 391,899 points outside the expanded Step 10 box.
- A subprocess exit code is not sufficient evidence for scripted graphics validation. Blender initially returned success while its Python script had failed; use `--python-exit-code`, remove stale owned preview/report files before rerendering, and require both outputs to reopen.
- Keep expensive stage outputs restartable by isolating attempt-owned directories and binding reuse to the current source, artifact, preview, tool, and renderer hashes. Persist visual rejection on the candidate itself so a restart cannot silently reconsider a failed mesh or create another alternative.
- Mesh component ratio is evidence, not a visual substitute. The Delaunay candidate had an 89.50% dominant component but formed giant unsupported sheets, while the visually plausible Poisson surface was fragmented by the partial capture. Use the metric and full-bound preview together, then apply only a measured deterministic cleanup rule.
- A parent-process PID is not enough for restart safety around native tools that may spawn workers. Terminate the full Windows process tree on interruption, check descendants before restart, and reject generated-output paths whose ownership is not bound by a persisted report.
- Describe large-geometry previews at their actual evidence level. A deterministic sampled-vertex view can establish bounds, silhouette, and obvious sheets, while an exact mesh surface claim requires importing or rendering the triangle topology itself.
- Cleanup subprocesses should not inherit console handles when their output is unused. MCP/headless hosts can expose unsupported stdin handles; route helper-process stdio to DEVNULL so timeout cleanup remains reliable.
- Technical mesh validity is not visual identity. V1 reached zero reported non-manifold mesh objects and aggregate silhouette IoU 0.8060 yet still looked unlike the photographed artifact; final presentation reconstruction must use component-level multi-view CV constraints and source-vs-render visual vetoes rather than treating manifold topology or one aggregate silhouette metric as sufficient.

## V2 projection fitting: silhouette extrema are not axis-center endpoints

For tilted cameras, the visible top/bottom of a finite-radius revolved component is the projected profile/ring extremum, not the projection of the 3D axis-center endpoint. Likewise, paired semantic left/right landmarks must be evaluated against the named component's projected tangent/extrema rather than the whole assembly silhouette. Confusing these quantities produced systematic elevated-view vertical offsets and inflated globe landmark errors. The accepted Plan-1 fit uses profile-extrema semantics and component-aware lateral landmarks; preserve this distinction in Blender/source-camera validation.
