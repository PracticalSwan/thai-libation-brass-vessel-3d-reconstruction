# V4 Full Reconstruction Repair Design

**Status:** Canonical V4 repair design as of 2026-09-14, with the 2026-09-15 execution-ownership override below. This supersedes the dense-only repair assumption and the earlier fast end-to-end continuation checkpoint for all new V4 work.

## Goal

Complete V4 end-to-end from the newest verified repository state with the strongest defensible genuine computer-vision result realistically recoverable from the captured evidence, but execute it in two ownership phases. The current Codex/local phase ends at a verified, versioned, hash-bound Poisson + evidence handoff. The later ChatGPT + Blender MCP phase performs Blender and everything after Blender. Use bounded evidence-backed repair/escalation ladders; strict research-grade gates remain preferred measurements but must not cause indefinite candidate churn. Never reuse incompatible dense maps, falsely relabel a failed gate as passed, or fabricate missing vessel-scale anatomy.

## Execution ownership boundary — 2026-09-15

- The **pre-Blender owner (Codex/local executor)** must recover persisted state after restart, complete remaining dense/Poisson work, correct/verify lineage, tests, hashes and pre-Blender docs, and complete the focused pre-Blender Git milestone.
- The handoff artifact is the strongest defensible scan-derived Poisson plus exact sparse/dense/Poisson provenance, reports, hashes and residual failures.
- Once that handoff is ready, the pre-Blender owner must stop and must not touch `.blend`/GLB scenes/artifacts, Blender-specific finalization, appearance/material finalization, fresh GLB re-import, or post-Blender publication.
- **ChatGPT + Blender MCP** owns the Blender cleanup, LOD0/UV/normal-detail/AO, appearance/material, final `.blend`/GLB, fresh re-import, visual inspection, final docs and final publication.
- A reboot or disconnected tool session invalidates assumptions about an in-flight compute job, not the persisted evidence. Inspect logs/workspaces/artifacts first and rerun only if necessary.

## Non-negotiable boundaries

- The immutable source set is `CSX4213_Project_V4_Images/`: 688 JPEGs = 158 uncoated appearance/reference, 107 empty-board/background, 423 coated/marked geometry images.
- Do not modify, rename, recompress, rotate, move, or delete those source JPEGs.
- Preserve `CSX4213_Project_V4_Images/`, the current `capture_v4/derived/` processed-image trees, `.codegraph/`, unrelated `.ai-bridge/`, unrelated user-owned work, and all current V4 reconstruction evidence. Superseded pre-V4 raw media/checkpoints are removed and must not be restored.
- Only the V4 capture lineage is an active reconstruction input; superseded pre-V4 reconstruction versions are not part of the active repository.
- No new capture is available.
- No Codex CLI.
- Use the existing ALIKED-N16Rot + LightGlue evidence path, COLMAP/pyCOLMAP, CUDA PatchMatch on GPU 0, Poisson surface reconstruction, and Blender 5.2.
- This is a full computer-vision pipeline: every accepted geometric and appearance result must be produced algorithmically from the project-captured image set and measured reconstruction evidence. External/reference-assisted modeling, CAD/profile tracing, manual sculpting/vertex editing, symmetry/lathe/primitive replacement, artist-authored shape completion, manual reference matching, and hand-authored texture/color fallback are prohibited.
- Blender is only a scripted/reproducible technical post-processing and export stage. It may remove measured noise/floaters/degenerates, repair normals, generate LOD/UV/bakes/material nodes from project-derived data, and export/verify; it may not create or reshape vessel anatomy.
- No hand-built replacement neck/lid/finial/bowl/pedestal, symmetry reconstruction, lathe replacement, or other synthetic vessel-scale substitute.
- The 158 uncoated appearance inputs are appearance-only. They cannot alter reconstructed geometry and may contribute appearance only through automated localization/projection/blending or reproducible dataset-derived statistics.
- Every repair candidate is versioned. Never overwrite the historical accepted sparse/dense/Poisson/Blender/GLB artifacts while experimenting.
- Exact staging only for Git milestones. Never use `git add .`; never force-push or rewrite history.

## Historical baseline to preserve, not re-accept blindly

The previous V4 path produced the following real artifacts and evidence:

- 372 selected geometry views, 372/372 registered in one sparse model.
- Sparse model: 65,560 points, 460,628 observations, mean track length about 7.026, mean reprojection error about 1.223 px, one `SIMPLE_RADIAL` camera model.
- Corrected 372/372 COLMAP fusion-mask contract: `<image_name>.png`.
- Historical accepted dense cloud: `reconstruction/v4/dense/fused_workspace_tiled6_g8g9_priority_true3_maskfix.ply`.
- Historical dense SHA-256: `df05019e2e56d1c54351f4b2cee161cbcfb7b782a3c6b0d4920d6e303f39d7d8`.
- Historical dense size: 1,426,482 finite colored points, rank 3, normals present.
- Historical raw Poisson: 887,770 vertices / 1,669,931 faces, SHA-256 `55a4e92c9491cced508360d1645355ed447785f175df0556b400528fa9522941`.
- Historical canonical `.blend` and `.glb` remain preserved baseline outputs.

These facts prove the pipeline ran. They do **not** prove the geometry is visually complete or that the sparse cameras are all correct.

## Confirmed diagnosis that changes the repair architecture

### 1. `geo_g10` contains real sparse pose inconsistencies

The prior assumption that sparse reconstruction was already accepted is no longer valid. Read-only comparison of final SfM camera poses against the already-verified adjacent two-view geometry found concrete contradictions:

- `IMG20260912145220.jpg -> IMG20260912145223.jpg`: about 449 verified two-view inliers; two-view relative rotation about 15.9 degrees; final sparse relative rotation about 103.8 degrees; final shared 3D tracks = 0.
- `IMG20260912145330.jpg -> IMG20260912145333.jpg`: two-view rotation about 11.8 degrees versus final sparse about 61.5 degrees; final shared tracks only 7.
- `IMG20260912145254.jpg -> IMG20260912145256.jpg`: two-view rotation about 14.7 degrees versus final sparse about 32.2 degrees; final shared tracks only 10.

The reconstructed `geo_g10` trajectory also contains an acquisition-adjacent virtual-camera ray jump around 82.6 degrees. The corresponding source frames are ordinary neighboring turntable images, so this is not credible real capture motion.

The sparse report also records repeated CHOLMOD matrix-not-positive-definite warnings during final bundle adjustment. Those warnings are supporting evidence, not by themselves the diagnosis.

**Conclusion:** full registration count and global mean reprojection error were insufficient acceptance tests. Sparse integrity must be repaired before dense work is treated as authoritative.

### 2. Raw `phase_01` is only ordered sequence phase, not measured physical azimuth

`phase_01` is derived from ordered frame position within each logical ring. It is useful for deterministic ordering and approximate neighborhood construction, but it is not a measured angle. Some reconstructed ring trajectories imply more than one virtual revolution or abrupt phase/pose discontinuities when interpreted literally.

Cross-ring offset confidence was also weak on critical transitions. Therefore repair logic must not force cross-ring pair/source choices merely because normalized phases are close. Verified inlier support, two-view geometry, shared-track evidence from an accepted sparse candidate, acquisition order, and camera-geometry sanity must dominate source selection.

### 3. Dense geometric consistency collapses exactly where sparse geometry is suspect

Measured valid depth support inside aligned vessel masks for the historical true3 workspace:

| Ring | Geometric median | Geometric p10 | Geometric min | Photometric median |
| --- | ---: | ---: | ---: | ---: |
| `g7_r1` | ~0.947 | ~0.922 | ~0.820 | ~1.000 |
| `geo_g8` | ~0.890 | ~0.851 | ~0.744 | ~1.000 |
| `geo_g9` | ~0.867 | ~0.825 | ~0.725 | ~1.000 |
| `geo_g10` | ~0.659 | ~0.142 | ~0.003 | ~1.000 |
| `geo_g11` | ~0.590 | ~0.299 | ~0.006 | ~1.000 |
| `geo_g12` | ~0.337 | ~0.034 | ~0.021 | ~1.000 |

Photometric hypotheses exist over the vessel, while geometric-consistency support collapses in the upper/high rings. Dense tuning alone cannot be trusted until the sparse geometry feeding PatchMatch is repaired.

### 4. Historical dense tiling permitted duplicate reference rewrites

The combined true3 PatchMatch configs contain 567 reference entries for only 372 unique views; 158 references appear more than once, with some up to four times. That permits later tiles to overwrite earlier maps for the same reference and makes the historical final map provenance ambiguous.

The repair dense runner must enforce **exactly one final PatchMatch reference assignment per registered image per candidate**. Dependency/source images may appear in many configs, but a reference image may be written only once in a complete candidate run.

### 5. Masks and gross background contamination are not the primary defect

Representative MVS masks preserve the real vessel silhouette, including the neck/lid/finial. The corrected StereoFusion resolver finds all 372 masks. Historical corrected-mask contamination measurements were low: board about 0.78%, cloth/background about 0.049%, pedestal-board webbing 0, with no dominant board slab/background curtain.

Mask repair remains allowed only for a view proven to truncate real vessel pixels; broad remasking is not the first repair step.

### 6. The historical Poisson is objectively fragmented

The historical raw Poisson has 644 connected components. The largest component contains only about 66.57% of faces, with two additional very large components. The old raw visual gate still passed because it lacked strict continuity/anatomical-completeness checks.

**Conclusion:** old dense/raw/Blender acceptance is superseded for the user's visually-complete-vessel requirement.

## Repair architecture

### A. Preserve historical outputs and create a repair namespace

New compact repository outputs go under:

- `reconstruction/v4/repair/`
- `reconstruction/v4/reports/repair_*`
- `reconstruction/v4/previews/repair_*`

Large transient sparse/dense workspaces go under versioned scratch directories such as:

- `D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_*`

No candidate may overwrite the historical sparse model, true3 dense maps, historical fused cloud, historical Poisson, canonical `.blend`, or canonical `.glb` until final promotion has passed every repair gate and the old canonical bytes have been preserved/hash-recorded.

### B. Add strict sparse-integrity evidence before reconstruction compute

A repair sparse gate must measure at minimum:

1. **Adjacent two-view consistency:** compare final candidate relative camera rotation against calibrated relative rotation recoverable from verified adjacent same-ring two-view geometry where it is well-conditioned.
2. **Final shared-track continuity:** acquisition-neighbor pairs with strong verified match support must not collapse to zero/near-zero shared final 3D tracks without an explicit reason.
3. **Trajectory continuity:** use acquisition order and robust within-ring statistics to flag catastrophic camera-center/ray jumps. Do not impose a fabricated fixed degree-per-frame rule.
4. **Registration/ring coverage:** useful registered coverage across all intended rings.
5. **Track/reprojection health:** finite points, healthy observations/track lengths, bounded reprojection distribution, no isolated camera islands.
6. **Cross-ring support:** at least enough strong, geometrically sane cross-ring constraints to bind the ring stack without forcing weak phase-nearest edges.
7. **Camera/intrinsics integrity:** shared camera only when the real imaging contract supports it; no silent camera-model drift.

The historical sparse model must fail the new integrity regression because of the proven `geo_g10` contradictions.

### C. Sparse repair ladder

The objective is not to restart image ingest, segmentation, or feature extraction. Reuse the immutable source, masks, ALIKED feature cache, LightGlue matches, and verified two-view database when their hashes/contracts still match.

Repair in the following order:

1. **Diagnose exact bad pose segments and bridge constraints.** Produce a hash-bound report containing per-ring acquisition-adjacent final relative rotation, two-view rotation where recoverable, discrepancy, final shared-track count, verified inlier count, and robust trajectory-outlier classification.
2. **Build a corrected sparse pair/constraint graph.** Prefer strong acquisition-local same-ring verified pairs and wider same-ring neighbors. Select cross-ring anchors by verified inlier support + camera geometry + evidence-backed phase/order compatibility. Do not force weak cross-ring pairs to meet a quota.
3. **Attempt bounded local/surgical pose recovery only if the existing APIs can do so without mutating the historical model and without freezing bad geometry.** Good existing cameras may be used as anchors only when their own integrity checks pass.
4. **If surgical recovery is not clean or cannot pass the sparse integrity gate, create a fresh versioned sparse candidate from the corrected graph using the existing cached features/matches.** This is the preferred safe fallback rather than preserving a known-bad pose solution.
5. **If multiple evidence-preserving incremental-mapper candidates remain order-sensitive or retain pose islands, change reconstruction architecture rather than accumulating mapper-order toggles.** The installed pyCOLMAP 4.2 `global_mapping`/GLOMAP path is the preferred bounded next candidate because it performs robust global rotation averaging and positioning from the same image-derived verified graph. Run it only on disposable/versioned databases, preserve the fixed independent audit unchanged, and require the same exact 372-view/one-model/pose-integrity gate. A custom turntable/orbit prior is a later fallback only if it is estimated entirely from project-image geometry and independently justified.
6. **Bundle-adjust and re-audit.** Reject any candidate that preserves a `geo_g10` pose island, creates new catastrophic ring discontinuities, or materially damages previously healthy rings.

Strict sparse integrity remains the preferred promotion path. After the final bounded sparse repair attempt, however, a hash-bound `best-defensible` sparse candidate may feed a fresh dense run if its failed checks remain explicitly failed, its selection comparison/provenance is recorded, and no incompatible historical dense evidence is reused. This is a completion fallback, not a strict-gate pass.

### D. Dense maps must be regenerated when camera geometry changes

If the accepted sparse repair changes camera extrinsics, intrinsics, undistortion, or reconstruction scale/coordinate contract, the historical true3 geometric/photometric maps are incompatible evidence and cannot be reused as final maps. Re-undistort images/masks consistently and generate a new versioned dense workspace.

The dense runner must enforce:

- one registered image = one final reference assignment;
- all expected reference names covered exactly once;
- no duplicate reference overwrites across tiles;
- explicit source list per reference;
- full config/hash lineage;
- CUDA GPU 0 only for PatchMatch; no silent CPU fallback.

### E. Source selection after sparse repair

Default repaired source strategy, bounded by evidence:

- Prefer approximately six strong same-ring sources when available.
- Use acquisition order/phase only as a locality prior, not physical angle truth.
- Add at most two adjacent-ring sources only when verified/candidate-sparse co-visibility and camera geometry support them.
- Do not force zero/near-zero shared-track cross sources.
- Stay inside existing verified pair evidence where practical; expand only with a documented reason and new geometric verification.

### F. Dense recovery ladder

Start with the smallest trustworthy candidate. Stop escalating when strict dense + raw-mesh gates pass **or** when the bounded evidence-backed ladder is exhausted and the strongest defensible genuine result can be selected without falsifying any failed check.

1. **Default 2000px repaired geometric PatchMatch** using the accepted repaired sparse model, corrected masks, repaired source graph, geometric consistency, and default filtering.
2. **Filtering isolation only if measured support still collapses:** test one factor at a time, first filtering off, then `filter_min_num_consistent=1` if needed.
3. **Source-graph refinement only when source evidence shows the default repaired graph is still weak.**
4. **Fusion tuning only when maps are good but fused projected coverage is insufficient:** `min_num_pixels 5 -> 3`, then `max_depth_error 0.01 -> 0.02`, then `max_reproj_error 2 -> 3`, one change at a time.
5. **3072px high-ring recovery last:** bounded GPU smoke first; use only if thin upper geometry remains unsupported at 2000px and memory/runtime are safe.

Historical photometric maps may be used for diagnosis, but a photometric-fusion shortcut from the known-bad historical sparse poses is not eligible for final promotion.

### G. Dense acceptance contract

The selected repaired cloud should satisfy all of the following strict targets. If the bounded dense ladder is exhausted, the strongest defensible genuine cloud may be selected with the failed targets preserved explicitly in its lineage and final limitations report:

- finite rank-3 cloud with colors/normals;
- exact expected mask resolution;
- one-reference-one-write dense provenance;
- accepted sparse candidate hash/pose lineage;
- ring depth-support evidence with no catastrophic high-ring collapse;
- projected vessel-mask coverage measured through stratified registered cameras;
- corrected cross-view/ring-transition evidence;
- board/background/webbing contamination within existing clean limits;
- at least eight hash-bound visual views;
- explicit anatomy findings for bowl shell/interior, rolled rim, globe/shoulder, continuous neck, lid tiers, finial, pedestal transitions, and base;
- `no_major_vessel_scale_holes=true`.

For geometric candidates, a useful quality target remains per-ring median support >= 0.75 and p10 >= 0.40, but numeric thresholds never override visible wrong geometry. A candidate with more points but doubled/fragmented anatomy is rejected.

### H. Poisson remains required and gets a hard continuity gate

Run Poisson depth 13 first on the accepted repaired dense cloud. Every repaired raw candidate is versioned.

Acceptance requires:

- finite nonzero mesh;
- dominant face component >= 0.95;
- second-largest face component <= 0.02;
- no unsupported large detached component;
- no major vessel-scale hole in bowl/rim/interior, body, neck/lid/finial, pedestal/base;
- contamination remains clean;
- at least eight hash-bound raw-mesh views.

The historical 644-component / ~0.6657-dominant Poisson is a regression fixture that must fail.

If Poisson misses strict thresholds, make only the bounded evidence-backed retry justified by the measured cause. Once that bounded ladder is exhausted, choose the strongest connected scan-derived Poisson, keep every failed threshold/visual limitation explicit, and freeze it for the pre-Blender handoff. The current Codex/local executor stops there; ChatGPT + Blender MCP owns the later Blender stage. Do not bridge missing vessel-scale anatomy in Blender.

### I. POST-BLENDER OWNER: ChatGPT + Blender MCP — Blender remains scan-preserving and non-creative

Only after a repaired raw Poisson is strictly accepted **or** the bounded Poisson ladder is exhausted and the strongest defensible connected scan-derived mesh is explicitly frozen with its failures documented:

1. Import and preserve the repaired raw mesh as immutable source evidence/rollback.
2. Duplicate to clean-high through a scripted/reproducible procedure.
3. Remove only algorithmically identified isolated floaters/noise, impossible internal debris, duplicate/degenerate geometry, normal problems, and tiny defensible defects. Manual vertex/face selection or sculpting may not be used to repair shape.
4. Compare at least eight clean-high views to the accepted raw mesh; reject material silhouette/anatomy loss.
5. Create LOD0 with enough topology to preserve silhouette and anatomy using reproducible technical processing.
6. UV unwrap and bake scan-supported normal/detail and AO through scripted/reproducible steps.

No whole-object remesh, sculpt, manual mesh edit, symmetry, lathe/revolve, primitive replacement, CAD/profile trace, or external-reference construction may manufacture or reshape neck/lid/finial/bowl/pedestal geometry. If vessel-scale anatomy is missing, return to the CV reconstruction stages.

#### V7 scan-preserving authoring override — 2026-09-16/17

For the later V7 refinement, the user explicitly authorized localized cleanup/sculpt/smooth operations **on the reconstructed scan/dense mesh itself** and localized same-object donor reuse where supported by scan evidence. This does not authorize generic replacement anatomy, profile/lathe reconstruction, primitives, external-reference construction, or rebuilt ornament. The rejected V113/V114 replacement-looking direction remains invalid.

The completed V7 object is `SM_V4_V139_SCAN_PRESERVING_FINAL`, derived from V94. It preserves the intentionally open 72-edge physical bottom boundary, uses guarded `SM_V4_Scan_CleanHigh` donor displacement only where supported, and finishes with topology-preserving scan-vertex smoothing/fairing across all inspected surface bands. Rotational repetition is used only as a bounded same-scan profile constraint; the terminal is repaired from existing vertices using local profile correction/fairing and robust sphere fitting. No remesh, primitive replacement, generic decoration, external/reference geometry, or wholesale component reconstruction is introduced. V139 has been promoted to the canonical `.blend`, exported to the canonical GLB, and verified by fresh Blender 5.2 re-import.

### J. POST-BLENDER OWNER: ChatGPT + Blender MCP — Automated project-image-derived appearance reconstruction

Attempt automated auxiliary localization of the 158 uncoated project images against the accepted repaired geometry without modifying geometry.

If enough appearance cameras pass measured alignment/visibility checks, automatically project/blend photographic BaseColor while rejecting background, cloth/wood, coating/markers, occlusions, grazing views, and clipped highlights.

If photographic projection cannot be verified, fail closed to a reproducible appearance derived algorithmically from the project data: for example robust statistics over the eligible uncoated image set and/or reconstructed point/image colors. Set `photographic_projection_verified=false`. Do not hand-pick a reference image, manually match a reference, hand-tune appearance to an external/reference image, or use an artist-authored texture/color fallback.

Final material may include verified projected BaseColor or reproducible project-dataset-derived BaseColor statistics, data-derived Roughness, scan-derived Normal/detail, AO, and physically appropriate brass metallic response. Every appearance input and derivation must be recorded.

### K. POST-BLENDER OWNER: ChatGPT + Blender MCP — Final export and verification

The repaired final must be validated in a fresh Blender 5.2 process and fresh GLB import.

Required checks:

- exactly the intended repaired LOD0 exported;
- finite positions/normals and sane transforms/bounds;
- UVs/material/textures resolve;
- no raw/debug/camera/light objects in GLB;
- at least eight fresh-reimport views match authoring geometry materially;
- bowl/rim/interior, globe/shoulder, continuous neck, lid tiers, finial, pedestal transitions, and base remain intact after optimization/export;
- accepted sparse/dense/raw/Blender/GLB hashes are recomputed and reports point to the exact current bytes.

Only then may repaired outputs replace canonical V4 names. Preserve and hash the previous canonical `.blend`/`.glb` first so promotion is reversible.

## Git milestone policy

The user has authorized focused commit and push for major verified V4 milestones. Appropriate milestones for this repair are:

1. repaired sparse integrity accepted;
2. repaired dense accepted;
3. repaired raw Poisson accepted;
4. final Blender + GLB fresh-reimport accepted.

Before each milestone: inspect root/branch/status/upstream/intended diff, stage only milestone files, exclude source JPEGs/private weights/scratch dense workspaces/logs/backups/unrelated changes, push normally, then verify `origin/main` equals the intended commit. No per-task noisy commits.

For the current split, the raw-Poisson milestone is the Codex/local executor's final milestone. The Blender+GLB milestone is reserved for ChatGPT + Blender MCP.

## Pre-Blender handoff definition

The current Codex/local phase is done only when the strongest defensible sparse lineage is correct; fresh dense evidence is compatible and the bounded dense ladder is finished; the strongest connected scan-derived Poisson is selected and hash-bound; residual strict failures are preserved; relevant pre-Blender tests/compile/hash checks and documentation are current; task-created pre-Blender residue is cleaned; and the focused pre-Blender Git state is pushed/verified when appropriate. At that point the executor stops and hands the Poisson/evidence package to ChatGPT + Blender MCP.

## Definition of done

Overall V4 full reconstruction is complete when all of the following are true across both ownership phases:

1. the strongest defensible image-derived sparse source is hash-bound and its strict pass/fail state is recorded honestly;
2. fresh dense maps are compatible with that sparse geometry and one-reference-one-write provenance is proven;
3. the strongest defensible genuine dense cloud is selected from the bounded ladder with measured depth/coverage/consistency/contamination/anatomy evidence and all residual failures recorded;
4. the strongest defensible scan-derived Poisson is selected after bounded attempts, with component/hole limitations recorded rather than hidden;
5. Blender cleanup is scan-preserving and LOD0/UV/detail/AO are verified without invented anatomy;
6. appearance provenance is honest, automated/project-derived and measured;
7. final `.blend` and `.glb` pass fresh-process/fresh-import technical checks and at least eight-view review, with unrecovered anatomy documented precisely;
8. relevant tests/compile/integrity checks pass for the implemented pipeline and every report/hash binds to the actual artifact bytes;
9. docs/memory state reflect the final best-defensible result, unrelated work remains untouched, and final Git push/remote synchronization is verified.

Strict gate success remains preferable. If a bounded approach fails, continue only through the remaining evidence-backed ladder; when that ladder is exhausted, freeze the best genuine result and move downstream. Never declare completion by relabeling failures, reusing incompatible evidence, or fabricating geometry.