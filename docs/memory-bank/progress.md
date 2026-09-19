# Progress

Updated: 2026-09-17

## Current project state

The project is in **V4 full repair and V7 refinement**. The repository has been consolidated around the authoritative V4 capture and current V4 reconstruction lineage. Superseded pre-V4 raw images, preprocessing products, analysis/checkpoint trees, legacy pipeline code/tests, and obsolete implementation documents have been removed and must not be restored.

The verified V4 baseline remains preserved as rollback/evidence while the active V7 scan-preserving repair is evaluated. The current raw and processed image sets are intended Git LFS publication content.

## Completed investigation and planning

The following repair diagnosis is verified and must be treated as the new implementation boundary.

### Historical sparse baseline

```text
372 selected geometry views
372 / 372 historically registered in one model
65,560 sparse points
460,628 observations
~7.026 mean track length
~1.223 px mean reprojection error
SIMPLE_RADIAL camera model
```

These global metrics are no longer sufficient evidence of pose correctness.

### Proven `geo_g10` sparse inconsistencies

```text
145220 -> 145223
  verified two-view inliers: 449
  two-view rotation: ~15.9 deg
  final sparse rotation: ~103.8 deg
  final shared 3D tracks: 0

145254 -> 145256
  two-view rotation: ~14.7 deg
  final sparse rotation: ~32.2 deg
  final shared tracks: 10

145330 -> 145333
  two-view rotation: ~11.8 deg
  final sparse rotation: ~61.5 deg
  final shared tracks: 7
```

Historical `geo_g10` also contains an acquisition-adjacent virtual-camera ray jump around 82.6 degrees. The sparse report records repeated CHOLMOD matrix-not-positive-definite warnings during final BA.

**Repair consequence:** sparse integrity must be diagnosed and repaired/safely replaced before final dense regeneration.

### Historical dense diagnosis

Corrected masks remain healthy and broad background contamination is low, but geometric depth support collapses in the high rings:

```text
g7_r1   median ~0.947 / p10 ~0.922
geo_g8  median ~0.890 / p10 ~0.851
geo_g9  median ~0.867 / p10 ~0.825
geo_g10 median ~0.659 / p10 ~0.142
geo_g11 median ~0.590 / p10 ~0.299
geo_g12 median ~0.337 / p10 ~0.034
photometric medians ~1.0
```

The historical true3 combined PatchMatch configs also contain:

```text
567 reference entries
372 unique reference images
158 duplicated reference images
max duplicate reference count: 4
```

This means later tiles could rewrite maps for the same reference. The repaired dense generator must enforce one reference assignment per registered image per complete candidate.

### Historical raw mesh diagnosis

Historical raw Poisson:

```text
887,770 vertices
1,669,931 faces
644 connected components
largest component face fraction ~0.6657
```

That mesh is objectively fragmented and must fail the repaired continuity gate even though the old raw visual gate passed.

### Source/mask/appearance facts that remain valid

```text
CSX4213_Project_V4_Images/ = immutable 688 JPEG source
158 uncoated appearance/reference
107 empty-board/background
423 coated/marked geometry
corrected COLMAP fusion masks = 372/372 under <image_name>.png
historical corrected-mask board fraction ~0.00778
historical cloth/background fraction ~0.000492
historical pedestal-board webbing = 0
```

Representative MVS masks preserve the vessel silhouette. Broad remasking is not the first repair action.

## Canonical repair documentation

Current execution is defined by:

```text
docs/superpowers/specs/2026-09-14-v4-full-repair-design.md
docs/superpowers/plans/2026-09-14-v4-full-repair.md
```

`AGENTS.md`, `CLAUDE.md`, `LESSONS.md`, and `docs/memory-bank/active-context.md` have been updated to point to the full-repair path.

The old dense-only repair documents now contain supersession pointers. The 2026-09-10 fast-end-to-end plan/spec are historical requirements/context, not the active continuation plan.

## Implementation progress against the new plan

```text
Task 0  live state/protection                  NOT YET EXECUTED BY REPAIR IMPLEMENTATION
Task 1  sparse-integrity diagnostics          NOT STARTED
Task 2  corrected sparse constraint graph     NOT STARTED
Task 3  sparse repair/safe replacement        NOT STARTED
Task 4  fresh compatible dense workspace      NOT STARTED
Task 5  default repaired 2000px PatchMatch    NOT STARTED
Task 6  bounded dense escalation              NOT STARTED
Task 7  repaired dense acceptance             NOT STARTED
Task 8  repaired Poisson acceptance           NOT STARTED
Task 9  scan-preserving Blender/LOD0           NOT STARTED
Task 10 honest appearance workflow            NOT STARTED
Task 11 final GLB fresh re-import              NOT STARTED
Task 12 full verification/docs/publication    NOT STARTED
```

Planning/investigation work is complete enough to begin Task 0/Task 1 without more architecture discovery unless live repository state has changed.

## Required repair order

```text
recover/protect newest state
-> sparse-integrity regression + diagnostics
-> corrected verified sparse graph
-> local repair if robust, otherwise fresh versioned sparse remap
-> sparse gate + milestone commit/push
-> fresh compatible image/mask undistortion and dense workspace
-> one-reference-one-write source/config generation
-> CUDA PatchMatch/fusion bounded ladder
-> strict dense anatomical gate + milestone commit/push
-> strict Poisson continuity/anatomy gate + milestone commit/push
-> scan-preserving Blender clean-high/LOD0/UV/detail/AO
-> verified photographic appearance projection OR documented reference-derived brass PBR fallback
-> versioned final .blend/.glb
-> factory-empty GLB re-import + eight-view anatomy check
-> tests/hashes/docs/final focused commit/push + origin/main verification
```

## Critical acceptance rules

- Do not accept sparse merely because 372/372 register.
- Do not use `phase_01` as measured physical azimuth.
- Do not reuse historical true3 maps as final evidence after accepted sparse camera geometry changes unless compatibility is explicitly proven.
- Do not permit duplicate PatchMatch reference writes across tiles.
- Do not accept dense solely by point count or finite/rank-3 status.
- Do not accept Poisson unless dominant face component >=0.95 and second-largest <=0.02, with no major anatomical hole.
- Do not repair missing vessel-scale anatomy by hand/symmetry/lathe/global reconstructive remesh in Blender.
- Do not claim photo BaseColor projection unless auxiliary appearance cameras pass measured alignment.
- Do not call the project complete until fresh GLB re-import retains the repaired anatomy.

## Protected/unrelated work

Preserve:

- `CSX4213_Project_V4_Images/` as the immutable 688-photo raw source set and intentional Git LFS content
- `capture_v4/derived/mvs_images/`, `capture_v4/derived/masks/`, and `capture_v4/derived/feature_masks/` as the current processed-image sets and intentional Git LFS content
- `.codegraph/` as local code-intelligence state
- unrelated `.ai-bridge/` and other unrelated user-owned work
- current V4 dense/mesh/Blender/GLB artifacts as rollback/evidence

Use exact-path staging and cleanup. Large dense/sparse transient candidate workspaces belong outside the repo under versioned scratch roots.

## Git policy

The user has authorized focused commit/push at these verified milestones:

1. sparse repair accepted;
2. dense repair accepted;
3. raw Poisson accepted;
4. final Blender + GLB fresh-reimport accepted.

Do not commit merely because planning docs changed. Before every milestone commit, inspect root/branch/upstream/status/intended diff, stage only intended files, push normally, and verify the remote commit. Never force-push or rewrite history.

## Completion condition

The new repair is not complete until the proven sparse defect is gone, dense evidence is compatible with the repaired cameras and passes strict anatomical/coverage/contamination checks, Poisson is genuinely connected and complete, Blender remains scan-derived, appearance provenance is honest, the final GLB survives fresh re-import, relevant tests/hash integrity pass, docs reflect measured final state, and focused Git publication is verified.

## Latest verified repair checkpoint (2026-09-14)

The current MAX_VISIBLE_POINTS_RATIO incremental candidate failed the unchanged v3 independent audit on ten full SO(3) rotation checks. Its output and gate remain preserved as negative evidence; no further incremental mapper/order variants were run.

The canonical v3 SQLite snapshot is now immutable-by-policy and manifest-bound. Repair/audit commands consume `canonical_sqlite_snapshot_v3.json`, copy disposable working bytes, and never open the canonical path with SQLite or pyCOLMAP. The final raw hash is exactly `658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a`, equal to the creation manifest, with no journal sidecars. The creation logical digest remains `ec4d9a4c9a548903d71704401e50e666408a933dfdb19cc02cbe8e93fd860f97`.

The fixed independent audit was rerun as `independent_audit_geometry_v6_deterministic.json` for all 1,868 fixed pairs using calibrated two-view re-estimation and seed 4201. Classification retained 1,527 well-conditioned calibrated rotations and preserved 341 contradictory/unconditioned records, including 13 measured local-cycle inconsistencies. The previously recorded single degenerate `geo_g10` edge remains the only mapping exclusion.

One and only one versioned pyCOLMAP 4.2.0 GLOMAP/global-mapping candidate was run from the corrected disposable graph. It produced one model with all 372 intended views, 34,092 sparse points, 304,023 observations, and mean reprojection error 1.180133 px. The final calibrated sparse gate failed on 18 well-conditioned same-ring fixed-audit pairs with SO(3) disagreement and 24 with shared final-track collapse. The pre-calibrated diagnostic and final calibrated results are preserved in `repair_sparse_gate_v6_glomap.json` and `repair_sparse_gate_v6_glomap_calibrated.json`; neither is promoted.

Current task status after this checkpoint:

```text
incremental MAX_VISIBLE_POINTS_RATIO candidate   COMPLETE / NEGATIVE EVIDENCE
manifest-bound v3 audit/remap/evaluator          COMPLETE / TESTED
calibrated fixed-audit classification            COMPLETE / 1527 retained, 341 preserved failures
pyCOLMAP 4.2 GLOMAP candidate                    COMPLETE / STRICT SPARSE GATE FAILED
sparse milestone commit/push                     BLOCKED BY FAILED GATE
dense/PatchMatch/fusion                          NOT STARTED (correctly blocked)
Poisson/Blender/GLB                              NOT STARTED (correctly blocked)
```

Fresh focused regression checks after these changes: `26 passed` across `tests/test_v4_repair.py`, `tests/test_v4_postfusion.py`, and `tests/test_v4_dense.py`; all touched Python modules compile successfully. The GLOMAP runtime reported no CUDA support and fell back to CPU solvers; that runtime limitation is recorded in `reconstruction/v4/repair/sparse_v1/glomap_v6_run_report.json`.

## Historical sparse architecture-switch checkpoint (superseded 2026-09-14)

The one permitted pyCOLMAP 4.2.0 GLOMAP candidate failed the unchanged calibrated fixed audit, so incremental mapper/order variants were stopped and preserved as negative evidence. The replacement computer-vision-only architecture used calibrated ALIKED/LightGlue evidence: robust same-ring SO(3) synchronization, deterministic robust cross-ring rotation consensus, and disjoint pair-local verified tracks rebuilt against the synchronized camera field. That candidate is now superseded because pair-local tracks cannot prove global continuity; it uses no phase/turntable pose prior, external/reference geometry, manual geometry, or appearance data.

Historical sparse artifacts (preserved, not current acceptance):

```text
model: D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1\sparse_candidate_v10_rotation_consensus_pairwise_tracks\0
model SHA-256: 78636a8279cb0114fad17a7e2d7f100b53188cbd21e7c96c224844d333da8c1d
gate: reconstruction/v4/repair/sparse_v1/repair_sparse_gate_v10_rotation_consensus_pairwise_tracks.json
gate SHA-256: dabafe769a0a696f53014fa735f49cf68d48024aef2f32d35c42fe3ce563c5e7
accepted lineage: reconstruction/v4/repair/sparse_v1/accepted_sparse_v4_rotation_consensus_v1.json
```

The gate passes every sparse check: exactly one model with all 372 intended views, 54,115 sparse points, 108,230 observations, mean reprojection error ~0.9514 px, camera/intrinsics integrity, trajectory continuity, full fixed-audit binding, same-ring SO(3) consistency, shared-track continuity, and actual ring-graph connectivity. The fixed 1,868-pair audit, its 341 contradictory/unconditioned records, and the single independently-degenerate `geo_g10` exclusion remain preserved and hash-bound. Pair-local tracks have mean length 2.0; this is recorded rather than hidden.

The canonical snapshot final raw hash is again exactly `658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a`, equal to the v3 creation manifest; no canonical journal sidecars exist. Direct SQLite/pyCOLMAP opens of the canonical path remain forbidden, and the accepted sparse gate records the disposable working-copy lineage.

Current measured task status:

```text
incremental MAX_VISIBLE_POINTS_RATIO candidate   COMPLETE / NEGATIVE EVIDENCE
v3 immutable SQLite snapshot + disposable lineage COMPLETE / HASH VERIFIED
calibrated fixed independent audit                COMPLETE / 1,868 pairs
pyCOLMAP 4.2 GLOMAP candidate                    COMPLETE / STRICT GATE FAILED, PRESERVED
rotation-consensus sparse replacement            HISTORICAL / superseded by independent global-continuity gate
sparse milestone publication                     HISTORICAL / prior publication retained as evidence; no new promotion
production dense provenance gate                 HISTORICAL / exact-one-write + zero-source-only + sparse-hash lineage hardened
fresh compatible dense workspace                 PRESERVED / no rerun until sparse acceptance reopens
PatchMatch/fusion/depth/coverage/contamination   PRESERVED DIAGNOSTICS / not promoted
Poisson/Blender/GLB                              BLOCKED / sparse and raw anatomy gates not accepted
```

## Dense provenance hardening checkpoint (2026-09-14)

Before any repaired PatchMatch run, `audit_final_tile_configs` now fails closed unless the actual geometric tile set contains every registered reference exactly once, contains no source-only reference writes, and carries the accepted repaired sparse model/gate hashes. `postfusion_evidence_gate` independently verifies the same sparse lineage and can compare it to the expected accepted model-directory hash. The historical duplicate-write regression fixture (567 configured references, 372 unique views, 158 duplicates) remains rejected, and a valid exact-one-write fixture is covered. No new repaired PatchMatch computation has started at this checkpoint.

## Fresh repaired dense and raw-Poisson checkpoint (2026-09-14)

The previously accepted rotation-consensus sparse hash was used to create a fresh external dense workspace at `D:\Side Projects\CSX4213_V4_Dense_Work\repair_rotation_consensus_v1_fullrun`. A 15-view CUDA smoke pass and one full COLMAP 4.2.0 GPU PatchMatch configuration completed, but that entire dense result is now preserved as diagnostic evidence only because its sparse lineage was superseded. The historical tiled orchestration was attempted once and preserved as negative evidence: it failed closed because source-only images outside the first tile had no photometric maps. No dense rerun is permitted until a new sparse candidate passes the strengthened global-continuity gate.

The full fresh fusion completed with 1,306,289 points and passed the post-fusion evidence gate. The report is `reconstruction/v4/reports/dense_repair_rotation_consensus_v1_fullrun_postfusion.json`, fused-cloud SHA-256 `4912b0d05f6e0b5358be8e3103ec6a9492acdf70a6a68a58316960766f43781d`. The gate verifies the accepted sparse model hash `78636a8279cb0114fad17a7e2d7f100b53188cbd21e7c96c224844d333da8c1d`, exact `colmap_image_name_plus_png` masks for all 372 views, multi-view contamination provenance, measured ring transitions, four semantic previews, and all eight distinct anatomy previews (`bowl_interior`, `rim`, `globe_shoulder`, `continuous_neck`, `lid_tiers`, `finial`, `pedestal_transitions`, and `base`). Direct inspection of those eight technical projections confirmed the measured bowl/interior, rim, globe/shoulder, neck, lid/finial, pedestal, and base evidence; this is not a replacement for the raw-mesh gate.

One bounded fusion escalation from the same geometric maps (`StereoFusion.min_num_pixels=3`) also passed the unchanged dense evidence gate; its report is `reconstruction/v4/reports/dense_repair_rotation_consensus_v1_fullrun_min3_postfusion.json`, fused-cloud SHA-256 `5023dd1589f07b3185e59c8567958aa76e0f48749ebfc53c5fdef79e260095f3`. Its measured contamination remained below the fixed quantitative limits (board `0.0069167`, cloth/background `0.0298167`, pedestal webbing `0.0`) and its sparse lineage is identical. It is retained as a bounded diagnostic candidate, not silently substituted for the default fusion.

Raw Poisson remains blocked. On the default fused cloud, depth-13 Poisson with trim 10 produced 110,552 vertices / 200,750 faces, 578 components, dominant face fraction `0.5302665`, and second-largest fraction `0.4201395`; it fails the hard `>=0.95` / `<=0.02` gate. A single evidence-driven trim reduction to 5 produced 3,884,610 vertices / 7,765,374 faces, 6,346 components, dominant `0.9473674`, and second-largest `0.0019468`; trim 4 was unchanged at dominant `0.9473692` and also fails closed. The min-3 fusion escalation produced a depth-13 trim-10 mesh with 4,129,158 vertices / 7,737,390 faces, 7,565 components, dominant `0.8824999`, and second-largest `0.0434767`; it is also rejected. All three meshes and logs are preserved as negative evidence. No Blender, manual bridge, reconstructive cleanup, or appearance work has started because the raw Poisson continuity gate has not passed.

Current measured task status:

```text
incremental MAX_VISIBLE_POINTS_RATIO candidate   COMPLETE / NEGATIVE EVIDENCE
v3 immutable SQLite snapshot + disposable lineage COMPLETE / HASH VERIFIED
calibrated fixed independent audit                COMPLETE / 1,868 pairs
pyCOLMAP 4.2 GLOMAP candidate                    COMPLETE / STRICT GATE FAILED, PRESERVED
rotation-consensus sparse replacement            HISTORICAL / superseded by independent global-continuity gate
sparse milestone publication                     HELD / no new sparse promotion
production dense provenance gate                 PRESERVED / exact-one-write + zero-source-only + sparse-hash lineage
fresh compatible CUDA dense reconstruction       PRESERVED DIAGNOSTIC / no rerun until sparse acceptance
depth/coverage/contamination/anatomy evidence    PRESERVED DIAGNOSTIC / anatomy gate remains strict and region-specific
raw Poisson continuity/anatomy gate              BLOCKED / sparse lineage reopened; prior meshes remain negative evidence
Blender/LOD0/UV/material/GLB                     BLOCKED / no sparse or raw anatomy promotion
```

## Sparse acceptance reopened after independent global-continuity audit (2026-09-14)

The earlier v10 sparse acceptance is superseded and remains diagnostic only. Its
54,115 points all have track length exactly two because the candidate used the
deliberate pair-local allocator; those constructed tracks are not evidence of a
globally coherent reconstruction. The strengthened gate now rejects that
provenance explicitly and projects the complete sparse cloud into every
registered view and ring against the fixed masks.

The independent measurements are preserved in the versioned sparse-v1 reports:

```text
v10 pair-local candidate: mean track length 2.0, fraction >=3 views 0.0;
  per-view precision p10 0.7219, recall p10 0.7196, IoU p10 0.5719;
  gate FAILED (constructed pair-local tracks and projection inconsistency)
v6 GLOMAP candidate: mean track length 8.9177, fraction >=3 views 0.9970,
  but duplicate-image track elements, 18 calibrated pose failures, and 24
  shared-track failures; mask precision p10 0.9347 and IoU p10 0.6289;
  gate FAILED (duplicate tracks and pose/shared-track health)
v8 database-track consensus: genuine multi-view distribution, but mask
  precision p10 0.6128, recall p10 0.5283, IoU p10 0.4700;
  gate FAILED (per-view/per-ring projection consistency)
v11 ring-aligned calibrated repair: same-ring pose check passes, but only
  49.067% tracks have >=3 views, five rings have low reference-gauge support,
  and translation/mask gates fail; model SHA
  5b587a3216e8d7ea8ed51c3e09a85391a5fe36a1dc2c184addb51103dc4c4850
v12 ring-aligned calibrated-center repair: 4,005 points, mean track length
  2.3243, mean reprojection 3.8662 px; translation, projection, track, and
  reprojection gates fail; model SHA
  4f9f791d3cb7d5c2d68cba9f7a428693183ffe847b7ac4859b63f625d7d847b0
```

The v11/v12 repair reports use the fixed calibrated ALIKED/LightGlue graph,
the immutable v3 SQLite manifest, and disposable working database copies. No
pair-local tracks are allocated. The calibrated center solver uses the audited
first-to-second translation vectors when present and records an explicit
database-essential fallback only for absent vectors. Neither candidate is
accepted or hash-bound as sparse lineage. No sparse milestone, dense rerun,
Poisson promotion, Blender work, or GLB work is authorized until a replacement
candidate passes genuine multi-view tracks, per-view/per-ring mask precision /
recall / IoU, calibrated center/translation-scale consistency, and the existing
trajectory, reprojection, camera, model, and immutable-lineage gates together.

The anatomy gate remains strict and region-specific. Whole-object projections
named for anatomy cannot satisfy it; the min-pixel-2 diagnostic is preserved as
negative evidence for resolved finial support in the fused cloud. Existing v10,
dense, and PW0 artifacts are retained for diagnosis and are not promoted.

## Unit-correct calibrated remap and final permitted GLOMAP candidate (2026-09-14)

The fixed independent audit was regenerated from the immutable v3 SQLite
snapshot using a disposable working copy. The v8 report is
`reconstruction/v4/repair/sparse_v1/independent_audit_geometry_v8_calibrated_units.json`:
all 1,868 pair IDs are unchanged, the raw audit SHA is
`39c46fb6e164655b590a776db4d1ec422b4edfb178e3ef492f908b7d5c9ad5a0`, and its
canonical raw SHA remains
`658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a` with no
canonical `-wal`/`-shm` sidecars. `TwoViewGeometry.tri_angle` and
`Rotation3d.angle()` are recorded as radians and converted explicitly to degree
fields; the old v7/v2 mislabeled reports remain preserved negative evidence.

The v3 calibrated classification is
`reconstruction/v4/repair/sparse_v1/calibrated_pair_classification_v3_calibrated_units.json`.
It is bound to the v8 raw report, the fixed-audit identity, and the v3
snapshot; 1,530 records are well-conditioned and each retained mapping record
contains the same calibrated R and t. The direct g7/g10/g11 rotation/position
diagnostic remains diagnostic-only: source pose-island corrections are about
97.58 degrees (g7), 51.23 degrees (geo_g10), and 49.71 degrees (geo_g11),
while the calibrated graph is connected but still contains geo_g10
same-ring synchronization outliers and 66 negative solved translation
directions.

The required full-1530 disposable remap
`strict_global_v3_full1530_calibrated_units_remap_report.json` passed every
provenance precondition: `pair_count=1530`, `rewritten_pair_count=1530`,
`failed_pair_count=0`, `pose_override_applied_count=1530`,
`pose_override_round_trip_pass_count=1530`, exact classification/raw-audit/
snapshot hashes, and unchanged frozen canonical bytes. The remapper now fails
closed when a requested disposition pair lacks its audited R+t; runtime pose
fallback is not allowed for this candidate.

The one fresh versioned pyCOLMAP 4.2 GLOMAP run from that remap is preserved as
negative evidence at
`D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1\sparse_candidate_v31_full1530_calibrated_global\0`.
It returned exactly one model with all 372 intended views, but the unchanged
independent sparse gate
`repair_sparse_gate_v31_full1530_calibrated_global.json` fails three required
checks: genuine conflict-free multiview track integrity, same-ring SO(3)
consistency, and shared-track continuity. The failed rotation evidence is 15
edges (six g7, three geo_g10, six geo_g11); the candidate has 5,952 duplicate
image track elements despite a mean track length of 9.29285. Masks,
camera-center/translation, trajectory, reprojection/intrinsics, ring
connectivity, one-model/372-view, angle-unit, and immutable-lineage checks
pass. No sparse promotion, milestone commit/push, dense/PatchMatch, Poisson,
Blender, or GLB work follows this failed candidate without new causal
evidence.

## Exact mapper-graph precondition and corrected GLOMAP (2026-09-14)

The v31 result above is now explicitly classified as an invalid-input
diagnostic rather than the final calibrated-graph verdict. Its remapped
disposable database still contained 2,118 nonempty `two_view_geometries`
rows, not the requested 1,530: 588 extra verified pairs carried 91,345
additional inlier rows (338 pairs were present in the v3 classification but
rejected, and 250 were outside the fixed audit). All 588 extras also had raw
`matches` rows, contributing 134,567 raw rows. The remapper previously
rewrote retained rows without physically deleting these graph inputs.

The exact-graph repair is now versioned and fail-closed. `validate_exact_mapper_graph`
compares the nonempty verified pair-ID set and raw-match pair-ID set against
the requested retained set before mapping. `prepare_v4_exact_mapper_database.py`
creates a fresh disposable copy from the v3 manifest and physically prunes
both tables; its preparation report records exactly 1,530 verified pairs,
1,146,567 verified inlier rows, exactly 1,530 raw-match pairs, no extras, and
unchanged canonical bytes. The remapper and GLOMAP runner repeat this
preflight. Read-only preflights use immutable SQLite URIs so disposable
inspection cannot create journal sidecars that are mistaken for provenance.

The installed pyCOLMAP 4.2 run log records `DatabaseCache` loading 1,530
verified graph pairs and building its correspondence graph from those
geometries. Raw `matches` are therefore pruned defensively, while
`two_view_geometries.inlier_matches` is the reconstruction track source.

The corrected exact-graph candidate is preserved at
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v33_exact1530_global\\0`.
Its model hash is
`2fc3891880e9122f6fa48141f806e48d80079784e3d993fb4f9e46bcaeb22115`; it has
one model with all 372 views, 34,014 points, 304,072 observations, mean track
length 8.9396131, and mean reprojection error 1.17576005 px. The exact
remap report records 1,530/1,530 rewrites, zero failed pairs, 1,530 audited
R/t overrides, and 1,530 passing R/t/E/F round-trips with the v8 audit,
v3 classification, fixed-audit identity, and v3 snapshot hashes.

The unchanged full independent gate remains negative in
`repair_sparse_gate_v33_exact1530_global_final.json`: mask projection,
camera-center/translation, trajectory, reprojection/intrinsics, ring
connectivity, one-model/all-372, model hash, and immutable lineage pass, but
`independent_multiview_track_distribution`, `same_ring_pose_consistency`, and
`shared_track_continuity` fail. Genuine tracks have p50/p90/p99 lengths
6/18/48 and 99.7942% length >=3, but 6,167 tracks contain duplicate image
observations; there are 18 calibrated same-ring SO(3) failures and 25
shared-track failures. This is the one corrected GLOMAP candidate, preserved
as negative evidence. Dense/PatchMatch, sparse promotion, Poisson, Blender,
GLB, milestone commits, and pushes remain blocked; no further sparse variant
is authorized without new causal evidence.

## Conflict-free graph diagnosis and v35 negative evidence (2026-09-14)

The next causal repair preserved the full calibrated classification and fixed
audit, but used a deterministic mapper graph selector that rejects any
correspondence cycle whose observation component contains two keypoints from
one image. The first selected graph (`global_conflict_free_graph_v1.json`)
contained 384 edges and no duplicate-image observation components. Its GLOMAP
candidate v34 was not accepted: the independent gate still found six geo_g8
same-ring SO(3)/shared-track failures and also failed the per-view/per-ring
mask-projection precision checks. Genuine multiview tracks, camera centers,
trajectory, reprojection, intrinsics, ring connectivity, and immutable lineage
passed. This is preserved as causal negative evidence, not as a sparse
promotion.

The v34 diagnosis identified two selected g7-to-geo_g8 bridges as robust
consensus outliers. The versioned graph
`global_conflict_free_graph_v3_g7_g8_consensus_outliers_excluded.json` removes
all 50 outlier pair IDs for that transition while keeping them in the fixed
independent audit. It is physically exact for its 383 retained mapper pairs:
the disposable source/remap reports show no raw-match or
`two_view_geometries` extras, 383 calibrated pose overrides, 383 passing R/t/
E/F round-trips, and unchanged canonical bytes.

The resulting pyCOLMAP 4.2 GLOMAP candidate v35 is preserved at
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v35_g7g8_consensus_outliers_excluded\\0\\0`.
Its stable model SHA is
`168c01ec9ffffb4a641fecda375a71a9addd97154d05302f09bfa8da40ec4394`.
It contains one model with all 372 intended views, 60,325 points, 384,756
observations, mean track length 6.3780522, and mean reprojection error
1.0398253 px. The mapper established genuine tracks (not pair-local
constructed tracks) and the exact graph preflight passed.

The unchanged full sparse gate
`repair_sparse_gate_v35_g7g8_consensus_outliers_excluded.json` remains
negative. Mask projection now passes for every view and ring, as do genuine
multiview track distribution, camera-center/translation scale, trajectory,
reprojection/intrinsics, ring connectivity, one-model/all-372, stable model
hash, and immutable SQLite lineage. Only same-ring SO(3) and shared-track
continuity fail: six geo_g11 audit pairs have 155.7--157.5 degree final-vs-
calibrated rotation disagreement and 0--2 shared final tracks. The selected
geo_g10-to-geo_g11 graph contains four robust consensus-outlier bridges; the
same-ring calibrated edge `IMG20260912150307.jpg ->
IMG20260912150309.jpg` is independently well-conditioned, but GLOMAP's
rotation-averaging log invalidated it and the final model flips it by about
155.7 degrees. Removing all geo_g10-to-geo_g11 outliers from the source graph
disconnects the image graph; iterative removal replaces them with further
outlier bridges. This is new causal evidence of a remaining calibrated
g10/g11 pose-island conflict, not permission to relax the gate or churn
mapper parameters.

For auditability, `global_conflict_free_graph_v4_selected_g10g11_outliers_excluded.json`
was generated only as a source-graph diagnostic after v35. Its selector
replaced the four removed geo_g10-to-geo_g11 bridges with three further
consensus-outlier bridges, so it was not remapped, mapped, or treated as a
candidate. The file and report remain preserved negative evidence.

The strict global-mapping run report now records the separate track-provenance
path before its hash is bound; the v35 report/provenance were refreshed with
`--reuse-existing` only, without changing the model. Sparse promotion,
dense/PatchMatch, Poisson, Blender, GLB, milestone commits, and pushes remain
blocked until a new independently justified sparse architecture resolves this
g10/g11 conflict. v34 and v35 remain diagnostic negative evidence.

## Fresh exact-graph v36 recheck (2026-09-14)

To ensure the exact-graph correction was not inferred from an older disposable
copy, a new source database and a new calibrated remap were created from the
frozen v3 manifest. The preparation report is
`reconstruction/v4/repair/sparse_v1/exact_mapper_graph_v36_preparation_report.json`;
the remap report is
`reconstruction/v4/repair/sparse_v1/strict_global_v36_exact1530_calibrated_units_remap_report.json`.
The source and remapped databases each contain exactly 1,530 nonempty
`two_view_geometries` pair IDs and exactly 1,530 nonempty raw-match pair IDs;
the verified inlier-row totals are 1,146,567 before remap and 1,146,417 after
calibrated R/t/E/F rewriting. The expected/actual pair-ID hash is
`c01f5037307ff9589efb78baef5ed6dd43c95bb2bbd719af8687e69691c2adf2`, with no
extras or missing IDs. All 1,530 pose overrides and all 1,530 R/t/E/F round trips
passed, `failed_pair_count=0`, and the classification, raw-audit, fixed-audit,
manifest, canonical raw SHA, and creation-time logical SHA are unchanged.

An installed pyCOLMAP 4.2 `DatabaseCache` probe on a disposable copy confirmed
that deleting one raw `matches` row leaves the corresponding graph pair in the
loaded `CorrespondenceGraph`; the mapper track source is
`two_view_geometries.inlier_matches`. Raw `matches` are nevertheless pruned in
the exact disposable graph so no non-retained pair can re-enter through a
future runtime path.

The one fresh corrected GLOMAP run from that exact remap is preserved at
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v36_exact1530_global\\0`.
It contains one model and all 372 intended views, with model SHA
`fef8ff6309027f0f81847ec6cc83d207739e870971daddf3aa8d817213d4883d`, 34,014
points, 304,072 observations, mean track length 8.9396131, and mean
reprojection error 1.17576005 px. The unchanged independent gate in
`repair_sparse_gate_v36_exact1530_global.json` remains negative: full
per-view/per-ring masks, camera-center/translation scale, trajectory,
reprojection/intrinsics, ring connectivity, one-model/all-372, stable model
hash, calibrated angle units, and immutable lineage pass, but genuine track
integrity fails because 6,167 tracks contain duplicate image observations;
same-ring SO(3) fails for 18 calibrated audit pairs and shared-track continuity
fails for 25. The genuine track distribution itself is healthy (99.7942% of
tracks have length >=3; p50/p90/p99 are 6/18/48), so it does not excuse the
duplicate-observation and pose failures. v36 is valid-input negative evidence,
not a sparse promotion or a basis for dense reconstruction. No dense,
Poisson, Blender, GLB, milestone commit, or push is authorized without new
causal evidence and a candidate that passes every sparse gate.

## Consensus-inlier-first sparse architecture diagnostic (2026-09-14)

Because the exact v36 graph isolated a remaining pose-island conflict rather
than an input contamination problem, one additional causal architecture was
tested. `build_v4_conflict_free_graph.py` now has an explicit
`--consensus-inlier-first` policy that requires the independent v30 rotation
consensus report and ranks `retained_robust_consensus_inlier` edges before
same-ring locality. The policy is regression-tested as an edge-ordering
decision; the fixed audit and all rejected pairs remain immutable evidence.
The resulting graph
`global_conflict_free_graph_v6_consensus_inlier_first.json` is connected,
contains 384 edges, has zero duplicate-image observation components, and uses
218 consensus-inlier plus six consensus-outlier cross-ring bridges.

The exact disposable source/remap reports for this 384-edge graph record
384/384 calibrated pose overrides, 384/384 R/t/E/F round trips, zero failed
pairs, exact raw/verified pair-ID sets, and unchanged canonical bytes. The
versioned pyCOLMAP 4.2 GLOMAP result is
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v37_consensus_inlier_first\\0`,
with model SHA
`21d53297e4287f6f3dc165de47a4ddb518af0ab561ab3dc973206f24881dfe6a`, one
model/all 372 views, 41,552 points, 262,218 observations, mean track length
6.31059877, and mean reprojection error 1.06392411 px. The independent gate
`repair_sparse_gate_v37_consensus_inlier_first.json` remains negative: genuine
track distribution and duplicate-image integrity improved to only two
duplicate-image tracks and 99.8724% tracks of length >=3, but six calibrated
same-ring SO(3) edges, 41 shared-track edges, and both per-view and per-ring
mask projection gates fail. v37 is diagnostic negative evidence only. Sparse
promotion, dense/PatchMatch, Poisson, Blender, GLB, milestone commits, and
pushes remain blocked; no further candidate is justified without new causal
evidence.

## Match-conflict filtering and direct calibrated-pose v43 (2026-09-14)

The v31 result remains invalid-input evidence because its remapped disposable
database still contained 2,118 nonempty `two_view_geometries` instead of the
classified 1,530-pair graph. The exact-graph preflight and the
`build_v4_match_conflict_free_database.py` path now compare both physical
pair-ID sets and prune non-retained pairs from `matches` and
`two_view_geometries`. The installed pyCOLMAP 4.2 probe confirms that track
loading uses `two_view_geometries.inlier_matches`; raw matches remain pruned as
defensive lineage. The new regression coverage keeps the exact set invariant
and tests deterministic match-level conflict filtering without changing the
fixed independent audit set.

v38b used that conflict filter on the exact 1,530-pair graph and established
healthy genuine multiview tracks (0 duplicate-image tracks; 99.5149% of tracks
with length at least three), but failed 15 calibrated SO(3) edges and 19
shared-track edges. Bounded global and g7-to-g8 caps (v39, v41, and v42) were
preserved as negative input evidence: asking the calibrated two-view estimator
to re-estimate deliberately down-weighted subsets failed closed for one or more
pairs. No estimator failure was hidden by a fallback pose.

The one permitted architecture correction is the direct calibrated-pose
override in `apply_v4_calibrated_pose_overrides_to_filtered_matches.py`. It
copies the exact disposable v39 conflict-filtered graph, retains its filtered
ALIKED/LightGlue inlier rows, and binds each pair's independent v3 calibrated
first-to-second R and t from the same audit record. It rebuilds E/F/q/t and
round-trips all 1,530 pairs before GLOMAP. The v43 remap report records
`failed_pair_count=0`, `pose_override_applied_count=1530`,
`pose_override_round_trip_pass_count=1530`, the exact pair-ID/classification/
raw-audit/snapshot manifest hashes, and unchanged canonical bytes.

The versioned pyCOLMAP 4.2 GLOMAP v43 candidate is preserved at
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v43_match_conflict_free_outlier_cap15_direct_calibrated_global\\0`.
It contains one model with all 372 intended views, model SHA
`7160a30d86f3b22a4a751cc34b4b55cc506ecb64570595b0f008a599cf34833b`, 53,204
points, 387,218 observations, mean track length 7.2779866, and mean
reprojection error 1.1178619 px. Genuine track integrity, SO(3), shared-track,
trajectory, camera/intrinsics, camera-center/translation, ring connectivity,
one-model/all-372, stable model hash, and immutable SQLite lineage all pass.
The unchanged mask gate is negative: view IoU p10 is 0.5849977 and precision
p10 is 0.7598000 versus the 0.6/0.9 thresholds; ring precision means range
from 0.7968294 to 0.8719864, and `geo_g11` ring IoU mean is 0.5745879. The
candidate therefore remains diagnostic negative evidence. Sparse promotion,
dense/PatchMatch, Poisson, Blender, GLB, milestone commits, and pushes remain
blocked; no dense work was started from v43.

## Bounded mask/pose tradeoff diagnostics v44-v45 (2026-09-14)

Two bounded follow-up tests were run from exact 1,530-pair disposable graphs
without changing the frozen source database, fixed audit, calibrated
classification, or sparse thresholds. v44 used the cap-35 conflict-filtered
graph with the same direct calibrated first-to-second R/t override and exact
R/t/E/F round trips. Its GLOMAP output is
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v44_match_conflict_free_outlier_cap35_direct_calibrated_global\\0`;
the model SHA is
`fa4bd9c123c415d13c2f05a303d3a15bd667529fd1921bece98f9ecfa30f2a11`, with
53,135 points, 384,262 observations, mean track length 7.23181, and mean
reprojection error 1.12013 px. v44 passes the exact model, genuine-track,
mask, camera/intrinsics, trajectory, ring-connectivity, and immutable-lineage
checks, but fails the six calibrated g7 same-ring SO(3) edges and five shared
track-continuity edges. The failures are coherent 56--58 degree final-pose
flips against 2--14 degree calibrated rotations, not an auditable threshold
margin.

v45 tested the one targeted g7-to-g8 cap-15 source graph for which the older
stored-E re-estimator had failed, using the direct calibrated override instead
of a runtime-pose fallback. Its exact remap report records 1,530 applied
overrides and 1,530 R/t/E/F round trips with zero failures. The GLOMAP output
is
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v45_match_conflict_free_outlier_cap35_g7g8_cap15_direct_calibrated_global\\0`;
the model SHA is
`0b30b1e917f09386ca9d35f1319d6ea738cc8e1ec6af7ed3be6f3f07bb67ab3b`, with
53,536 points, 384,498 observations, mean track length 7.18205, and mean
reprojection error 1.11538 px. It preserves the same six g7 pose failures,
adds three shared-track failures, and fails the mask gate (view precision p10
0.84821; ring precision/IoU failures), so it is negative evidence only.

The v43-only mask hypothesis was tested without promoting or rewriting v43.
Its own supporting observations are already mask-consistent for 53,158 of
53,204 points, so track-local mask filtering cannot address the failure. A
read-only all-visible-view support diagnostic found that global support
thresholds trade precision against recall/IoU: at thresholds 0.80, 0.85,
0.90, and 0.95 the retained point counts were 32,161, 28,072, 25,172, and
21,383, while view precision p10 rose from 0.89996 to 0.96972 but view IoU
p10 fell from 0.49232 to 0.41479 and recall p10 fell from 0.52840 to
0.41886. No threshold can satisfy the unchanged 0.90 precision, 0.60 recall,
and 0.60 IoU gates. The diagnostic therefore does not justify a
scan-preserving point filter or another mapper-parameter variant.

v31 remains invalid-input evidence, not the calibrated-graph verdict. v43 is
the strongest current sparse diagnostic (all independent pose/track and
lineage gates pass but geo_g11 mask support fails); v44/v45 establish the
measured mask-versus-pose tradeoff. Sparse promotion, dense/PatchMatch,
Poisson, Blender, GLB, milestone commits, and pushes remain blocked pending
new causal image-derived evidence or a separately justified architecture.

## Transition-isolated v46 and fixed-pose retriangulation v47 (2026-09-14)

The one transition-isolated graph experiment v46 is preserved as negative
evidence. It starts from the exact 1,530-pair cap-15 graph and restores only
the 48 calibrated-consensus outlier pairs on `geo_g10:geo_g11` from 15 to 35
rows, adding exactly 960 filtered inlier rows. Its direct calibrated remap
still records 1,530/1,530 pose overrides and R/t/E/F round trips with zero
failures and unchanged canonical SQLite bytes. Versioned GLOMAP produced one
model containing all 372 views, model SHA
`65d6455665833ee254bad62498b7db47c1d9d3e0a1787e6f5cdb915be96e28f6`, 53,569
points, 384,620 observations, mean track length 7.17990, and mean reprojection
error 1.11496 px. Genuine multiview tracks, cameras, trajectories, ring
connectivity, and immutable lineage passed, but the six g7 SO(3) edges and
seven shared-track edges failed; view precision p10 was 0.85205 and view IoU
p10 was 0.65329. v46 therefore does not justify another cap-hybrid mapper
variant or sparse promotion.

The bounded architecture diagnostic v47 then reused the v43 accepted camera
pose field and invoked native pyCOLMAP 4.2 `triangulate_points` with
`clear_points=true`, `ignore_two_view_tracks=true`, `refine_intrinsics=false`,
and seed 4201 on the exact v43 disposable graph. This is not a hand-authored
consensus model: all tracks were re-established by the installed pyCOLMAP
implementation from the disposable two-view geometry. Its model is preserved
at
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v47_v43_fixed_pose_retriangulation`,
with stable model SHA
`b1c4f142172a5d9652e00e47b9c66a273d4961ef9c115caea6b8debb2d23922e`, 372
registered views, 87,964 points, 461,983 observations, mean track length
5.25196, and mean reprojection error 1.20282 px. Its model-specific track
provenance is bound in
`track_provenance_v47_v43_fixed_pose_retriangulation.json`.

The unchanged full independent sparse gate passes one-model/all-372,
genuine-track distribution (81.2196% of tracks have length at least three,
p50=4, p90=10, zero duplicate-image tracks), calibrated SO(3), shared-track,
trajectory, camera/intrinsics, camera-center/translation, ring connectivity,
stable model hash, and immutable canonical SQLite lineage. It still fails the
independent mask-projection gates: view precision p10 is 0.77318, view recall
p10 is 0.77171, view IoU p10 is 0.63827, and `geo_g11` precision mean is
0.86395 (<0.90). The bound v47 gate report is therefore negative evidence,
not an accepted sparse model. No dense/PatchMatch, Poisson, Blender, GLB,
milestone commit, or push was started from v46 or v47; further sparse
variants require new causal image-derived evidence.

## Audit-only v48 v6-center/v43-rotation composition (2026-09-14)

The bounded v6-center/v43-rotation experiment is preserved as an audit-only
diagnostic in
`reconstruction/v4/repair/sparse_v1/v48_v6_centers_v43_rotations_diagnostic.json`,
generated by
`scripts/diagnose_v4_v6_centers_v43_rotations.py`. It never writes or promotes
a COLMAP model. It copies the v43 exact-graph disposable database to a
temporary working database, opens only that copy, and hashes the frozen
canonical snapshot as raw bytes without opening it through SQLite or pyCOLMAP.

The in-memory composition used the v43 calibrated rotation field and the v6
image-derived camera-center field. Both source models contain the exact same
372 registered image names. The constructed pose has maximum rotation
round-trip error `2.9576e-6` degrees versus v43 (pycolmap representation
noise), maximum center delta `0.0` m versus v6, maximum `t=-R*C` translation
round-trip residual `6.56e-15` m, maximum rotation orthogonality residual
`3.11e-15`, and minimum determinant `0.9999999999999984`. The fixed exact
1,530-pair graph preflight passed. The unchanged local/bridge evidence
selector evaluated 1,383 of those graph records for the SO(3)/trajectory
comparison, and all 1,383 passed; intrinsics were finite with positive focal,
and the frozen canonical bytes still matched SHA-256
`658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a` with no
`-wal`/`-shm` sidecars. These facts establish only that the in-memory field is
well formed; they do not make the spliced field a promotable model.

The calibrated camera-center/translation-direction gate failed decisively:
across 1,530 evaluated edges, direction-error p90 was `79.5907` degrees
(threshold <=35 degrees), orthogonal-residual p90 was `0.982963`
(threshold <=0.6), and positive projected scale fraction was `0.947059`
(threshold >=0.95). The unchanged mask projection gate also failed
catastrophically: view precision/recall/IoU p10 were all `0.0` (means
`0.066653`/`0.060502`/`0.046341`), with ring means at or near zero for
geo_g8--geo_g12. Because the combined field is translation-incoherent and
mask-incoherent, no pyCOLMAP reconstruction/optimization was run from it.
This is negative diagnostic evidence only; sparse promotion, dense/PatchMatch,
Poisson, Blender, GLB, milestone commits, and pushes remain blocked.

## Exact-graph v49 retriangulation from v26 pose initialization (2026-09-14)

The bounded v49 diagnostic is preserved at
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v49_v26_pose_exact_graph_retriangulation`,
with report
`reconstruction/v4/repair/sparse_v1/v49_v26_pose_exact_graph_retriangulation_diagnostic.json`
and model SHA
`0ce7187706ea8d8061ecd32c5f40028cc2b1548f1181744c6184ebb38a59c79f`. The
native pyCOLMAP 4.2 `triangulate_points` path copied the already-disposable
v43 graph to a temporary working database, verified exactly 1,530 nonempty
`two_view_geometries` and raw `matches` pair IDs, and measured the actual
959,765 correspondence rows in each table. The frozen canonical snapshot was
never opened by SQLite/pyCOLMAP and remained byte-identical at SHA-256
`658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a` with no
WAL/SHM sidecars.

v49 retained the v26 image-derived pose field exactly, then rebuilt points and
tracks natively with `clear_points=true`, `ignore_two_view_tracks=true`,
`refine_intrinsics=false`, and seed 4201. It produced 372 registered views,
89,408 points, 458,301 observations, mean track length 5.12595, 80.4346% of
tracks with length at least three, zero duplicate-image tracks, and mean
reprojection error 1.19032 px. The unchanged full independent gate is bound in
`v49_sparse_gate.json` and passes camera/intrinsics, trajectory, translation,
ring connectivity, genuine-track distribution, exact one-model/all-372,
lineage, and projected-mask precision/recall/IoU (view p10
0.92504/0.81366/0.74707). It still fails the unchanged calibrated SO(3) and
shared-track gates: 15 rotation failures and 11 shared-track failures remain
on the unchanged v26 pose field. v49 is diagnostic negative evidence only and
does not justify sparse promotion or downstream work.

## Solved-center/v43-rotation native v50b diagnostic (2026-09-14)

The direct v6-center/v43-rotation splice remains rejected by v48's failed
translation and mask evidence. A separate, bounded v50b initializer instead
used v6 centers only as the initial condition of the existing calibrated
translation-direction solve, retained v43's image-derived rotation field as
the initializer, and solved all 372 centers from all 1,530 exact-graph
directions. The solve passed the unchanged translation checks: direction-error
p90 18.74756 degrees, orthogonal-residual p90 0.321399, and positive
projected-scale fraction 0.981046. Rotation orthogonality, determinant,
`t=-R*C` round-trip, exact 372-view identity, exact graph pair identity, and
frozen canonical SHA checks also passed.

The initializer was consumed only by native pyCOLMAP 4.2: points were cleared
and retriangulated from a fresh disposable copy, then one joint Ceres BA pass
ran with fixed intrinsics, a single gauge anchor, and Huber loss. The completed
diagnostic is
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v50b_solved_center_native_diagnostic\\joint_ba`,
model SHA
`0fb2deadf574c4f25f68ba68da9ac1103197d4fac0014629f28ce4fce070a14f`, with
372 registered views, 91,906 points, 202,129 observations, mean track length
2.19930, and mean reprojection error 1.17421 px. The unchanged full gate is
persisted in `v50b_sparse_gate.json`: one model/all 372, SO(3), trajectory,
intrinsics, camera-center/translation, ring connectivity, reprojection, and
lineage pass, but genuine multiview tracks fail (462 shared-track failures,
track p50=2, p90=3, only 16.1513% of tracks length at least three) and the
projected-mask gate fails severely (view p10 precision/recall/IoU
0.05850/0.17576/0.05770). v50b is therefore negative diagnostic evidence,
not an accepted sparse model. The first v50 directory is preserved as a
technical retry record whose native/BA computation completed but whose output
directory precondition was missing; no different graph or parameter candidate
was run.

The v49/v50b results establish that neither fixed-v26 retriangulation nor the
single solved-center/v43-rotation native path yields a coherent accepted sparse
model. Sparse promotion, dense/PatchMatch, Poisson, Blender, GLB, milestone
commits, and pushes remain blocked; no further sparse variant is justified
without new causal image-derived evidence.

## Corrected v51 BA harness attempt and best-defensible sparse freeze (2026-09-14)

The v50b report was not treated as BA evidence after inspection found that its
`BundleAdjustmentConfig` selected zero images: `summary.num_residuals` was
`0`, termination was `FAILURE`, and the persisted BA model hash was identical
to the native retriangulation hash. The runner was corrected to add every
registered image to `BundleAdjustmentConfig` before the gauge and intrinsic
constraints, assert `config.num_residuals(reconstruction) > 0`, and reject any
summary whose residual count is zero or whose termination is not
`SUCCESS`/`CONVERGENCE`/`USER_SUCCESS`.

The one corrected v51 experiment ran in the new directory
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\workspace_v4repair_sparse_v1\\sparse_candidate_v51_solved_center_native_joint_ba`.
It selected all 372 registered images and measured 404,258 configured
residuals, but pyCOLMAP returned `NO_CONVERGENCE`. The runner therefore failed
closed before writing a BA model or evaluating a BA sparse gate. Its native
pre-BA output remains preserved with 372 views, 91,906 points, 202,129
observations, mean track length 2.19930, 16.1513% tracks of length at least
three, and mean reprojection error 1.17421 px. The durable failure record is
`reconstruction/v4/repair/sparse_v1/v51_solved_center_native_joint_ba_failure.json`;
the canonical snapshot SHA remained
`658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a` with no
WAL/SHM sidecars.

Under the updated realistic-best-result policy, sparse candidate churn stops
here. The comparison and hash-bound freeze are recorded in
`best_defensible_sparse_v1_selection.json` and
`best_defensible_sparse_v1.json`. v47
(`b1c4f142172a5d9652e00e47b9c66a273d4961ef9c115caea6b8debb2d23922e`) was
selected over v43, v49, and v50b because it is the strongest candidate that
retains exact 372-view/one-model lineage, genuine non-pair-local multiview
tracks, zero independently detected calibrated pose/trajectory failures, and
the connected ring graph. Its strict gate is **not** claimed to pass: the
remaining failures are `mask_projection_per_ring` and
`mask_projection_per_view`. The dense lineage explicitly labels this as
`best_defensible_sparse_candidate`, keeps `sparse_gate_passed=false`, and
binds the model, strict-gate report, selection report, fixed audit, calibrated
classification, exact 1,530-pair graph, and immutable canonical snapshot
hashes. Fresh dense work may consume this explicit best-defensible lineage;
it must not relabel it as a strict sparse acceptance.

## Dense runtime-phase assertion and fresh geometric candidate (2026-09-15)

The first `dense_best_defensible_v1_full` attempt was interrupted after its
smoke phase had completed and while its full tile was still in the expected
photometric pre-pass. Its workspace and logs remain preserved as interrupted
diagnostic evidence; it was not promoted or reused as a completed dense
candidate. The smoke log records 15 `geom_consistency: 0` / `filter: 0`
blocks followed by 15 `geom_consistency: 1` / `filter: 1` blocks (the first
geometric block begins at log line 923).

The production runner now validates the runtime log for every completed
geometric PatchMatch call. `validate_patch_match_runtime_phases()` requires
both phases and their photometric-before-geometric order before the call is
accepted; `tests/test_v4_dense.py` covers the positive two-phase case and the
photometric-only failure. The COLMAP command builder and its argument ordering
were left unchanged.

A fresh versioned run completed under
`D:\\Side Projects\\CSX4213_V4_Dense_Work\\dense_best_defensible_v1_full_geometric_phase`.
Its production log contains exactly 372 photometric `geom_consistency: 0 /
filter: 0` blocks followed by exactly 372 geometric/filter `1/1` blocks, and
the runtime validator passed with `phase_order_valid=true`.  The completed
workspace has exactly 372 geometric depth maps and 372 geometric normal maps.
The fresh stereo fusion produced 1,931,168 points and fused-cloud SHA-256
`4a596f562d8bace90cac39deb38479f21a01eb9c09e02eb86fa26876e38c7d73`.

The unchanged strict post-fusion gate remains failed, not relabeled:
source-selection, ring-transition, contamination-provenance, mask-resolution,
and all eight region support/coverage measurements passed, but the anatomy gate
failed `finial:resolved_narrow_top_element` (finial/lid radius ratio
`1.0177845894` versus the fixed maximum `0.85`) and therefore
`no_major_vessel_scale_holes=false`.  Contamination remained measured and low
(board `0.0071916667`, cloth/background `0.0028833333`, pedestal webbing
`0.0`), with mean region support `0.9956062` and mean projected coverage
`1.0`.

The dense runner now preserves this complete strict failure and classifies it
as an explicit `dense_best_defensible` continuation record; it does not weaken
`postfusion_evidence_gate`.  The hash-bound selection is
`reconstruction/v4/reports/dense_best_defensible_v1_with_min3.json`, whose
selected default fusion remains the strongest measured candidate.  One bounded
same-map `StereoFusion.min_num_pixels=3` escalation was audited in a separate
versioned cloud (3,779,917 points, SHA-256
`24c4f28bc5de1a36aca238c5803d2235ca2387ae3ac1d91c701ad15b19008e9e`).  It
also failed the same finial anatomy gate (ratio `0.9994892707`) and had
slightly lower mean region support (`0.9934389`) and higher contamination, so
it remains comparison evidence rather than a substitution.

Poisson continued only from the selected fresh fused cloud and remains
diagnostic-only.  Depth-13/trim-10 produced 1,337 components with dominant
face fraction `0.6847469289` and second-largest `0.2619244906`, failing the
hard component gate.  The bounded scan-derived trim-5 variant produced 2,158
components and passes the fraction gate (`0.9925222754` dominant,
`0.0007196492` second), but its eight basis-bound region projections still fail
the real anatomy gate: the finial is a broad noisy cap rather than a resolved
narrow top element (ratio `1.0083513910`) and
`no_major_vessel_scale_holes=false`.  Its diagnostic report is
`reconstruction/v4/reports/raw_poisson_best_defensible_v1_depth13_trim5_diagnostic.json`.
No raw-Poisson promotion, Blender, GLB, appearance work, or publication is
authorized from these failures.

## Best-defensible Blender/GLB continuation with explicit anatomy limitation (2026-09-15)

The approved completion-first policy continued downstream from the selected
fresh fused cloud and trim-5 scan-derived Poisson diagnostic without relabeling
the failed dense or raw-anatomy gates. The new versioned authoring result is
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v4/`. It imports
the exact trim-5 raw Poisson source (`33fe1f6e7f696d8fd46d4a7c324d0e4f5fd9ff6a2ad4fb08e9f851a0437d6941`),
preserves `SM_V4_Poisson_Raw` and `SM_V4_Scan_CleanHigh`, and applies only
technical LOD0 cleanup. Blender's mesh validator removed 1,923 duplicate
polygons from the LOD0 copy (151,547 vertices, 294,713 faces); no vessel-scale
geometry was created, bridged, sculpted, lathed, mirrored, or reference-assisted.

The authoring blend SHA is
`e94fc932c5368b18e3d4fa22127968543465dec9913840944eaf4adabf0df3c9`.
The material `MAT_V4_Brass` uses the deterministic 158-image appearance
statistics (`photographic_projection_verified=false`) from
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v2/appearance_stats.json`
(SHA-256
`bae0f78aa119a14930c2045d0b27ecbe40bf6ea98b6a6da4e9b7c7d2d196a77e`, source
manifest SHA-256
`1b075d2d6e250ad562b2c4dd2158cc691cbe6b20cd2b49766ac08b9c63bc4a5c`). The
packed AO image is `T_V4_BestDefensible_AO`; one UV layer and the scripted
LOD0/normal/cleanup state are recorded in `mesh_cleanup_report.json` and
`blender_best_defensible_v1_trim5.json`.

The repaired GLB is
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v4/Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5.glb`
with SHA-256
`e35be3364c82521e2201f824ff4e21a02dfdbb420e722d1e25c923cfaab79011`.
The fresh factory-empty Blender re-import at
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v4/reimport/`
passed the complete technical checks: exactly one final mesh object, 296,967
imported vertices and 294,713 faces, finite positions and normals, one UV
layer, `MAT_V4_Brass`, packed/resolved AO, sane bounds, no debug objects, and
eight deterministic re-import renders. The re-import report is
`glb_reimport_verification.json` and hashes the same GLB.

This is a best-defensible technical downstream result, not a strict anatomy
acceptance. The source dense selection still has
`dense_strict_postfusion_gate_failed`; the raw trim-5 mesh passes the component
fractions but still fails `finial:resolved_narrow_top_element` and
`no_major_vessel_scale_holes=false`. The re-import renders preserve the same
scan-derived broad/noisy upper cap and do not establish a resolved narrow
finial. These failures remain visible in both export and re-import reports;
Blender was not used to compensate for missing vessel-scale anatomy.

## Execution ownership split and pre-Blender reset (2026-09-15)

After the host computer restart, the active direction was narrowed deliberately. The local Codex executor now owns **only the remaining work before Blender**: recover persisted 3072px `geo_g12` state rather than assuming the interrupted process completed; finish the bounded high-ring recovery if necessary; compare and freeze the strongest genuine dense candidate; finish/freeze the strongest defensible Poisson; correct/verify lineage, hashes and relevant pre-Blender tests; update the pre-Blender documentation; and complete the focused pre-Blender Git milestone with verified `origin/main` synchronization.

The handoff boundary is the verified, versioned, hash-bound Poisson mesh plus its complete evidence package. Once that exists, the Codex/local executor must stop. The superseded `best_defensible_v1_trim5_authoring_v4` Blender/GLB artifacts remain documented historical provenance; their bulky directory was removed during the authorized 2026-09-16 storage cleanup after the accepted v6/canonical assets and hashes were verified. ChatGPT using Blender MCP owns the scan-preserving Blender repair/finalization path, with the accepted v6 baseline and active v7 repair workspace preserved.

## Pre-Blender handoff freeze after restart (2026-09-15)

The persisted restart state was checked before rerunning compute. The old
`upper_geo_g12_3072_v1_smoke` and broad high-ring smoke stopped during their
geometric phases (3/8 and 4/15), with no complete production reports, so they
were retained as incomplete evidence. The bounded fresh run
`upper_geo_g12_3072_v2` completed in a separate workspace using CUDA GPU 0:
the smoke was 8/8 photometric plus 8/8 geometric and the production run was
37/37 of each map type. Its runtime phase gate and exact 37-reference
one-reference-one-write config passed. Mask-aware StereoFusion produced a
646,666-point geo_g12 cloud with SHA-256
`5bce4a2adb4b53900a0e1881e79978c83d77e1f2300e01097d7795ac0a82c980`.

The high-ring result did not improve the complete dense winner: its finial/lid
radius ratio was `1.0922269378` versus the baseline `1.0177845894`, the narrow
finial remained unresolved, and lid-tier connected support plus the
no-major-hole finding remained failed. Its run report is
`reconstruction/v4/reports/dense_upper_geo_g12_3072_v2_run.json` and its
status is `recovery_exhausted_unresolved`; it is comparison-only evidence.

The authoritative sparse V2 report is
`reconstruction/v4/repair/sparse_v1/best_defensible_sparse_v2.json` (SHA-256
`5121e2fd1834ede45423b8a302613fd85996a4e8889fe98690d4cecc7d6580f5`; source
model SHA-256 `b1c4f142172a5d9652e00e47b9c66a273d4961ef9c115caea6b8debb2d23922e`;
selection SHA-256 `d48687a4e92272ae99184eaf903011a0526e1023bbf3de009c1c782bc12ad2ce`).
The selected full dense cloud remains the fresh 372-view 2000px candidate
(`4a596f562d8bace90cac39deb38479f21a01eb9c09e02eb86fa26876e38c7d73`). A new
lineage wrapper,
`reconstruction/v4/reports/dense_best_defensible_v2_with_upper_recovery.json`
(SHA-256
`d6fad39ac7f30476eb89a2a4b2b49acb4b2eb2aa5549178bfcc4d900ed76fcf4`), binds
that unchanged cloud and its historical run/post-fusion reports to
`best_defensible_sparse_v2.json`. The wrapper verifies the current sparse
model, selection, sparse-gate, and track-provenance hashes; historical V1
reports are preserved rather than rewritten.

The final bounded Poisson comparison retained depth-13/trim-10 as negative
evidence (1,337 components; dominant `0.6847469289`, second `0.2619244906`)
and selected depth-13/trim-5 (2,158 components; dominant `0.9925222754`,
second `0.0007196492`) as the strongest connected scan-derived shell. The
mesh SHA-256 is
`33fe1f6e7f696d8fd46d4a7c324d0e4f5fd9ff6a2ad4fb08e9f851a0437d6941`. The
versioned handoff report is
`reconstruction/v4/reports/raw_poisson_best_defensible_v2_handoff.json`
(SHA-256
`4e1251e216b34783b87ff1e58664df8c0f306e44383431e4c9644443f8a9655a`). It
explicitly says `best_defensible_poisson_handoff`, `handoff_allowed=true`, and
strict `promotion_allowed=false`. The dense strict gate, unresolved
`finial:resolved_narrow_top_element`,
`no_major_vessel_scale_holes=false`, and semantic detached-component proof
remain failures/limitations; no anatomy was fabricated.

Fresh verification after the handoff edits: 106 V4 tests passed; 67 root,
script, and V4-test Python files compiled; and an independent hash/lineage
audit passed for sparse V2, the dense wrapper, the 3072px reports/fused cloud,
the Poisson attempts, the anatomy evidence, and the selected mesh. No
Blender/GLB path was opened or mutated by this pre-Blender phase.

## Final Blender cleanup, appearance, export, and publication candidate (2026-09-15)

The post-Blender continuation began from the frozen trim-5 Poisson handoff and the preserved v4 diagnostic authoring scene. The apparent duplicate model was diagnosed as an authoring visibility issue: `SM_V4_Scan_CleanHigh` and `SM_V4_Vessel_LOD0` occupied essentially the same scan-derived surface and were both visible in the viewport. Raw and CleanHigh are now preserved but hidden at both object and source/work-collection level; only LOD0 is visible and only LOD0 is exported. No missing anatomy was sculpted, bridged, mirrored, lathed, primitive-replaced, or reference-assisted.

A first technically valid finalization exposed a second defect during texture inspection: the inherited LOD0 UV map had only `0.0001723316` summed face UV area, so AO and normal-detail bakes used almost none of the texture. The accepted v6 finalizer therefore replaced only LOD0 UV coordinates with a deterministic signed-dominant-normal six-way box atlas. Geometry and transforms remain byte-independent but numerically identical before/after (Raw 4,952,940 / 9,899,268; CleanHigh 597,208 / 1,187,912; LOD0 151,547 / 294,713). The new atlas sums to `0.7165840083` UV face area with 11 near-zero-area faces, after which fresh 2048px AO and selected-to-active CleanHigh-to-LOD0 tangent normal/detail were baked successfully.

Appearance is derived from the complete 158-image uncoated project set rather than the coated geometry colors or any external reference. The frozen appearance-statistics SHA-256 is `bae0f78aa119a14930c2045d0b27ecbe40bf6ea98b6a6da4e9b7c7d2d196a77e`; BaseColor sRGB median is `[0.5176471, 0.4156863, 0.3568628]`, Roughness is `0.3079020830`, and metallic is `0.85`. `photographic_projection_verified=false` remains explicit. Final vertex colors are not exported.

The accepted versioned final is `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v6/`. Its `.blend` SHA-256 is `622a676a480a9b2a59c546de255146b1bd665ab54320bb04dff9eb8e03e3a273`; its GLB SHA-256 is `15ca1f76b773b5124a2b1c4abf16c63ca2380a1dbf019c84ac8214a5063e43de`. Fresh Blender 5.2 authoring verification passed preserved-source counts, finite geometry, source visibility, usable UV, packed texture and material-link checks. Fresh factory-empty GLB reimport passed exactly one final mesh, no nonmesh/debug export, finite positions/normals, sane bounds, UVs, material, BaseColor/Roughness/Normal/AO connectivity, and no vertex-color export.

Eight authoring views and eight fresh GLB reimport views were rendered with the same deterministic QA setup. Image comparison measured mean 8-bit MAE `0.00063673` and minimum PSNR `78.5622 dB`, so export/reimport is materially equivalent to authoring. All eight reimport views were visually inspected. The duplicate/z-fighting artifact is gone and the material is stable, but the reconstructed upper neck/lid/finial still contains scan-derived gaps, tears, broad/broken top geometry and detached-looking regions from several angles. Those findings match the frozen dense/Poisson failures (`finial:resolved_narrow_top_element=false`, `no_major_vessel_scale_holes=false`) and remain documented instead of being hidden with fabricated Blender geometry.

The previous canonical `.blend`/GLB hashes (`a06b6b75…0033` and `f3adb3a3…97c1`) plus rollback commit `625168138471dce0c8f01eda284dab7acd9be55f` were recorded before promotion. The verified v6 bytes were then copied to canonical `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend` and `.glb`, with exact hash equality checked. Overall V4 completion is therefore **best-defensible with explicit anatomy limitations**, not a strict anatomical pass.

Final publication verification completed after the asset checks: focused milestone commit `2d6e7b81bfe837321587669713b665396ad66592` was pushed to `origin/main`, Git LFS uploaded the 344 MB final `.blend`, a fresh fetch showed local HEAD and `origin/main` equal at that milestone, and `git lfs fsck` passed. Unrelated modified/untracked historical and user-owned workspace material remained unstaged and untouched.

## External V4 scratch cleanup (2026-09-16)

The external scratch root `D:\\Side Projects\\CSX4213_V4_Dense_Work` measured 258,953,992,807 bytes before cleanup. After verifying the authoritative selected sparse source model against `best_defensible_sparse_v2.json`, all failed/retry/smoke dense workspaces, superseded dense generations, the unpromoted `upper_geo_g12_3072_v2` recovery workspace, obsolete sparse candidates, partial mask snapshots, and disposable SQLite databases were deleted. The retained D: tree is only `workspace_v4repair_sparse_v1/sparse_candidate_v47_v43_fixed_pose_retriangulation` (5 files, 21,746,079 bytes), and its stable directory SHA-256 remains `b1c4f142172a5d9652e00e47b9c66a273d4961ef9c115caea6b8debb2d23922e`. Historical D: paths in older run/selection reports are provenance-only after this cleanup; the selected fused cloud, Poisson mesh, authoritative reports, curated previews, and active/historical Blender assets remain on C:.

## Public project documentation refresh and V7 boundary (2026-09-16)

`README.md` was rewritten as the concise repository entry point and `docs/PROJECT_REPORT.md` was added as the detailed project report. Both describe the real 688-image V4 capture, controlled acquisition, Grounding DINO-T + SAM 2.1 segmentation, ALIKED-N16Rot + LightGlue matching, COLMAP/pyCOLMAP SfM, CUDA PatchMatch Stereo, mask-aware fusion, Poisson meshing, scan-preserving Blender preparation, appearance derivation from 158 uncoated captures, and final GLB verification. `docs/README.md` now provides a documentation index. The public README/report were checked to contain no assistant/orchestration references.

The former V6 canonical asset remains historical rollback evidence; V139 is now the verified canonical V4 release. The V94 dense diagnostic used a 218-view 1000px GPU-0 PatchMatch workspace with the accepted corrected sparse graph while excluding `geo_g10`, `geo_g11`, and `geo_g12`; PatchMatch completed all 218 photometric and 218 geometric depth/normal maps, mask-aware fusion produced 194,205 points, and a diagnostic Poisson/QA set was generated. V94 became the selected V7 dense-derived geometry base after V96 completed but measured worse on the 3 cm translated-match ghost metric (`0.17226` versus `0.14061`).

## V7 scan-preserving Blender refinement (2026-09-17)

The reference-assisted-looking V113/V114 direction was rejected. V7 returned to the actual reconstructed V94 mesh. The synthetic planar bottom cap was isolated as exactly 70 faces with total area `0.3022912687200048 m²` and removed, leaving the physical bottom as one 72-edge boundary loop. V119 straightened the scan centerline from about `6.85°` to `0.67°` using a shear that preserved a level bottom plane. Guarded same-object CleanHigh donor transfer corrected the lower band without moving the opening boundary or accepting >15 mm donor correspondences. The top terminal was rounded and smoothed only through existing scan vertices; no sphere primitive or rebuilt finial was added.

The user then requested repeated full-surface cleanup passes, with special attention to the holder-to-middle-ring region and the upper tower/terminal. V139 (`SM_V4_V139_SCAN_PRESERVING_FINAL`) is the final scan-preserving result. The full vessel was audited in segmented height bands from eight angles, with additional top-oblique and underside views. Bounded smoothing/fairing reduced scan chatter across the holder, bowl, globe and tower. Same-object rotational repetition was used only as a capped per-height profile constraint where the scan was locally defective. Folded/cavity-like terminal geometry was repaired through existing vertices, local profile correction and robust sphere fitting; no remesh, primitive, generic side, rebuilt finial or lathed component was introduced. Final topology is 451,312 vertices / 902,838 triangular faces, one connected component, one intentional 72-edge bottom boundary loop, zero other non-manifold edges, no zero-area faces, no loose vertices, finite positions and identity transforms.

The versioned authoring master is `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/Thai_Libation_Vessel_V4_V7_SCAN_PRESERVING_FINAL_V139.blend`, 604,340,108 bytes, SHA-256 `951d11585d3b459251bb349acc5515877f4f223fb55189e37d5910f1716ee4f3`. `MAT_V4_V115_BrassStatsOnly` remains the final material and `photographic_projection_verified=false`. V139 was promoted to `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend` and exported to `Thai_Libation_Vessel_V4_FINAL.glb`. The current canonical `.blend` is 604,339,933 bytes with SHA-256 `5eacf012b9940d1997291c4c1e0720776e2a658bf2c60b3a7e567947695e3436`; a later re-save changed the byte hash from `6663d83303fefac34bbbee85132ad64322e2e9b0bc4db73776f2a882a9d07a0c` without changing the V139 mesh. Background Blender 5.2 verification matched the versioned master at 451,312 vertices / 902,838 triangles, `MAT_V4_V115_BrassStatsOnly`, identity transforms, dimensions, and semantic mesh SHA-256 `82348c02e61529d1f19d6e0a11e2c3fcf3228a354d1186c73691d84beb5a393b`. The canonical GLB is unchanged at SHA-256 `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`; its fresh Blender 5.2 factory-startup import verified exactly one mesh object with matching topology, material, transforms and dimensions. Final clay and brass QA were reviewed before promotion.

## Final professor submission handoff (2026-09-19)

A root-level `submission/` package was created from the verified V139 release. The package contains a direct copy of the canonical GLB, a new Blender 5.2 binary PLY export with triangulated geometry and normals, eight turntable renders, four regional closeups, two contact sheets, `README.txt`, and `SHA256SUMS.txt`. The PLY export is 24,373,990 bytes with SHA-256 `42ab10ebec3b69dc8dc7ce28463a9d04fd3f0ecb3b35f888b62bc3c222eb9ff1`; the GLB remains 27,082,828 bytes with SHA-256 `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`. Independent factory-startup import checks passed for both: one mesh, 451,312 vertices, 902,838 triangles, matching dimensions and identity transforms; the GLB also carries the expected statistics-only brass material. The final formatted paper is present at `docs/Thai_Libation_Brass_Vessel_Final_Report.docx`.
