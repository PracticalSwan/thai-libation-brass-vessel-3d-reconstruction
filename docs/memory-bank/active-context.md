# Active Context

Updated: 2026-09-17

## Current focus

**V4 full reconstruction repair is the only active direction.** The repository now contains only the authoritative V4 capture lineage and current V4 reconstruction/repair assets. Superseded pre-V4 image datasets, processing outputs, code, tests, checkpoints, and implementation documents have been removed from the active project.

The active goal is:

> Complete V4 end-to-end now with the strongest defensible full computer-vision result realistically recoverable from the captured evidence. Use bounded repair/escalation ladders, keep strict research-grade gate failures honest, and stop indefinite candidate churn once a best-defensible image-derived sparse/dense/Poisson artifact can be frozen. Continue through fresh compatible dense reconstruction, Poisson, scan-preserving Blender/LOD0/UV/material work, final `.blend`/GLB, fresh GLB re-import, tests, hashes, docs, and focused publication. Never fabricate missing anatomy or convert a failed gate into a false pass.

## Completion-first best-result policy

Strict gates remain preferred targets and diagnostics, not permission to stall the project indefinitely. When a bounded evidence-backed stage cannot satisfy every strict threshold, select the strongest genuine CV artifact by measured geometry/coverage/continuity/contamination/provenance, label it `best-defensible`, preserve its failed checks unchanged, document the residual limitation, and continue downstream. Missing anatomy that the photographs cannot recover must remain a documented limitation rather than being manually/reference-assisted modeled in Blender.

Canonical current documents:

```text
docs/superpowers/specs/2026-09-14-v4-full-repair-design.md
docs/superpowers/plans/2026-09-14-v4-full-repair.md
```

Older superseded implementation documents have been removed; the two V4 full-repair documents above are the canonical technical references for current execution.

## Historical V4 baseline — preserve, do not blindly reuse

```text
source: CSX4213_Project_V4_Images/ (immutable)
688 JPEGs total
158 uncoated appearance/reference
107 empty-board/background
423 coated/marked object-bearing geometry
selected geometry views: 372
historical sparse registered: 372 / 372 in one model
historical sparse points: 65,560
historical observations: 460,628
historical mean track length: ~7.026
historical mean reprojection error: ~1.223 px
historical camera model: SIMPLE_RADIAL
historical corrected fusion masks: 372 / 372 under <image_name>.png
historical accepted dense SHA: df05019e2e56d1c54351f4b2cee161cbcfb7b782a3c6b0d4920d6e303f39d7d8
historical raw Poisson: 887,770 vertices / 1,669,931 faces
historical raw Poisson SHA: 55a4e92c9491cced508360d1645355ed447785f175df0556b400528fa9522941
historical canonical .blend/.glb: reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.*
```

Those artifacts prove the old pipeline executed. They do **not** prove sparse pose integrity or anatomical completeness.

## Confirmed root-cause findings

### Sparse `geo_g10` defect is real

Read-only investigation compared the final sparse camera solution with already-verified adjacent two-view geometry:

```text
IMG20260912145220 -> IMG20260912145223
verified two-view inliers: 449
two-view relative rotation: ~15.9 deg
final sparse relative rotation: ~103.8 deg
final shared 3D tracks: 0

IMG20260912145254 -> IMG20260912145256
two-view relative rotation: ~14.7 deg
final sparse relative rotation: ~32.2 deg
final shared 3D tracks: 10

IMG20260912145330 -> IMG20260912145333
two-view relative rotation: ~11.8 deg
final sparse relative rotation: ~61.5 deg
final shared 3D tracks: 7
```

The historical `geo_g10` virtual-camera trajectory also contains an acquisition-adjacent ray jump around 82.6 degrees. The source frames are ordinary neighboring turntable views, so this is not credible capture motion. The sparse report additionally records repeated CHOLMOD matrix-not-positive-definite warnings during final BA.

**Consequence:** full 372/372 registration and the global reprojection average are insufficient. The old sparse model fails the new integrity requirement and cannot remain the unquestioned dense coordinate system.

### `phase_01` is not physical azimuth

`phase_01` comes from ordered frame position within a logical ring. It is useful for deterministic order/locality, but it is not a measured angle. Critical cross-ring phase-offset confidence is weak. Repair pairing/source selection must prioritize verified inliers, two-view geometry, accepted co-visibility and camera sanity rather than forced phase-nearest edges.

### Dense collapse follows the suspect geometry

Historical true3 valid geometric depth support inside masks:

```text
g7_r1   median ~0.947  p10 ~0.922  min ~0.820
geo_g8  median ~0.890  p10 ~0.851  min ~0.744
geo_g9  median ~0.867  p10 ~0.825  min ~0.725
geo_g10 median ~0.659  p10 ~0.142  min ~0.003
geo_g11 median ~0.590  p10 ~0.299  min ~0.006
geo_g12 median ~0.337  p10 ~0.034  min ~0.021
```

Photometric medians are near 1.0, so image evidence exists while geometric consistency collapses. Dense-only tuning is therefore not the first repair step.

### Historical dense tiling provenance is invalid for the repair final

The historical combined true3 PatchMatch config set contains:

```text
567 reference entries
372 unique reference images
158 reference images duplicated
maximum reference repetition: 4
```

This allowed later tiles to overwrite earlier maps. The repaired dense runner must prove every registered image is a reference exactly once per complete candidate.

### Masks/background are not the main problem

The corrected fusion mask contract resolves 372/372 masks. Historical corrected-mask contamination evidence was low:

```text
board fraction ~0.00778
cloth/background fraction ~0.000492
pedestal-board webbing fraction 0
```

Representative masks retain the full vessel silhouette. Broad remasking is not the initial fix.

### Historical raw Poisson fails continuity

The preserved raw Poisson has 644 connected components and only about 66.57% of faces in the largest component, with two other very large components. The old raw gate passed despite this, so old raw/Blender acceptance is superseded.

## Current execution boundary

Sparse repair is no longer the active loop. The corrected v51 joint-BA harness selected all 372 images and configured 404,258 residuals, but the single permitted joint BA terminated `NO_CONVERGENCE`; that result remains negative evidence. The project has therefore frozen v47 as the current hash-bound **best-defensible** sparse source, with its strict mask-projection failures preserved rather than relabeled as passed. The V2 sparse selector additionally fixes the v50b provenance defect: the historical zero-residual failed BA is not eligible for downstream CV, while v47 remains selected.

The current work is split at a strict **pre-Blender handoff**:

```text
Codex/local executor now:
1. recover the post-restart state from persisted logs/workspaces/reports; do not assume the interrupted 3072px geo_g12 attempt finished
2. complete or resume the bounded 3072px high-ring recovery only when persisted evidence shows it is incomplete
3. compare/freeze the strongest genuine dense cloud with every strict failure preserved
4. finish the bounded Poisson ladder and freeze the strongest connected scan-derived Poisson input
5. verify sparse/dense/Poisson lineage, hashes, relevant tests/compile checks, pre-Blender docs, and focused pre-Blender commit/push + origin/main synchronization
6. STOP before Blender and hand off the verified Poisson + evidence package

ChatGPT + Blender MCP later:
7. scripted scan-preserving Blender cleanup/LOD0/UV/normal-detail/AO
8. project-derived appearance/material work
9. final .blend/GLB, fresh factory-empty re-import, >=8-view verification
10. final docs/hash integrity/final focused publication
```

Historical true3 depth/normal maps remain diagnostic-only and must not be reused as final evidence for the v47-derived reconstruction. No further sparse candidate churn is justified unless a new concrete causal defect blocks the fresh dense route itself. The pre-Blender executor must not touch Blender/GLB artifacts or Blender-specific finalization while completing this handoff.

## 2026-09-15 downstream best-defensible artifact checkpoint

The fresh v47-derived geometric PatchMatch run is complete and remains
untouched: its runtime log contains the full photometric pre-pass followed by
the geometric/filter phase, with exactly 372 geometric depth maps and 372
geometric normal maps. The unchanged strict post-fusion gate is preserved as a
failure, not weakened. The selected default fused cloud is the strongest
measured candidate by support/coverage/contamination evidence; the bounded
`min_num_pixels=3` same-map variant remains comparison evidence only. Poisson
continued from the selected fresh cloud, and trim-5 is retained as a
component-connected diagnostic while its unresolved finial and major-hole
anatomy gates remain failed.

The best-defensible technical Blender/GLB continuation is versioned under
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v4/`. It is
computer-vision-only and scan-preserving: raw Poisson and clean-high objects
are preserved, LOD0 received only duplicate/invalid polygon cleanup, UV/AO and
the deterministic uncoated-reference brass material were scripted, and
photographic texture projection remains explicitly unverified. The
hash-bound appearance statistics are the 158-image report
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v2/appearance_stats.json`
with SHA-256
`bae0f78aa119a14930c2045d0b27ecbe40bf6ea98b6a6da4e9b7c7d2d196a77e` and source
manifest SHA-256
`1b075d2d6e250ad562b2c4dd2158cc691cbe6b20cd2b49766ac08b9c63bc4a5c`.
The blend SHA
is `e94fc932c5368b18e3d4fa22127968543465dec9913840944eaf4adabf0df3c9`; the
GLB SHA is
`e35be3364c82521e2201f824ff4e21a02dfdbb420e722d1e25c923cfaab79011`.

A fresh factory-empty Blender re-import passed exactly one final mesh,
finite geometry/normals, UV/material/packed-AO resolution, sane bounds, no
debug objects, and eight deterministic renders. The report is
`reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v4/reimport/glb_reimport_verification.json`.
This does not promote the raw mesh or claim full vessel anatomy: the dense
strict gate, `finial:resolved_narrow_top_element`, and
`no_major_vessel_scale_holes=false` remain explicit limitations. No missing
anatomy was fabricated or hidden by Blender.

That Blender/GLB continuation is historical/diagnostic provenance, not the active acceptance target. During the authorized 2026-09-16 storage cleanup, the superseded bulky `best_defensible_v1_trim5_authoring_v2/` and `best_defensible_v1_trim5_authoring_v4/` directories were removed after the accepted v6/canonical assets and their hashes were independently verified. The measurements and hashes above remain provenance records, but those superseded paths are no longer expected to exist. The accepted v6 baseline and the active v7 repair workspace remain preserved.

## Protected boundaries

- `CSX4213_Project_V4_Images/` is the only authoritative raw source tree. Keep all 688 photographs immutable; it is intentional Git LFS repository content.
- `capture_v4/derived/mvs_images/`, `capture_v4/derived/masks/`, and `capture_v4/derived/feature_masks/` are the only current processed-image trees. They are intentional Git LFS repository content and must remain aligned with the V4 manifests.
- `.codegraph/` runtime/database artifacts remain local and uncommitted.
- Unrelated `.ai-bridge/` and unrelated user-owned changes remain outside this cleanup/publication scope.

Current V4 sparse/dense/Poisson/Blender/GLB artifacts remain rollback/evidence and must not be overwritten during candidate work.

## Completion standard

For the current Codex/local phase, completion means a strongest-defensible dense result and strongest connected scan-derived Poisson are frozen with correct provenance, hashes, relevant pre-Blender verification/docs, and verified focused Git publication, then handed off without touching Blender. Overall V4 completion occurs only after ChatGPT + Blender MCP performs the scan-preserving Blender/final-asset stages, fresh GLB re-import, final verification/docs, and final publication. Strict research-grade failures may remain only after the bounded repair ladder is exhausted; those checks must remain explicitly failed and be reported as residual limitations rather than being weakened or hidden.

## 2026-09-14 sparse architecture-switch checkpoint

The MAX_VISIBLE_POINTS_RATIO incremental candidate remains negative evidence: the unchanged v3 audit rejects ten full SO(3) relative-rotation checks. No further incremental mapper/order variants are authorized for this repair.

SQLite provenance is now manifest-bound. `canonical_sqlite_snapshot_v3.json` records the frozen snapshot raw SHA `658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a` and creation logical SHA `ec4d9a4c9a548903d71704401e50e666408a933dfdb19cc02cbe8e93fd860f97`. Audit, remap, and evaluator commands validate that manifest, byte-copy a disposable working database, and open only that working copy. The final independent hash check remained exact and no canonical `-wal`/`-shm` sidecars exist.

The fixed 1,868-pair audit was rerun with deterministic calibrated RANSAC seed 4201 as `independent_audit_geometry_v6_deterministic.json`. The raw records remain complete and hash-bound. Calibrated classification retained 1,527 well-conditioned rotations and preserved 341 contradictory/unconditioned records; 13 had measured local-cycle inconsistency, while pairs without a local two-hop cycle are recorded as `not_available` rather than treated as contradictory. The only previously recorded mapping exclusion remains `IMG20260912145303.jpg -> IMG20260912145304.jpg`.

One pyCOLMAP 4.2.0 `global_mapping`/GLOMAP candidate was then run from the corrected disposable graph. It returned one model containing all 372 views, 34,092 points, 304,023 observations, and mean reprojection error about 1.1801 px. The calibrated fixed-audit gate is **not passed**: 18 well-conditioned same-ring pairs retain large SO(3) disagreement and 24 retain shared-track collapse. The candidate and both the pre-calibrated diagnostic gate and final calibrated gate are preserved at `D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1\sparse_candidate_v6_glomap_global\0`, `reconstruction/v4/repair/sparse_v1/repair_sparse_gate_v6_glomap.json`, and `reconstruction/v4/repair/sparse_v1/repair_sparse_gate_v6_glomap_calibrated.json` as negative evidence. pyCOLMAP reported that this build lacks CUDA and recorded a CPU fallback for GLOMAP positioning/BA; this is not being hidden.

Because the GLOMAP sparse gate failed, no sparse promotion, milestone commit/push, dense workspace, PatchMatch, fusion, Poisson, Blender, or GLB work may start from this candidate. The next repair iteration must change architecture or evidence selection without weakening the fixed audit; all existing candidate databases, reports, and contradictions remain preserved.

## 2026-09-14 accepted sparse architecture-switch checkpoint

The failed GLOMAP result was not followed by more incremental/order variants. A new computer-vision-only architecture was evaluated from the same immutable calibrated evidence:

- deterministic maximum-spanning initialization plus robust SO(3) synchronization for each acquisition-local ring;
- deterministic robust cross-ring consensus over calibrated rotations, with every fixed-audit record retained and consensus outliers recorded rather than deleted;
- disjoint pair-local 3D tracks rebuilt from verified two-view inlier correspondences against the synchronized camera field;
- no phase/turntable pose prior, external/reference geometry, manual geometry, or appearance assistance.

The versioned candidate is accepted by the complete fixed independent audit at:

```text
model: D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1\sparse_candidate_v10_rotation_consensus_pairwise_tracks\0
model directory SHA-256: 78636a8279cb0114fad17a7e2d7f100b53188cbd21e7c96c224844d333da8c1d
gate: reconstruction/v4/repair/sparse_v1/repair_sparse_gate_v10_rotation_consensus_pairwise_tracks.json
gate SHA-256: dabafe769a0a696f53014fa735f49cf68d48024aef2f32d35c42fe3ce563c5e7
accepted lineage: reconstruction/v4/repair/sparse_v1/accepted_sparse_v4_rotation_consensus_v1.json
```

Measured acceptance evidence is 372/372 registered views in exactly one model, 54,115 finite sparse points, 108,230 observations, mean reprojection error ~0.9514 px, a connected six-ring graph, and all sparse checks passing. The fixed audit remains 1,868 pairs (1,527 well-conditioned calibrated records and 341 preserved contradictory/unconditioned records); the existing single independently-degenerate `IMG20260912145303.jpg -> IMG20260912145304.jpg` mapping exclusion is unchanged. Pair-local tracks have mean length 2.0 by construction; shared-track continuity is nevertheless measured against every accepted same-ring audit edge and passes the unchanged minimum-20 gate.

The canonical SQLite snapshot was restored from a verified byte-identical backup after an exploratory probe accidentally opened the canonical path; the final raw hash is again exactly `658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a`, the creation logical digest is unchanged, and no `-wal`/`-shm` sidecars exist. The direct-open policy remains forbidden: future audit, dense-preparation, or verification commands must create and open only disposable working copies from `canonical_sqlite_snapshot_v3.json`.

Sparse milestone publication is now authorized by the passing gate. Dense work may begin only from this accepted sparse hash, in a fresh compatible workspace; all historical dense maps remain diagnostic-only.

## 2026-09-15 pre-Blender V4 handoff freeze

The post-restart state was audited before any rerun. The earlier
`upper_geo_g12_3072_v1_smoke` and broad high-ring smoke were incomplete (their
geometric phases stopped at 3/8 and 4/15), so a fresh versioned run was
justified. `scripts/run_v4_upper_dense_recovery.py --tag
upper_geo_g12_3072_v2 --smoke-count 8` then completed on CUDA GPU 0 in its
separate workspace with 8/8 smoke and 37/37 production photometric and
geometric depth/normal maps. The production runtime phase gate passed, the
one-reference-one-write config contains 37 unique geo_g12 references, and
mask-aware StereoFusion produced 646,666 points (fused SHA-256
`5bce4a2adb4b53900a0e1881e79978c83d77e1f2300e01097d7795ac0a82c980`).

The high-ring post-fusion evidence remains a failure, not a promotion: the
candidate finial/lid radius ratio is `1.0922269378` versus the full dense
baseline `1.0177845894`, `resolved_narrow_top_element=false`, and the
lid-tier connected-support and no-major-hole checks remain failed. The run is
therefore recorded as `recovery_exhausted_unresolved` comparison evidence.

The authoritative sparse V2 report is
`reconstruction/v4/repair/sparse_v1/best_defensible_sparse_v2.json` (SHA-256
`5121e2fd1834ede45423b8a302613fd85996a4e8889fe98690d4cecc7d6580f5`; source
model SHA-256 `b1c4f142172a5d9652e00e47b9c66a273d4961ef9c115caea6b8debb2d23922e`;
selection SHA-256 `d48687a4e92272ae99184eaf903011a0526e1023bbf3de009c1c782bc12ad2ce`).
The complete 372-view 2000px dense winner is rebound, without changing its
fused bytes, to the current sparse V2 lineage in
`reconstruction/v4/reports/dense_best_defensible_v2_with_upper_recovery.json`
(report SHA-256
`d6fad39ac7f30476eb89a2a4b2b49acb4b2eb2aa5549178bfcc4d900ed76fcf4`; fused
SHA-256 `4a596f562d8bace90cac39deb38479f21a01eb9c09e02eb86fa26876e38c7d73`).
Historical V1-named dense reports remain unchanged evidence. Sparse V2 is
rehash-validated for its source model, selection report, sparse gate, and
track-provenance file whenever it enters dense lineage.

The bounded Poisson attempts were compared from that dense lineage. Depth 13 /
trim 10 remains rejected (1,337 components, dominant fraction `0.6847469289`,
second `0.2619244906`); trim 5 is the strongest connected candidate (2,158
components, dominant `0.9925222754`, second `0.0007196492`, mesh SHA-256
`33fe1f6e7f696d8fd46d4a7c324d0e4f5fd9ff6a2ad4fb08e9f851a0437d6941`). The
versioned handoff is
`reconstruction/v4/reports/raw_poisson_best_defensible_v2_handoff.json`
(report SHA-256
`4e1251e216b34783b87ff1e58664df8c0f306e44383431e4c9644443f8a9655a`). It is
`best_defensible_poisson_handoff` with `handoff_allowed=true` and strict
`promotion_allowed=false`; `finial:resolved_narrow_top_element`,
`no_major_vessel_scale_holes=false`, and semantic detached-component proof
remain explicit failures/limitations. No Blender or GLB artifact was opened,
modified, regenerated, cleaned, exported, or re-imported in this phase.

## 2026-09-15 final Blender + GLB completion

The post-Blender owner completed the frozen Poisson handoff without changing the reconstruction geometry. The historical apparent duplicate was traced to `SM_V4_Scan_CleanHigh` being visible directly over `SM_V4_Vessel_LOD0` in the authoring viewport; the GLB itself contained only one mesh. The final master preserves `SM_V4_Poisson_Raw` (4,952,940 vertices / 9,899,268 faces) and `SM_V4_Scan_CleanHigh` (597,208 / 1,187,912) as hidden rollback/source objects, while final LOD0 remains exactly 151,547 vertices / 294,713 faces with unchanged mesh geometry and transforms.

The accepted versioned authoring directory is `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v6/`. The final master SHA-256 is `622a676a480a9b2a59c546de255146b1bd665ab54320bb04dff9eb8e03e3a273` and the final GLB SHA-256 is `15ca1f76b773b5124a2b1c4abf16c63ca2380a1dbf019c84ac8214a5063e43de`. The inherited UV layout had only ~0.0001723 summed face UV area; a deterministic signed-dominant-normal six-way box atlas replaced only the LOD0 UV coordinates, increasing summed UV area to ~0.716584 without changing vertices, faces, or transforms. Fresh 2048px AO and CleanHigh-to-LOD0 tangent normal maps were then baked from scan geometry.

Final appearance is fail-closed and project-derived. BaseColor and Roughness come from the frozen 158-image uncoated appearance statistics (`bae0f78aa119a14930c2045d0b27ecbe40bf6ea98b6a6da4e9b7c7d2d196a77e`), with source-manifest SHA-256 `1b075d2d6e250ad562b2c4dd2158cc691cbe6b20cd2b49766ac08b9c63bc4a5c`. `photographic_projection_verified=false`; coated geometry vertex colors are not exported, and no manual/reference-assisted color or texture matching is claimed.

A fresh factory-empty Blender 5.2 GLB import passed one-final-mesh, finite position/normal, UV, BaseColor/Roughness/Normal/AO connection, material, bounds, no-nonmesh-export and no-vertex-color checks. Eight final authoring and eight fresh-reimport views were rendered. Their mean 8-bit pixel MAE is ~0.000637 and minimum PSNR is ~78.56 dB, establishing material/shape export equivalence. All eight reimport views were also visually inspected: duplicate z-fighting is gone and the material is stable, but the scan-derived upper neck/lid/finial remains broken/open/noisy in multiple views. This matches the frozen dense/Poisson failures and was deliberately not repaired with fabricated geometry. The result is therefore complete as the strongest defensible genuine CV artifact, not a strict anatomy pass.

The verified v6 bytes were promoted to `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend` and `.glb` only after the final checks. Previous canonical hashes and Git rollback commit `625168138471dce0c8f01eda284dab7acd9be55f` are recorded in the v6 `canonical_promotion_report.json`.

Final publication is complete. The focused Blender/GLB milestone commit `2d6e7b81bfe837321587669713b665396ad66592` was pushed to `origin/main`; Git LFS uploaded the 344 MB final Blender object, a post-push fetch confirmed local HEAD and `origin/main` were identical at that milestone, and `git lfs fsck` returned `Git LFS fsck OK`. The later documentation-only closure records these completed checks without changing the accepted final asset bytes.

## 2026-09-16 storage-cleanup boundary

The external scratch root `D:\\Side Projects\\CSX4213_V4_Dense_Work` was reduced from approximately 258.954 GB to 21.746 MB after the selected C: dense cloud, Poisson mesh, reports, report images, and Blender rollback/current assets were independently preserved. All failed/retry/smoke dense workspaces, the unpromoted 3072px upper-ring workspace, obsolete sparse candidates, and disposable sparse databases were removed. Historical D: paths in preserved reports remain provenance records and are not expected to exist after this cleanup. The only retained D: artifact is `workspace_v4repair_sparse_v1/sparse_candidate_v47_v43_fixed_pose_retriangulation`, containing five COLMAP model files and matching authoritative source-model SHA-256 `b1c4f142172a5d9652e00e47b9c66a273d4961ef9c115caea6b8debb2d23922e`.

## 2026-09-16 documentation/publication refresh

The human-facing project documentation was rebuilt around the actual image-to-3D workflow. `README.md` now gives the concise project overview, 688-image dataset split, end-to-end computer-vision pipeline, selected artifacts, representative evidence, material workflow, validation approach and repository map. `docs/PROJECT_REPORT.md` is the detailed report covering acquisition, Grounding DINO-T + SAM 2.1 isolation, ALIKED-N16Rot + LightGlue correspondence, COLMAP/pyCOLMAP sparse reconstruction, CUDA PatchMatch Stereo, mask-aware fusion, Poisson meshing, scan-preserving Blender asset preparation, project-image-derived brass appearance and clean GLB re-import verification. `docs/README.md` indexes current and historical technical documentation.

The former V6 canonical asset remains historical rollback evidence; V139 is now the canonical V4 release. The V94 COLMAP diagnostic excluding `geo_g10`, `geo_g11` and `geo_g12` completed successfully on GPU 0: 218 photometric and 218 geometric depth/normal maps were produced, mask-aware fusion completed with 194,205 points, and the diagnostic Poisson/QA outputs were generated. V94 became the selected V7 dense-derived geometry base after later comparison showed V96 had a worse 3 cm translated-match ghost metric (`0.17226` versus V94 `0.14061`).

## 2026-09-17 V7 scan-preserving final release

The rejected V113/V114 reference-assisted-looking direction remains prohibited. The completed authoring object is `SM_V4_V139_SCAN_PRESERVING_FINAL`, saved in `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/Thai_Libation_Vessel_V4_V7_SCAN_PRESERVING_FINAL_V139.blend` (604,340,108 bytes; SHA-256 `951d11585d3b459251bb349acc5515877f4f223fb55189e37d5910f1716ee4f3`). Relevant Raw/CleanHigh/V6/V94/V115/V116/V119/V120/V124/V128/V136 rollback/provenance objects remain inside the final authoring scene where applicable; superseded standalone V6/V128 files are removed from the finished working tree and remain recoverable from Git history.

V139 remains the reconstructed V94 mesh. Exactly 70 synthetic planar cap faces (area `0.3022912687200048 m²`) are removed and the physical bottom remains one intentional 72-edge boundary loop. The lower band uses guarded same-object CleanHigh evidence; later repair stages use only existing reconstructed vertices and same-scan rotational profile constraints. Every major height band was inspected from eight angles, with extra top-oblique and underside views. The holder-to-middle-ring region, bowl, globe, tower, upper ring and terminal were smoothed/fared without remeshing or replacing anatomy. The terminal was repaired from existing scan vertices by bounded profile correction, local fairing and robust sphere fitting. Final topology is 451,312 vertices / 902,838 triangular faces, one connected component, one 72-edge boundary loop, zero other non-manifold edges, no zero-area faces, no loose vertices, finite positions and identity transforms.

The material is `MAT_V4_V115_BrassStatsOnly` from the verified 158-image appearance statistics (`photographic_projection_verified=false`). Final clay QA covers nine segmented surface bands plus the full turntable, four top-oblique views and underside views; final brass QA covers an eight-angle turntable and closeups. V139 has been promoted to `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend` (current SHA-256 `5eacf012b9940d1997291c4c1e0720776e2a658bf2c60b3a7e567947695e3436`) and exported to `Thai_Libation_Vessel_V4_FINAL.glb` (SHA-256 `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`). The canonical `.blend` was later re-saved; background Blender 5.2 verification matched the versioned V139 master in 451,312 vertices / 902,838 triangles, material, identity transforms, dimensions, and semantic mesh SHA-256 `82348c02e61529d1f19d6e0a11e2c3fcf3228a354d1186c73691d84beb5a393b`. The immediately previous canonical `.blend` byte hash was `6663d83303fefac34bbbee85132ad64322e2e9b0bc4db73776f2a882a9d07a0c`. A fresh Blender 5.2 factory-startup import verified the unchanged GLB.

## 2026-09-19 final professor submission package

The project is packaged for final handoff under `submission/`. `Thai_Libation_Vessel_Final.glb` is an exact copy of the verified canonical GLB (27,082,828 bytes; SHA-256 `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`). `Thai_Libation_Vessel_Final.ply` is a Blender 5.2 binary triangulated export of `SM_V4_V139_SCAN_PRESERVING_FINAL` with normals (24,373,990 bytes; SHA-256 `42ab10ebec3b69dc8dc7ce28463a9d04fd3f0ecb3b35f888b62bc3c222eb9ff1`). Fresh factory-startup imports verified both deliverables as one mesh with 451,312 vertices / 902,838 triangles, identity transforms, and matching dimensions; the GLB also retains `MAT_V4_V115_BrassStatsOnly`. The user-approved ready package contains eight 45-degree turntable images, one turntable contact sheet, `README.txt`, and a regenerated `SHA256SUMS.txt`. The formatted final paper is retained as both `docs/Thai_Libation_Brass_Vessel_Final_Report.docx` and `docs/Thai_Libation_Brass_Vessel_Final_Report.pdf`.

## 2026-09-20 finished-project cleanup

Final cleanup removed 624 obsolete/local files totaling about 1.685 GiB: the superseded V6 authoring/export directory, standalone V128 checkpoint, pre-V139 V7 repair/diagnostic renders, historical V6 comparison renders and V6-only verification reports, transient `reconstruction/v4/work/` logs, the old interim authoring/AO artifacts, and the duplicate canonical `.blend1`. The immutable V4 source/processed images, selected sparse/dense/Poisson lineage, current reports, scripts/tests, V139 master/final QA, canonical `.blend`/GLB, final DOCX/PDF report, setup photographs, and user-approved submission remain protected. Previously committed historical bytes remain available from Git history.
