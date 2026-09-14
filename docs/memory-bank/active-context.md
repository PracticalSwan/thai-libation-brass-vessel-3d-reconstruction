# Active Context

Updated: 2026-09-14

## Current focus

**V4 full reconstruction repair is the only active direction.** V1/V2 are professor-rejected historical work and V3 is user-rejected historical work. The prior V4 final is preserved as historical evidence but is no longer accepted for the user's requirement of a visually complete Thai libation vessel.

The active goal is:

> Complete V4 end-to-end now with the strongest defensible full computer-vision result realistically recoverable from the captured evidence. Use bounded repair/escalation ladders, keep strict research-grade gate failures honest, and stop indefinite candidate churn once a best-defensible image-derived sparse/dense/Poisson artifact can be frozen. Continue through fresh compatible dense reconstruction, Poisson, scan-preserving Blender/LOD0/UV/material work, final `.blend`/GLB, fresh GLB re-import, tests, hashes, docs, and focused publication. Never fabricate missing anatomy or convert a failed gate into a false pass.

## Completion-first best-result policy

Strict gates remain preferred targets and diagnostics, not permission to stall the project indefinitely. When a bounded evidence-backed stage cannot satisfy every strict threshold, select the strongest genuine CV artifact by measured geometry/coverage/continuity/contamination/provenance, label it `best-defensible`, preserve its failed checks unchanged, document the residual limitation, and continue downstream. Missing anatomy that the photographs cannot recover must remain a documented limitation rather than being manually/reference-assisted modeled in Blender.

Canonical current documents:

```text
docs/superpowers/specs/2026-09-14-v4-full-repair-design.md
docs/superpowers/plans/2026-09-14-v4-full-repair.md
```

The prior dense-only 2026-09-14 plan/spec and 2026-09-10 fast-end-to-end checkpoint are superseded for new execution.

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

Sparse repair is no longer the active loop. The corrected v51 joint-BA harness selected all 372 images and configured 404,258 residuals, but the single permitted joint BA terminated `NO_CONVERGENCE`; that result remains negative evidence. The project has therefore frozen v47 as the current hash-bound **best-defensible** sparse source, with its strict mask-projection failures preserved rather than relabeled as passed.

The active downstream order is now:

```text
1. finish the fresh v47-derived CUDA dense run with exactly one PatchMatch reference write per registered image
2. select the strongest genuine dense cloud by measured depth support, projected coverage, contamination and anatomy; preserve any failed strict targets
3. run bounded Poisson reconstruction and select the strongest connected scan-derived mesh
4. perform only scripted scan-preserving Blender cleanup/LOD0/UV/detail/AO
5. build automated project-image-derived brass appearance; no invented/reference-assisted texture or geometry
6. save final .blend and GLB, fresh factory-empty GLB re-import, >=8-view verification
7. run relevant tests/hashes/docs and complete focused commit/push + remote verification
```

Historical true3 depth/normal maps remain diagnostic-only and must not be reused as final evidence for the v47-derived reconstruction. No further sparse candidate churn is justified unless a new concrete causal defect blocks the fresh dense route itself.

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

## Protected boundaries

Do not modify or publish:

- `CSX4213_Project_V4_Images/`
- `IMG20260826122949/`
- `analysis/ml/checkpoints/`
- `.codegraph/` runtime/database artifacts
- unrelated `.ai-bridge/`
- unrelated modified files such as existing user-owned `local_reconstruction.py` / `tests/test_learned_sparse_recovery.py` changes unless the repair independently requires and verifies them

Historical V4 sparse/dense/Poisson/Blender/GLB artifacts remain rollback/evidence and must not be overwritten during candidate work.

## Completion standard

Mark V4 complete when the strongest defensible full CV result has been carried through fresh compatible dense reconstruction, Poisson, scan-preserving Blender, final `.blend`/GLB, fresh factory-empty GLB re-import, relevant tests/hash integrity, current docs, and verified focused Git publication. Strict research-grade failures may remain only after the bounded repair ladder is exhausted; those checks must remain explicitly failed and be reported as residual limitations rather than being weakened or hidden.

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
