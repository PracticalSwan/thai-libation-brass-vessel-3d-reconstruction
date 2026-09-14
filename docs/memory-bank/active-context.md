# Active Context

Updated: 2026-09-14

## Current focus

**V4 full reconstruction repair is the only active direction.** V1/V2 are professor-rejected historical work and V3 is user-rejected historical work. The prior V4 final is preserved as historical evidence but is no longer accepted for the user's requirement of a visually complete Thai libation vessel.

The active goal is:

> Complete the V4 repair end-to-end from the newest verified repository state: first correct or safely replace the proven `geo_g10` sparse pose inconsistencies while preserving validated V4 evidence and unrelated work, then rebuild and select the best evidence-backed dense reconstruction using corrected source selection and strict trajectory/depth/coverage/contamination/continuity gates, produce a genuinely connected and anatomically complete Poisson vessel, finish only scan-preserving Blender/LOD0/UV/material work with honest appearance provenance, export and fresh-reimport the final GLB, run all relevant regression/integrity checks, update the V4 documentation, and complete focused milestone/final commit-and-push verification.

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

Implementation has **not** yet completed the new full-repair plan. Start from Task 0/Task 1 of the canonical plan, not from the old “final V4 complete” checkpoint.

Immediate order:

```text
1. protect/recover live state
2. implement sparse-integrity diagnostics + historical-failure regression
3. build corrected verified sparse constraint graph
4. repair or safely replace sparse reconstruction
5. accept repaired sparse and milestone commit/push
6. create fresh compatible dense workspace; reject duplicate reference writes
7. run bounded repaired CUDA PatchMatch/fusion ladder
8. strict dense anatomical acceptance + milestone commit/push
9. strict connected Poisson acceptance + milestone commit/push
10. scan-preserving Blender/LOD0/UV/detail/AO
11. honest uncoated-reference appearance workflow
12. final .blend/.glb, fresh GLB re-import, eight-view verification
13. tests/hashes/docs/final focused commit+push and remote verification
```

If accepted sparse camera geometry changes, historical true3 depth/normal maps are diagnostic-only and must not be reused as final dense evidence unless compatibility is explicitly proven.

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

Do not mark V4 complete until the repaired sparse candidate passes integrity, compatible dense reconstruction passes depth/coverage/contamination/anatomical gates, Poisson passes the hard component/no-major-hole gate, Blender cleanup remains scan-derived, appearance provenance is honest, and the final GLB passes fresh factory-empty re-import with anatomy preserved. Completion must also include relevant tests/hash integrity, current docs, and verified focused Git publication.

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
