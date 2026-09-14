# Progress

Updated: 2026-09-14

## Current project state

The project is at **V4 full-repair implementation start**. The former V4 “final-model complete” checkpoint is superseded for the user's visually-complete-vessel requirement because later investigation proved an upstream sparse-pose defect and a dense-provenance defect that invalidate the old acceptance chain.

Historical status:

- V1: rejected by professor; historical only.
- V2: rejected by professor; historical only.
- V3: rejected by user after Blender inspection; historical only.
- Prior V4 final: preserved as historical baseline/evidence, not current acceptance.
- Large intentional historical cleanup diff remains user-owned and must not be restored.

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

- `CSX4213_Project_V4_Images/`
- `IMG20260826122949/`
- `analysis/ml/checkpoints/`
- `.codegraph/`
- unrelated `.ai-bridge/`
- unrelated modified files, including current user-owned `local_reconstruction.py` and `tests/test_learned_sparse_recovery.py` unless the repair independently requires them
- historical V4 dense/mesh/Blender/GLB artifacts as rollback/evidence

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

## Accepted sparse architecture-switch checkpoint (2026-09-14)

The one permitted pyCOLMAP 4.2.0 GLOMAP candidate failed the unchanged calibrated fixed audit, so incremental mapper/order variants were stopped and preserved as negative evidence. The replacement computer-vision-only architecture uses calibrated ALIKED/LightGlue evidence: robust same-ring SO(3) synchronization, deterministic robust cross-ring rotation consensus, and disjoint pair-local verified tracks rebuilt against the synchronized camera field. It uses no phase/turntable pose prior, external/reference geometry, manual geometry, or appearance data.

Accepted sparse artifacts:

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
rotation-consensus sparse replacement            COMPLETE / STRICT GATE PASSED
sparse milestone publication                     NEXT / authorized after focused tests and diff review
fresh compatible dense workspace                 NOT STARTED
PatchMatch/fusion/depth/coverage/contamination   NOT STARTED
Poisson/Blender/GLB                              NOT STARTED
```
