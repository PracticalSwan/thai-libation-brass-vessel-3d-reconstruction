# Codex Handoff — V4 Full Reconstruction Repair

**Status:** Canonical implementation handoff for the current V4 repair.

## Read first

1. `AGENTS.md`
2. `CLAUDE.md`
3. `LESSONS.md`
4. `docs/memory-bank/active-context.md`
5. `docs/memory-bank/progress.md`
6. `docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`
7. `docs/superpowers/plans/2026-09-14-v4-full-repair.md`

## Completion goal

Complete the V4 repair end-to-end from the newest verified repository state as a full computer-vision-only reconstruction pipeline: first correct or safely replace the proven `geo_g10` sparse pose inconsistencies while preserving validated V4 evidence and unrelated work, then rebuild and select the best evidence-backed dense reconstruction using corrected source selection and strict trajectory/depth/coverage/contamination/continuity gates, produce a genuinely connected and anatomically complete Poisson vessel, perform only scripted/reproducible non-creative Blender LOD0/UV/baking/material/export work, derive appearance automatically from project-captured images/reconstructed data, export and fresh-reimport the final GLB, run all relevant regression/integrity checks, update the V4 documentation, and complete focused milestone/final commit-and-push verification; no external/reference-assisted geometry, manual mesh building/sculpting/shape completion, manual reference matching, or hand-authored appearance fallback is allowed, and fragmented geometry, incompatible dense maps, or fabricated vessel-scale anatomy must never be promoted.

## Start boundary

Start from Task 0/Task 1 of the canonical full-repair plan after recovering the newest live repository/runtime state. The historical V4 final is preserved evidence, not current acceptance. The first technical objective is the sparse-integrity regression and repaired/safely replaced sparse solution for the proven `geo_g10` pose inconsistencies. If accepted sparse camera geometry changes, historical true3 PatchMatch maps are diagnostic-only unless compatibility is explicitly proven; prepare a fresh compatible dense workspace and enforce one PatchMatch reference assignment per registered image.

Do not use Codex CLI. Preserve immutable source media, historical V4 evidence, private checkpoints, `.codegraph/`, unrelated `.ai-bridge/`, and unrelated user-owned changes. Use focused milestone commits/pushes only after verified sparse, dense, raw Poisson, and final Blender+GLB acceptance as defined in `AGENTS.md` and the canonical plan.

## Current hold point — sparse acceptance reopened

The former v10 pair-local sparse candidate and all downstream dense/PW0
artifacts are diagnostic evidence only. The strengthened sparse gate now
requires actual multi-view track distributions, complete-cloud per-view and
per-ring mask precision/recall/IoU, calibrated camera-center translation-scale
consistency, the fixed independent audit, one coherent 372-view model, and
immutable SQLite/model lineage. v10 fails the first two independent global
checks; v6 has strong mask agreement but duplicate-image tracks and calibrated
pose/shared-track failures; v8 has genuine multi-view tracks but fails mask
projection; the bounded calibrated ring-aligned v11/v12 repairs also fail
track, mask, translation, and/or reprojection gates. These candidates and
reports are preserved under `reconstruction/v4/repair/sparse_v1/` and the
external sparse workspace. No sparse promotion, milestone commit/push, dense
rerun, Poisson promotion, Blender work, or GLB work may start until a new
image-derived candidate passes every sparse gate.

## 2026-09-14 sparse-only diagnostic continuation

The source-component pose repair candidate v17 is the strongest tested
replacement, but it remains negative evidence: 372 registered images in one
model, 62,519 points, mean track length 6.3868, genuine multi-view track
distribution, all rotation/trajectory/camera-center checks passing, but the
independent mask gate fails (view precision p10 0.8527 versus the hard 0.90
floor; eight calibrated adjacent shared-track failures). It is preserved at
the external model path recorded in its gate/provenance reports.

The bounded source-track sanitizer candidates v18/v19/v20 also remain
negative. They have no deliberately allocated pair-local tracks and all
retained tracks are multi-view, but they fail independent mask projection,
shared-track continuity, and (for v18/v19) camera-center translation-scale
consistency. The source-point/source-translation variant v20 restores the
camera-center check but still fails mask recall/IoU and has 213 shared-track
failures.

The scan-derived mask-support diagnostics v21/v22/v25 are retained as
negative evidence, not acceptance tuning. v22 passes all ring-level mask
checks and v25 retains the v17 two-view tracks, but their per-view precision
p10 values are only 0.8729 and 0.8758 respectively (hard floor 0.90), with
36 and 32 shared-track failures. No candidate has therefore been promoted.
The pose-fixed point-only pyCOLMAP BA diagnostic made no measurable change to
v17 (mean reprojection 2.19998, 62,519 points) and is not an acceptance
result. Dense/PatchMatch, Poisson, Blender, GLB, milestone publication, and
pushes remain paused pending a genuinely coherent sparse candidate.

## Latest calibrated-unit hold point (2026-09-14)

The fixed audit was regenerated as v8 with explicit pyCOLMAP radian source
units and converted degree fields. The v3 classification retains 1,530
well-conditioned pairs, each with a same-solution calibrated rotation and
translation. The full-1530 remap report records zero failed pairs, 1,530
applied overrides, 1,530 passing R/t/E/F round-trips, exact raw-audit,
classification, and snapshot hashes, and an unchanged frozen canonical SQLite
file. The remapper is invoked with `--require-pose-overrides` and fails closed
instead of silently using a runtime pose.

The single fresh v31 pyCOLMAP 4.2 GLOMAP candidate is preserved at the external
workspace path recorded in its run report. It contains one reconstruction with
all 372 views and genuine multiview mapper tracks, but the unchanged sparse
gate fails independent track integrity, same-ring SO(3), and shared-track
continuity (15 calibrated rotation failures; 5,952 duplicate-image track
elements). v30's earlier 1,527-edge run is also negative evidence and is not
reused. Masks, camera-center/translation, trajectory, reprojection/intrinsics,
ring connectivity, one-model/372-view, angle units, and immutable-lineage
checks pass. Sparse promotion and all dense/Poisson/Blender/GLB work remain
blocked; no milestone or final commit/push is authorized from this state.

## Exact graph correction and v33 hold point (2026-09-14)

The v31 remap was not an exact 1,530-pair mapper graph: its disposable DB
contained 2,118 nonempty `two_view_geometries` rows. The 588 extras carried
91,345 inlier rows; all had raw `matches` rows as well. Treat v31 as an
invalid-input diagnostic, not as the final calibrated-graph verdict.

The fail-first exact-graph regression and production preflight are now in
`v4_repair.validate_exact_mapper_graph`,
`scripts/prepare_v4_exact_mapper_database.py`,
`scripts/reestimate_v4_two_view_geometry.py`, and
`scripts/run_v4_strict_global_mapping.py`. A fresh disposable DB physically
prunes every non-retained pair from both `two_view_geometries` and `matches`,
then verifies the exact set/count before the remap and before GLOMAP. The
canonical v3 SQLite snapshot is only copied/hashed; it is never opened by
SQLite or pyCOLMAP. Immutable SQLite read-only preflights avoid creating
journal sidecars on disposable inputs.

The exact source/preparation report records 1,530 verified pairs and
1,146,567 verified inlier rows, 1,530 raw-match pairs, no extra IDs, and a
stable canonical raw SHA. The calibrated remap records 1,530 rewrites,
zero failures, 1,530 pose overrides, and 1,530 passing R/t/E/F round-trips,
bound to the v8 audit, v3 classification, fixed-audit IDs, and v3 snapshot.

Exactly one corrected pyCOLMAP 4.2 GLOMAP candidate was then run at the
external model path recorded in
`track_provenance_v33_exact1530_global.json`. It returns one reconstruction
with all 372 intended views and genuine multiview tracks, but the unchanged
independent gate remains negative: 6,167 duplicate-image tracks, 18
same-ring calibrated SO(3) failures, and 25 shared-track failures. Track
length quantiles are p50/p90/p99 = 6/18/48 with 99.7942% >=3 views; masks,
camera-center/translation, trajectory, reprojection/intrinsics, ring
connectivity, one-model/372-view, model hash, and immutable lineage pass.
No sparse promotion, dense/PatchMatch, Poisson, Blender, GLB, milestone
commit, or push is authorized from this candidate. Further sparse work
requires new causal evidence; the exact-graph candidate and the invalid-input
v31 candidate remain preserved diagnostics.
