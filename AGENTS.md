# Project Agent Instructions

## Scope and objective

This repository is the CSX4213 Computer Vision project for reconstructing a real Thai brass libation vessel from photographs.

Current phase: **V7 scan-preserving refinement and publication maintenance.** The V4 image-to-3D pipeline has a verified canonical end-to-end baseline: selected sparse lineage, selected dense fused cloud, selected Poisson surface, canonical Blender master, and canonical GLB. Those artifacts remain the protected baseline while bounded V7 reconstruction diagnostics investigate whether additional image-derived evidence can improve the visible reconstruction without manual/reference-assisted modeling.

The current goal is to preserve the verified V4 baseline, evaluate only completed and reproducible V7 evidence, promote a V7 result only when it is demonstrably stronger and passes the required technical verification, and keep public documentation synchronized with the actual project pipeline and deliverables. A live dense/CUDA workspace is never publication-ready merely because files exist; wait for its owning process to finish, then inspect its reports, hashes, geometry, and visual evidence before promotion.

Canonical technical history remains in:

- `docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`
- `docs/superpowers/plans/2026-09-14-v4-full-repair.md`
- `docs/memory-bank/active-context.md`
- `docs/memory-bank/progress.md`

Public-facing project documentation is:

- `README.md`
- `docs/PROJECT_REPORT.md`
- `docs/README.md`

The public README and project report must describe the project, captured dataset, computer vision pipeline, produced artifacts, and verification without exposing implementation-assistant/orchestration details or internal repair-loop narrative.

## Current execution boundary — 2026-09-16

- Preserve the verified canonical V4 `.blend`/GLB and selected sparse/dense/Poisson artifacts until a complete V7 candidate is independently verified and explicitly promoted.
- Do not interrupt, delete, overwrite, stage, or publish a live V7/COLMAP workspace while its process is still running.
- Keep incomplete V7 runtime logs, temporary dense workspaces, large experimental meshes, caches, and repair-review probes out of Git publication.
- After a computer restart or process interruption, inspect persisted logs, workspaces, reports, and hashes before rerunning anything.
- Continue to prohibit fabricated/manual anatomy completion: all promoted geometry must remain image-derived and scan-preserving.

## Protected material

- `CSX4213_Project_V4_Images/` is the immutable final V4 source set: 688 JPEGs = 158 uncoated appearance inputs, 107 empty-board/background, and 423 coated/marked geometry images. Never rename, move, recompress, rotate, overwrite, or delete them.
- `capture_v4/derived/mvs_images/`, `capture_v4/derived/masks/`, and `capture_v4/derived/feature_masks/` are the current V4 processed-image sets. They contain 372 files each and are versioned project data; regenerate them only through the V4 pipeline and never substitute pre-V4 imagery.
- Pre-V4 raw images, preprocessing outputs, analysis products, private legacy checkpoints, and their obsolete pipeline code/docs have been removed from the active repository. Do not restore or reuse them.
- `.codegraph/` is local code-intelligence state. Preserve it and keep its database/log/socket artifacts out of commits.
- Preserve unrelated `.ai-bridge/`, unrelated modified files, and user-owned work.
- Preserve current V4 sparse/dense/Poisson/Blender/GLB artifacts and hashes while repair candidates are evaluated. Do not overwrite them merely to simplify paths.

## Capture boundary and real-media facts

The authoritative V4 geometry capture is fixed-camera turntable/object-rotation photogrammetry on a white background. The vessel rotates; the background is stationary. The camera may move only between complete capture passes.

Audited geometry/empty frames share the OPPO Reno12 F rear 26mm-equivalent lens state, 3072x4080, digital zoom 1, with consistent geometry-capture exposure metadata. Use the actual sequence/manifests rather than inventing idealized ring counts or physical angles.

Because the background is stationary while the vessel rotates, background features are invalid reconstruction features. Vessel segmentation/masking is mandatory before learned matching and dense fusion. The rotating wooden board is also dangerous because leaked board pixels can form geometrically consistent false structure.

The 158 uncoated images are appearance-only project inputs. Geometry comes from coated/marked object-bearing images; final brass appearance must not be derived from the marker/dry-shampoo coloration.

## Full computer-vision-only project boundary

This project must remain a genuine end-to-end computer-vision reconstruction pipeline. Every accepted geometric and appearance result must be algorithmically derived from the captured project image set and measured reconstruction evidence.

- No external reference-assisted geometry, CAD/profile tracing, manual sculpting, manual vertex/face modeling, symmetry completion, lathe/revolve replacement, primitive replacement, hand-built neck/lid/finial/bowl/pedestal, or artist-authored shape completion is allowed.
- No manual reference matching or hand-authored texture/color fallback is allowed. Uncoated project images may contribute appearance only through automated localization, projection/blending, or reproducible dataset-derived statistics; reconstructed point/image colors may also be used.
- Blender is a technical post-processing/export stage only. Geometry cleanup, decimation/LOD, UV generation, baking, normals, AO, material construction, and export must be scripted/reproducible and must not invent or reshape vessel anatomy.
- If the reconstruction is incomplete, make only bounded evidence-backed CV repairs. When the bounded repair budget is exhausted, continue with the strongest genuine image-derived reconstruction and document the missing anatomy precisely. Do not repair missing vessel-scale anatomy manually in Blender or from a reference image.
- Documentation, demos, and the final report must present the result as a full CV pipeline and clearly distinguish automated reconstruction from non-geometric visualization/export processing.

## Confirmed repair diagnosis

The old rule “do not reopen sparse unless new evidence proves an upstream defect” has been triggered. The upstream defect is now proven.

Key evidence:

- `IMG20260912145220.jpg -> IMG20260912145223.jpg`: about 449 verified two-view inliers; two-view relative rotation about 15.9 degrees; final sparse relative rotation about 103.8 degrees; final shared 3D tracks = 0.
- `IMG20260912145254.jpg -> IMG20260912145256.jpg`: two-view about 14.7 degrees versus final about 32.2 degrees; shared tracks only 10.
- `IMG20260912145330.jpg -> IMG20260912145333.jpg`: two-view about 11.8 degrees versus final about 61.5 degrees; shared tracks only 7.
- The historical `geo_g10` virtual-camera trajectory contains an acquisition-adjacent ray jump around 82.6 degrees.
- The sparse report records repeated CHOLMOD matrix-not-positive-definite warnings during final BA.
- Historical geometric depth support collapses in `geo_g10`-`geo_g12` while photometric support remains near full silhouette.
- Historical combined true3 PatchMatch configs contain 567 reference entries for 372 unique views; 158 references were assigned more than once, so later tiles could overwrite prior maps.
- The historical raw Poisson has 644 connected components and only about 66.57% of faces in its largest component. It is not acceptable for the user's visually complete vessel requirement.

Full registration count and global reprojection mean are therefore not enough to accept sparse geometry.

## Current reconstruction route

Use one evidence-backed primary pipeline, with bounded fallback only when the preceding gate proves it necessary.

1. **Reuse verified upstream media work**: immutable source, manifests, masks, ALIKED-N16Rot feature cache, LightGlue matches/two-view evidence when hashes/contracts still match. Do not rerun ingest/segmentation/features merely because sparse needs repair.
2. **Sparse-integrity diagnostics**: compare final adjacent same-ring camera relations against verified two-view geometry, final shared-track continuity, robust trajectory smoothness, ring coverage, track/reprojection health, cross-ring support, and camera/intrinsics integrity.
3. **Sparse repair or safe replacement**: attempt bounded local repair only if it is robust and versioned; otherwise build a fresh sparse candidate from corrected verified pair/constraint evidence. `phase_01` is ordering/locality metadata, not measured physical azimuth.
4. **Sparse selection**: strict sparse integrity is preferred. After the final bounded sparse repair attempt, stop sparse churn and freeze the best defensible 372-view image-derived candidate by explicit comparison of SO(3), translation, trajectory, genuine multiview tracks, reprojection, mask projection, connectivity, and provenance. Keep any strict failures recorded as failures; downstream lineage must identify the candidate as best-defensible rather than strictly accepted.
5. **Fresh compatible dense workspace**: if sparse camera geometry changes, old true3 dense maps are diagnostic-only. Re-undistort the accepted repaired sparse model/images/masks and regenerate dense evidence.
6. **Dense source selection**: prefer strong same-ring sources; add cross-ring sources only when co-visibility/two-view/camera geometry supports them. Never force weak cross-ring edges to meet a quota.
7. **Dense tiling contract**: every registered image is a PatchMatch reference exactly once per complete candidate. Source/dependency reuse is allowed; duplicate reference writes are not.
8. **CUDA PatchMatch**: COLMAP 4.2 geometric consistency on GPU 0. Start at 2000 px with defaults; isolate filtering/source/fusion changes one factor at a time only if measured evidence requires it. No silent CPU fallback.
9. **Dense fusion/selection**: exact `<image_name>.png` masks, finite/rank/color/normal checks, depth support, projected vessel-mask coverage, cross-view/transition evidence, low board/background/webbing contamination, and at least eight hash-bound anatomy views. Prefer the strict gate winner; after bounded dense escalation, select the strongest genuine cloud by measured support/coverage/contamination/anatomy even if some strict targets remain unmet, and record those limitations.
10. **Poisson**: Poisson remains the required mesher. Prefer dominant face component >=0.95, second-largest <=0.02, no unsupported large detached component, and no major vessel-scale hole. If bounded Poisson attempts cannot satisfy every research-grade threshold, choose the strongest connected scan-derived mesh and preserve the failed thresholds/visual limitations honestly.
11. **PRE-BLENDER HANDOFF**: freeze the strongest scan-derived Poisson input, all lineage/gate reports, hashes, relevant tests, and pre-Blender docs; complete the focused pre-Blender commit/push verification; then stop the Codex/local executor.
12. **POST-BLENDER — ChatGPT + Blender MCP only**: use the selected strongest scan-derived Poisson result for scripted, reproducible, non-creative Blender cleanup, LOD0/UV/detail/AO, and appearance/material work. Preserve immutable raw evidence and do not invent anatomy.
13. **POST-BLENDER — ChatGPT + Blender MCP only**: save the repaired master, export only repaired LOD0 to GLB, perform fresh factory-empty Blender re-import, at least eight-view equivalence/anatomy checks, final artifact/hash integrity, final docs, and final focused publication.

## Acceptance requirements

These are preferred research-grade acceptance targets. Under the user's completion-first policy, a bounded stage may advance with a separately labeled **best-defensible** artifact when further evidence-backed attempts are exhausted. The gate result itself must remain unchanged and failed checks must be carried into lineage, documentation, and the final limitations report.

### Sparse

- Known `geo_g10` contradictions are gone or the old sparse model has been safely replaced.
- No catastrophic acquisition-adjacent final pose discontinuity unsupported by two-view evidence.
- Strong verified adjacent pairs do not collapse to zero/near-zero shared final tracks without an explicit reason.
- Useful ring coverage, one coherent reconstruction, healthy track/reprojection distributions, evidence-backed cross-ring connectivity.

### Dense

- Sparse lineage and dense maps are compatible.
- Exact mask resolution for every expected reference.
- One-reference-one-write provenance across tiles.
- No catastrophic high-ring depth collapse; projected vessel-mask coverage is measured.
- No dominant board/background/webbing contamination.
- At least eight views explicitly confirm bowl shell/interior, rolled rim, globe/shoulder, continuous neck, lid tiers, finial, pedestal transitions, base, and no major vessel-scale holes.

### Raw mesh

- Poisson only.
- Dominant face component >=0.95.
- Second-largest face component <=0.02.
- No unsupported large detached component or major anatomical hole.
- Historical 644-component/~0.6657-dominant raw mesh must fail the repaired gate.

### Blender/GLB

- Cleanup is scan-preserving, not reconstructive modeling.
- LOD0/UV/material provenance is verified.
- Final `.blend` opens correctly.
- Fresh GLB re-import contains exactly the intended final mesh/material state, has finite geometry/normals/UVs/textures, sane transforms/bounds, and preserves anatomy in at least eight views.

## Time and compute policy

The user wants the **best practical complete result**, not an indefinitely perfect research result. Do not run arbitrary parameter sweeps, duplicate audits, or repeated candidate variants after a bounded causal ladder has been exhausted. Stop escalating when either the strict stage target passes or the bounded attempt budget is exhausted and a strongest defensible genuine CV result can be selected.

Do not save time by falsifying gate status, reusing incompatible old dense maps, allowing duplicated PatchMatch reference provenance, or fabricating Blender geometry. It is acceptable to finish with documented reconstruction limitations when the captured evidence cannot recover some anatomy.

## CodeGraph navigation

- At the start of architecture-sensitive work, inspect code-intelligence status first.
- Use CodeGraph structural navigation when current; use direct guarded reads/search for exact literals/config/generated evidence.
- Rebuild the index only when stale and when repository maintenance is in scope.
- `.codegraph/` stays uncommitted.
- Never use Codex CLI for CodeGraph or any other task.

## Core engineering rules

- Read before editing; preserve unrelated work.
- Prefer existing architecture/dependencies and the smallest coherent root-cause fix.
- Reproduce/evidence a defect before changing the affected path.
- Preserve lens/camera/undistortion coordinate consistency.
- Never fabricate geometry, poses, metrics, screenshots, visual review results, texture provenance, or remote state.
- Historical accepted artifacts are evidence, not permission to reuse them after their input geometry becomes incompatible.
- No Codex CLI agent/AI execution.
- GLM subagents may be used only as allowed by `CLAUDE.md`; parent must independently verify their work.

## Git milestone authorization

The user explicitly authorizes focused commit and push to the configured upstream for major verified V4 repair milestones or the final verified best-defensible end-to-end result:

1. repaired sparse integrity accepted, or best-defensible sparse frozen after the bounded repair ladder;
2. repaired dense accepted, or best-defensible dense selected after bounded escalation;
3. repaired raw Poisson accepted, or strongest defensible connected scan-derived Poisson selected after bounded attempts;
4. final repaired Blender + fresh GLB re-import accepted.

For the current ownership split, milestone 3 is the **Codex/local pre-Blender completion point**. Milestone 4 belongs to the later ChatGPT + Blender MCP phase.

Before each milestone commit: inspect root/branch/status/upstream/intended diff, stage only milestone-related files, exclude secrets/private weights/temp/log/scratch/unrelated files, use a descriptive commit, push normally, and verify the remote commit. The current V4 raw source photographs and the three `capture_v4/derived/` processed-image sets are intentional repository content and must use Git LFS. Do not make noisy per-task commits, create branches/tags/releases, force-push, or rewrite history without separate authorization.

## Session startup

1. Read this file and `CLAUDE.md`.
2. Read `LESSONS.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md`.
3. Read the canonical full-repair spec and plan.
4. Inspect current workspace/Git state and preserve unrelated user-owned changes.
5. Verify CodeGraph status before changing reusable V4 source.
6. Continue from the newest verified repair checkpoint, not stale historical checkboxes.
7. If operating as the current Codex/local executor, enforce the pre-Blender hard stop above even though the repository also documents the later Blender/GLB stages.

## Completion standard

The **current Codex/local assignment** is complete when sparse/dense provenance is correct, the bounded dense ladder is exhausted or accepted, the strongest defensible Poisson is selected and hash-bound, pre-Blender verification/docs are current, and the focused pre-Blender publication is verified. The **overall V4 project** is complete only later, after ChatGPT + Blender MCP carries that frozen Poisson through scan-preserving Blender, final `.blend`/GLB, fresh GLB re-import, final verification/docs, and final focused publication. Strict research-grade gate failures may remain only when bounded evidence-backed attempts are exhausted; they must stay explicitly failed and be documented as residual limitations. Do not declare completion by falsifying a gate, reusing incompatible evidence, or fabricating missing vessel-scale geometry.