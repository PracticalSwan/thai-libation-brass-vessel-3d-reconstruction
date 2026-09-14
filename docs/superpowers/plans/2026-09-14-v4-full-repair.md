# V4 Full Repair Implementation Plan

> **Execution rule:** Continue from the newest verified repository/runtime state. Do not replay completed work mechanically. Use the canonical design at `docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`. Do not use Codex CLI.

**Goal:** Complete the V4 repair end-to-end: repair or safely replace the proven `geo_g10` sparse pose defects, regenerate compatible dense reconstruction with corrected source selection and strict provenance/quality gates, obtain a genuinely connected anatomically complete Poisson vessel, finish scan-preserving Blender/LOD0/UV/material work, export/fresh-reimport GLB, verify everything, update docs, and complete focused milestone/final commit-and-push verification.

## Global constraints

- Preserve immutable `CSX4213_Project_V4_Images/` and `IMG20260826122949/`.
- Preserve private checkpoints, `.codegraph/`, unrelated `.ai-bridge/`, unrelated modified files, and all historical V4 artifacts.
- Existing historical V4 sparse/dense/Poisson/Blender/GLB remain evidence, not current acceptance.
- This must remain a full computer-vision pipeline. Accepted geometry and appearance must be algorithmically derived from the project-captured images/reconstruction evidence; no external/reference-assisted modeling, manual sculpting/vertex-face editing, symmetry/lathe/primitive replacement, hand-built anatomy, manual reference matching, or artist-authored appearance fallback is allowed.
- Blender is limited to scripted/reproducible technical cleanup, LOD/UV/baking/material construction from measured project data, and export/verification; it may not invent or reshape vessel anatomy.
- No hand-built or symmetry/lathe replacement of missing vessel-scale geometry.
- Poisson remains the surface reconstruction route.
- CUDA PatchMatch on GPU 0 only; no silent CPU fallback.
- The corrected StereoFusion mask contract is `<image_name>.png`.
- Use versioned repair outputs and refuse overwrite of historical accepted artifacts.
- A sparse pose change invalidates dense maps unless compatibility is explicitly proven.
- One complete dense candidate must assign each registered image as a PatchMatch reference exactly once.
- Do not claim photographic texture projection without verified appearance-camera alignment.
- Exact Git staging only; no `git add .`, no force-push/history rewrite.
- Major milestone commits are authorized only after verified sparse repair, dense acceptance, raw Poisson acceptance, and final Blender+GLB acceptance.

## Completion-first override — 2026-09-14 user directive

This directive supersedes later wording that would otherwise block the entire project indefinitely on a research-grade gate. Strict gates remain preferred targets and must still be computed exactly; they are never to be relabeled as passed. After a bounded evidence-backed repair/escalation ladder has been exhausted, freeze the strongest defensible genuine CV artifact, record every remaining failed check and limitation, and continue to the next stage. The required end state is a complete full-CV sparse -> fresh dense -> Poisson -> scan-preserving Blender -> `.blend`/GLB pipeline with honest limitations, not endless candidate churn. Missing anatomy must never be invented manually, reference-assisted, symmetrized, lathed, primitive-modeled, sculpted, or texture-fabricated to improve the result.

---

## Task 0 — Recover the live boundary and protect the workspace

- [ ] Read `AGENTS.md`, `CLAUDE.md`, `LESSONS.md`, `docs/memory-bank/active-context.md`, `docs/memory-bank/progress.md`, this plan, and the canonical full-repair design.
- [ ] Inspect Git root, branch, upstream, current status, recent commits, and exact unrelated user-owned changes.
- [ ] Verify CodeGraph/index status before structural source changes.
- [ ] Inventory historical V4 sparse/dense/Poisson/canonical Blender+GLB artifacts and hashes without rewriting them.
- [ ] Confirm COLMAP 4.2 CUDA, pyCOLMAP, Python, GPU 0, and Blender 5.2 identities before compute.
- [ ] Create/confirm versioned repair namespaces under `reconstruction/v4/repair/` and `D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_*`.

**Acceptance:** historical artifacts remain byte-identical; no unrelated files are changed; live repair state is documented.

---

## Task 1 — Implement sparse-integrity diagnostics and make the historical sparse model fail

**Primary files:** create `v4_repair.py`; create `tests/test_v4_repair.py`; reuse `v4_sparse.py`, `v4_database.py`, `v4_matching.py`, existing COLMAP database and sparse model.

- [ ] Add a helper that extracts final camera relative rotation/center/ray change for acquisition-adjacent same-ring images.
- [ ] Add a helper that reads verified two-view geometry/inlier counts and recovers a calibrated relative rotation where well-conditioned.
- [ ] Add final shared-track intersection counts for each evaluated pair.
- [ ] Add robust within-ring trajectory-outlier classification based on acquisition order and median/MAD or equivalent robust statistics. Do not hardcode an assumed degree-per-frame.
- [ ] Add a sparse integrity report containing, per evaluated pair: ring, filenames, verified inlier count, recoverable two-view rotation, final sparse relative rotation, rotation disagreement, shared final 3D tracks, camera-center/ray jump, evaluability/reason.
- [ ] Add camera/intrinsic/ring-coverage/track/reprojection summary fields.
- [ ] Measure genuine COLMAP track-length/distribution evidence from actual
  track elements; explicitly reject pair-local two-view allocators and
  duplicate image observations as global-continuity proof.
- [ ] Project the complete finite sparse cloud into every registered view and
  summarize mask precision, recall, and IoU per view and per selected ring;
  do not accept majority-across-cameras contamination as a substitute.
- [ ] Validate camera-center direction and positive shared-scale consistency
  against calibrated first-to-second image translation evidence.
- [ ] Write focused tests for well-behaved synthetic/fixture pairs and fail-closed non-evaluable geometry.
- [ ] Add a regression using persisted/compact historical evidence proving the current V4 sparse model fails because of the known `geo_g10` contradictions.
- [ ] Reproduce at minimum the known evidence around `145220->145223`, `145254->145256`, and `145330->145333` before proceeding.

**Acceptance:** tests pass; the historical sparse model is explicitly classified as unsuitable for final dense regeneration despite 372/372 registration.

---

## Task 2 — Build a corrected sparse constraint graph

- [ ] Treat `phase_01` as ordered locality metadata only, not measured physical azimuth.
- [ ] Build candidate same-ring edges from strong verified acquisition-local neighbors and wider same-ring neighbors already supported by cached ALIKED/LightGlue evidence.
- [ ] Score cross-ring candidates primarily by verified inliers/geometric sanity and only secondarily by phase/order compatibility.
- [ ] Do not force weak cross-ring edges to meet a fixed count.
- [ ] Exclude or explicitly collapse redundant same-second/near-duplicate edges where they create degeneracy rather than baseline.
- [ ] Record why each retained/dropped edge was chosen, including verified inliers and any pose/phase sanity evidence.
- [ ] Ensure every intended ring is internally connected and the ring stack has enough strong cross-ring anchors to form one reconstruction.
- [ ] Add deterministic graph-construction tests, including low-confidence cross-ring rejection and stable tie-breaking.

**Acceptance:** a hash-bound corrected sparse graph report exists, with no reliance on raw phase proximity alone.

---

## Task 3 — Repair or safely replace the sparse reconstruction

Create versioned sparse candidates; never mutate `reconstruction/v4/sparse/models/0` in place.

### Candidate A: bounded local/surgical repair

- [ ] Attempt only if current pyCOLMAP/COLMAP APIs can preserve verified-good cameras while re-registering/re-optimizing proven bad `geo_g10` pose segments without contaminating the historical model.
- [ ] Use strong same-ring verified geometry and trusted neighboring ring anchors.
- [ ] Bundle-adjust the candidate.
- [ ] Run the full sparse-integrity gate.

### Candidate B: fresh corrected-graph sparse remap

Run if Candidate A is unsupported, fragile, or fails integrity.

- [ ] Reuse the existing immutable images, masks, cached ALIKED features, LightGlue correspondences, and verified two-view evidence where hashes/contracts still match.
- [ ] Build a fresh versioned COLMAP database/constraint set from the corrected graph rather than regenerating source media.
- [ ] Run geometric verification as required by the candidate graph.
- [ ] Run pyCOLMAP incremental mapping + bundle adjustment with the real shared-camera contract.
- [ ] Preserve all candidate logs/reports needed to diagnose rejection; keep bulky transient DB/workspace data out of publication unless required.
- [ ] Audit every ring, with special attention to `geo_g10` and ring transitions.

### Sparse promotion

- [ ] Select only a candidate that resolves the known `geo_g10` contradictions and does not create new catastrophic ring discontinuities.
- [ ] Require useful intended ring coverage, exactly one coherent model with all
  intended views, genuine multi-view track distributions, per-view/per-ring
  projected-mask precision/recall/IoU, calibrated center/scale health, sane
  reprojection distributions, no bad camera islands, and evidence-backed
  cross-ring connectivity.
- [ ] Bind the accepted candidate to a stable model-directory hash and the
  immutable canonical SQLite snapshot plus disposable working-copy lineage;
  never open the canonical snapshot directly with SQLite or pyCOLMAP.
- [ ] Render/inspect sparse points + camera trajectory from multiple views.
- [ ] Write `reconstruction/v4/reports/repair_sparse_gate.json` and hash-bind the accepted candidate.
- [ ] At first verified sparse acceptance, inspect Git status/diff, stage only sparse-repair code/tests/compact evidence/docs, commit, push, and verify `origin/main` synchronization.

**Acceptance:** accepted repaired sparse candidate passes all sparse integrity checks; historical sparse remains preserved.

---

## Task 4 — Invalidate incompatible historical dense maps and prepare a clean repaired dense workspace

- [ ] Compare accepted sparse camera/intrinsic/coordinate hashes against historical true3 inputs.
- [ ] If any extrinsic/intrinsic/undistortion/scale contract changed, mark historical true3 photometric/geometric maps diagnostic-only and ineligible for final promotion.
- [ ] Re-undistort the accepted repaired sparse model, images, and vessel masks consistently into a fresh versioned dense workspace.
- [ ] Resolve all expected fusion masks under exact `<image_name>.png` names.
- [ ] Build repaired PatchMatch source lists from accepted sparse evidence.
- [ ] Enforce approximately six strong same-ring sources where available and at most two supported adjacent-ring sources.
- [ ] Add a dense-config validator that rejects duplicate reference assignment across tiles and proves every registered reference appears exactly once.
- [ ] Add regression evidence showing the historical combined true3 configs had 567 reference entries / 372 unique / 158 duplicated references, and ensure the repaired generator cannot reproduce that failure.

**Acceptance:** fresh repaired dense workspace has compatible sparse geometry, 372/372 masks if all 372 remain registered, and one-reference-one-write provenance.

---

## Task 5 — Run default repaired 2000px geometric PatchMatch

- [ ] Run a bounded CUDA smoke on GPU 0 using the repaired workspace/source graph.
- [ ] Abort on CUDA OOM/runtime corruption rather than falling back to CPU.
- [ ] Run the full repaired photometric preparation/geometric PatchMatch sequence needed by COLMAP 4.2 with geometric consistency and default filtering.
- [ ] Capture exact commands/config hashes/logs.
- [ ] Compute per-image/per-ring valid depth support inside masks for photometric and geometric maps.
- [ ] Compute vertical-band support and transition/cross-view diagnostics.
- [ ] Do not fuse if maps contain clear wrong-depth anatomy or catastrophic high-ring support loss.

**Quality target:** geometric per-ring median >= 0.75 and p10 >= 0.40 where achievable, with no visible wrong geometry. This is a target, not permission to accept invalid maps.

---

## Task 6 — Bounded dense escalation only where evidence requires it

If Task 5 passes map quality, skip directly to fusion.

### Filter isolation

- [ ] Candidate `filteroff`: same repaired sparse/source graph, geometric consistency on, `PatchMatchStereo.filter=0`.
- [ ] Candidate `minconsistent1`: reset from preserved parent, filtering on, `filter_min_num_consistent=1`.
- [ ] Change one major factor at a time and compare depth support/transition evidence before fusion.

### Source refinement

- [ ] If weak source evidence remains, revise only the affected source lists using accepted sparse co-visibility/geometric support.
- [ ] Do not start an unconstrained parameter sweep.

### High resolution last

- [ ] If 2000px candidates remain incomplete specifically in thin upper geometry, run a 3072px CUDA smoke on a small high-ring subset.
- [ ] Only if memory/runtime are safe, run a consistent high-resolution high-ring candidate in a separate workspace.
- [ ] Never mix map dimensions blindly inside one COLMAP workspace.

**Acceptance:** at least one map candidate is quantitatively and visually suitable for fusion, or the next bounded escalation is justified by explicit evidence.

---

## Task 7 — Fuse and select the repaired dense candidate

- [ ] Start with COLMAP 4.2 geometric stereo fusion defaults and corrected masks.
- [ ] Compute finite/rank/point count/colors/normals.
- [ ] Measure board/background/webbing contamination using the existing corrected methodology.
- [ ] Measure projected vessel-mask coverage through stratified registered cameras from every ring.
- [ ] Record corrected ring-transition/cross-view evidence.
- [ ] Render at least eight hash-bound views: front, rear, left, right, four quarter views, plus top/bottom or interior views when needed.
- [ ] Explicitly review: bowl shell/interior, rolled rim, globe/shoulder, continuous neck, lid tiers, finial, pedestal transitions, base, and no major vessel-scale holes.

If maps are good but fusion coverage is still weak, change one fusion parameter at a time:

- [ ] `StereoFusion.min_num_pixels: 5 -> 3`.
- [ ] If still needed, `max_depth_error: 0.01 -> 0.02` on the best parent.
- [ ] If still needed, `max_reproj_error: 2 -> 3` on the best parent.

Reject candidates that gain points by adding board/background, doubled anatomy, or incoherent layers.

- [ ] Write `repair_dense_gate.json`, candidate lineage/config/hash report, and compact accepted previews.
- [ ] At verified dense acceptance, make the focused dense milestone commit/push and verify the remote commit.

**Acceptance:** one repaired dense cloud passes compatibility, depth, projected coverage, consistency, contamination, one-reference provenance, and explicit anatomical completeness gates.

---

## Task 8 — Generate and accept repaired raw Poisson

- [ ] Run Poisson depth 13 first on the accepted repaired dense cloud into a versioned repair path; never overwrite historical `poisson_raw.ply`.
- [ ] Record exact dense input SHA and raw mesh SHA.
- [ ] Compute finite vertices/faces, connected-component face counts, dominant and second-largest face fractions, bounds and rank.
- [ ] Hard gate: dominant face component >= 0.95.
- [ ] Hard gate: second-largest face component <= 0.02.
- [ ] Reject unsupported large detached components.
- [ ] Render and inspect at least eight raw-mesh views using the same explicit anatomy fields as dense acceptance.
- [ ] Require `no_major_vessel_scale_holes=true`.
- [ ] Keep the historical 644-component / ~0.6657-dominant raw Poisson as a regression that must fail the repaired gate.
- [ ] If raw Poisson misses strict thresholds, perform only the bounded evidence-backed retry justified by the measured cause; once that ladder is exhausted, select the strongest connected scan-derived Poisson, keep the failed thresholds explicit, and continue to Blender without fabricating missing anatomy.
- [ ] At raw Poisson acceptance, make the focused raw-mesh milestone commit/push if the artifact is publication-appropriate and verify remote/LFS state as applicable.

**Acceptance:** genuinely connected, anatomically complete reconstruction-derived raw vessel.

---

## Task 9 — Build scan-preserving Blender master and LOD0

Use connected Blender MCP and/or absolute Blender 5.2 executable. Read the Blender project skills/instructions before mutation. Every geometry operation in this stage must be scripted/reproducible and non-creative.

- [ ] Import accepted repaired raw Poisson into a versioned repaired `.blend` and preserve an immutable raw object/collection + rollback copy.
- [ ] Audit components, boundary/non-manifold edges, loose/degenerate geometry, normals, bounds/transforms and material/UV state.
- [ ] Duplicate to clean-high through a scripted/reproducible procedure.
- [ ] Remove only algorithmically identified isolated noise/floaters, impossible internal debris, duplicates/degenerates, bad normals, and tiny defensible defects.
- [ ] Do not manually select/edit vertices or faces to repair shape, sculpt, globally remesh, symmetrize, lathe/revolve, primitive-replace, trace profiles, or otherwise create missing vessel anatomy.
- [ ] If vessel-scale anatomy is missing, do not repair it in Blender. Only return to a CV stage when one bounded evidence-backed retry remains justified; otherwise continue with the strongest genuine reconstruction and document the missing anatomy as a final limitation.
- [ ] Render at least eight clean-high views and compare to accepted raw geometry; reject material silhouette/anatomy loss.
- [ ] Create repaired LOD0 with sufficient topology to preserve silhouette and anatomy using reproducible technical processing.
- [ ] UV unwrap LOD0 through a reproducible procedure.
- [ ] Bake scan-supported normal/detail and AO.
- [ ] Record raw -> clean-high -> LOD0 counts/bounds/component comparisons and the exact scripted operations used.

**Acceptance:** scan-derived clean-high and LOD0 remain anatomically equivalent to accepted raw reconstruction, with no manually/reference-assisted constructed geometry.

---

## Task 10 — Build automated project-image-derived brass appearance

- [ ] Implement/verify fail-closed filtering for the 158 uncoated project appearance images.
- [ ] Attempt automated appearance-camera localization/alignment against accepted repaired geometry without changing geometry.
- [ ] Measure alignment/reprojection/silhouette validity for each candidate appearance camera.
- [ ] If enough views verify, automatically project/blend BaseColor while rejecting background/cloth/wood, markers/coating, occlusion, grazing angles and clipped specular pixels.
- [ ] Produce Roughness from project-image evidence when defensible; use scan-baked Normal/detail and AO; metallic response appropriate to brass.
- [ ] If photographic projection cannot be verified, set `photographic_projection_verified=false` and derive appearance reproducibly from the eligible project-image set and/or reconstructed point/image colors; do not hand-pick a reference image or hand-tune color/texture to a reference.
- [ ] Render neutral lookdev from multiple lighting/view setups and ensure the result does not reproduce black marker/dry-shampoo coloration as final appearance.

**Acceptance:** material provenance is explicit, automated, project-data-derived and verifiable; no manual reference matching, artist-authored fallback, or fabricated photo-texture claim.

---

## Task 11 — Final Blender gate, GLB export and fresh re-import

- [ ] Strengthen final Blender gate to require accepted repaired raw hash, clean-high/LOD0 continuity, UVs, material provenance, finite/sane transforms and explicit anatomical completeness.
- [ ] Save versioned repaired master first.
- [ ] Export only repaired LOD0 to versioned GLB; exclude raw/debug objects, cameras, lights and unrelated data.
- [ ] Start fresh factory-empty Blender 5.2 and import the GLB.
- [ ] Verify exactly one intended final mesh, finite positions/normals, UVs, material/textures, sane bounds/transforms and no hidden debug export.
- [ ] Render at least eight fresh-reimport views and compare against authoring views.
- [ ] Reconfirm bowl/rim/interior, globe/shoulder, continuous neck, lid tiers, finial, pedestal transitions and base after export/reimport.
- [ ] Preserve/hash previous canonical `Thai_Libation_Vessel_V4_FINAL.blend` and `.glb`, then promote repaired outputs to canonical names only after the repaired final gate passes.

**Acceptance:** fresh GLB re-import is materially equivalent to the accepted Blender authoring model and remains anatomically complete.

---

## Task 12 — Full verification, documentation and final publication

- [ ] Run all relevant V4 repair tests, including sparse integrity, graph selection, dense provenance, depth/coverage/contamination, mesh continuity, appearance provenance and Blender/GLB gate tests.
- [ ] At minimum run `py -3 -m pytest -q tests/test_v4_*.py tests/test_v4_repair.py -p no:cacheprovider` plus `tests/test_v4_appearance.py` if created.
- [ ] Run directly affected shared-module tests if reusable helpers were changed.
- [ ] Run `py -3 -m compileall -q v4_*.py scripts` for touched V4 Python paths as appropriate.
- [ ] Recompute SHA-256 for accepted repaired sparse report/model evidence, dense cloud, raw Poisson, repaired `.blend`, `.glb`, and textures.
- [ ] Verify every report points to the exact current artifact bytes.
- [ ] Remove only task-created temporary residue that is neither accepted evidence nor needed for reproducibility; do not delete failed candidate reports needed for diagnosis.
- [ ] Update `docs/memory-bank/active-context.md`, `docs/memory-bank/progress.md`, `LESSONS.md`, and any affected project documentation with measured final state.
- [ ] Inspect final Git status/diff; stage only intended repair code/tests/docs/compact evidence/artifacts.
- [ ] Create the final focused commit and push.
- [ ] Verify `origin/main` equals the final local commit and verify Git LFS objects where applicable.

## Final completion report must include

- accepted repaired sparse candidate lineage and exact evidence that the `geo_g10` contradictions are gone;
- final registered counts, sparse points/observations/track/reprojection and trajectory integrity summary;
- accepted dense workspace/config lineage, one-reference-one-write proof, SHA, point count, ring depth table, projected coverage, transition evidence and contamination fractions;
- accepted raw Poisson SHA, vertices/faces, component count, dominant/second component fractions;
- eight-view dense/raw/final evidence;
- clean-high/LOD0 counts/bounds preservation comparison;
- appearance-camera result and whether photographic projection was actually verified;
- final texture list/provenance;
- final `.blend`/`.glb` hashes and fresh-reimport metrics;
- exact verification commands/results;
- final Git commit and verified remote state;
- any residual limitation stated precisely.

## Completion rule

Do not stop before producing the complete end-to-end 3D deliverable. Use the next bounded evidence-backed repair level only while it has a realistic causal path to improvement; once a stage's bounded ladder is exhausted, freeze the strongest defensible genuine CV artifact and continue through dense, Poisson, Blender and fresh GLB. Completion may include explicitly failed research-grade checks and documented missing anatomy, but never by relabeling failures as passes, accepting incompatible historical dense maps, or fabricating vessel-scale geometry.
