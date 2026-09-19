# V4 Full Repair Implementation Plan

> **Execution rule:** Continue from the newest verified repository/runtime state. Do not replay completed work mechanically. Use the canonical design at `docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`. Do not use Codex CLI. As of 2026-09-15, the current Codex/local executor has a hard stop at the verified Poisson/pre-Blender handoff; ChatGPT + Blender MCP owns Blender and everything after Blender.

**Goal:** Complete V4 in two controlled ownership phases. The current Codex/local phase must finish every remaining task through the strongest verified, versioned, hash-bound scan-derived Poisson mesh, including post-restart recovery of any interrupted dense work, dense/Poisson selection, lineage, tests/hashes/docs and the focused pre-Blender Git milestone. It must then stop. The later ChatGPT + Blender MCP phase will perform scan-preserving Blender/LOD0/UV/baking/material work, final `.blend`/GLB, fresh re-import, final verification/docs and final publication.

## Global constraints

- Preserve immutable `CSX4213_Project_V4_Images/` as the only raw source set and preserve the current processed image trees under `capture_v4/derived/`; both are intentional Git LFS publication content.
- Preserve `.codegraph/`, unrelated `.ai-bridge/`, unrelated user-owned work, and the current V4 reconstruction artifacts/evidence. Superseded pre-V4 checkpoints and media are not part of the active repository.
- The verified V4 sparse/dense/Poisson/Blender/GLB baseline remains rollback/evidence while V7 refinement is evaluated; do not overwrite it during candidate work.
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

## Execution ownership override — 2026-09-15 user directive

This override is authoritative for current execution:

- **Codex/local executor:** recover the newest state after restart; inspect persisted 3072px `geo_g12` logs/workspaces/reports before rerunning; complete or resume that bounded recovery only if evidence shows it is incomplete; freeze the strongest genuine dense candidate; finish the bounded Poisson ladder; freeze/hash-bind the strongest connected scan-derived Poisson; verify pre-Blender lineage/tests/compile/hashes/docs; clean only pre-Blender residue; perform the focused pre-Blender commit/push and verify `origin/main`.
- **Codex/local hard stop:** after the verified Poisson + evidence package is ready, do not open, modify, regenerate, clean, save, export, or otherwise touch Blender scenes, `.blend`/GLB artifacts, Blender-specific finalization, appearance/material finalization, fresh GLB re-import, or post-Blender publication.
- **ChatGPT + Blender MCP:** owns Tasks 9-12 and all final Blender/GLB work after the handoff.
- Existing Blender/GLB candidates are historical/diagnostic evidence until the post-Blender owner revisits them.

## Completion-first override — 2026-09-14 user directive

This directive supersedes later wording that would otherwise block the entire project indefinitely on a research-grade gate. Strict gates remain preferred targets and must still be computed exactly; they are never to be relabeled as passed. After a bounded evidence-backed repair/escalation ladder has been exhausted, freeze the strongest defensible genuine CV artifact, record every remaining failed check and limitation, and continue only within the current executor's ownership boundary. The required overall end state remains a complete full-CV sparse -> fresh dense -> Poisson -> scan-preserving Blender -> `.blend`/GLB pipeline with honest limitations, but the current Codex/local executor stops at the verified Poisson handoff and ChatGPT + Blender MCP owns the remaining stages. Missing anatomy must never be invented manually, reference-assisted, symmetrized, lathed, primitive-modeled, sculpted, or texture-fabricated to improve the result.

---

## Task 0 — Recover the live boundary and protect the workspace

- [ ] Read `AGENTS.md`, `CLAUDE.md`, `LESSONS.md`, `docs/memory-bank/active-context.md`, `docs/memory-bank/progress.md`, this plan, and the canonical full-repair design.
- [ ] Inspect Git root, branch, upstream, current status, recent commits, and exact unrelated user-owned changes.
- [ ] Verify CodeGraph/index status before structural source changes.
- [ ] Inventory historical V4 sparse/dense/Poisson/canonical Blender+GLB artifacts and hashes without rewriting them.
- [ ] Confirm COLMAP 4.2 CUDA, pyCOLMAP, Python, and GPU 0 identities before pre-Blender compute. Blender 5.2 identity is deferred to the post-Blender owner.
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
- [ ] If raw Poisson misses strict thresholds, perform only the bounded evidence-backed retry justified by the measured cause; once that ladder is exhausted, select the strongest connected scan-derived Poisson, keep the failed thresholds explicit, and freeze it for the pre-Blender handoff. Do not enter Blender or fabricate missing anatomy.
- [ ] At raw Poisson acceptance, make the focused raw-mesh milestone commit/push if the artifact is publication-appropriate and verify remote/LFS state as applicable.

**Acceptance:** genuinely connected, anatomically complete reconstruction-derived raw vessel.

---

## Task 8.5 — Freeze and hand off the complete pre-Blender state — COMPLETE (2026-09-15)

- [x] After any host restart, inspect persisted high-ring logs/workspaces/reports first. Do not assume a previously launched CUDA/COLMAP process completed and do not start a duplicate run unless the persisted state proves the prior attempt incomplete or invalid.
- [x] If the bounded 3072px `geo_g12` recovery is incomplete, resume or rerun only the missing/invalid portion in its separate workspace; if it completed, audit the persisted result instead of replaying it.
- [x] Compare the 3072px result against the current best-defensible dense candidate using measured support/coverage/contamination/anatomy evidence. Promote only if objectively stronger; otherwise document recovery exhaustion and preserve the existing winner.
- [x] Complete the bounded Poisson ladder from the selected dense source and freeze the strongest connected scan-derived Poisson with every failed anatomy/continuity gate preserved exactly.
- [x] Verify the sparse V2 lineage, dense selection, Poisson input/output hashes, reports, and artifact paths all point to the exact current bytes.
- [x] Run the relevant pre-Blender V4 tests, directly affected shared tests, compile checks, and `git diff --check`/equivalent whitespace validation.
- [x] Remove only task-created pre-Blender residue that is neither required evidence nor needed for reproducibility. Do not alter Blender/GLB directories.
- [x] Update the pre-Blender instructions/current-status docs and any directly affected reports with measured state.
- [x] Inspect Git root/branch/status/upstream/intended diff; stage only intended pre-Blender source/tests/docs/compact evidence/artifacts; create/push the focused pre-Blender milestone commit when appropriate; verify `origin/main` equals the intended commit.
- [x] Produce a concise handoff containing the selected sparse/dense/Poisson paths and SHA-256 values, remaining strict failures, exact verification results, and any work intentionally deferred to Blender.
- [x] **STOP. Do not proceed to Task 9.**

Measured handoff artifacts: `dense_best_defensible_v2_with_upper_recovery.json`
(SHA-256 `d6fad39ac7f30476eb89a2a4b2b49acb4b2eb2aa5549178bfcc4d900ed76fcf4`)
and `raw_poisson_best_defensible_v2_handoff.json` (SHA-256
`4e1251e216b34783b87ff1e58664df8c0f306e44383431e4c9644443f8a9655a`). The
selected trim-5 Poisson mesh is hash-bound as
`33fe1f6e7f696d8fd46d4a7c324d0e4f5fd9ff6a2ad4fb08e9f851a0437d6941`. The
handoff is best-defensible, not strict: dense/anatomy failures and the
detached-component semantic-proof limitation remain visible. Blender/GLB
work is owned by the post-Blender executor.

**Acceptance:** a verified, versioned, hash-bound Poisson input plus complete evidence package is ready for ChatGPT + Blender MCP, with the pre-Blender Git state verified and no Blender/GLB mutation performed.

---

## Task 9 — POST-BLENDER OWNER: ChatGPT + Blender MCP — Build scan-preserving Blender master and LOD0

This task is **out of scope for the current Codex/local executor**. ChatGPT + Blender MCP will start it only after Task 8.5 is accepted. Use connected Blender MCP and/or absolute Blender 5.2 executable. Read the Blender project skills/instructions before mutation. Every geometry operation in this stage must be scripted/reproducible and non-creative.

- [x] Start from the versioned authoring scene already hash-bound to the accepted trim-5 Poisson and preserve the exact raw source object/collection plus rollback history.
- [x] Re-audit final Raw/CleanHigh/LOD0 counts, finite geometry, bounds/transforms, visibility, material and UV state; component continuity remains bound to the frozen Poisson handoff evidence.
- [x] Preserve the existing reproducibly generated scan-derived CleanHigh rather than rebuilding or manually editing it.
- [x] Preserve the prior technical-only duplicate/invalid-polygon cleanup; no new anatomical cleanup or manual mesh surgery was introduced.
- [x] Do not manually select/edit vertices or faces to repair shape, sculpt, globally remesh, symmetrize, lathe/revolve, primitive-replace, trace profiles, or otherwise create missing vessel anatomy.
- [x] Keep unrecovered upper anatomy as a documented CV limitation rather than repairing it in Blender; the bounded high-ring recovery was already exhausted.
- [x] Render eight deterministic final authoring views and compare them against the fresh GLB re-import views; no material silhouette/export loss was found.
- [x] Preserve the verified scan-derived LOD0 at 151,547 vertices / 294,713 faces with unchanged mesh geometry/transforms.
- [x] Replace the unusably fragmented inherited LOD0 UVs with a deterministic signed-dominant-normal six-way geometry-derived atlas; vertices/faces/transforms remain unchanged.
- [x] Bake scan-supported 2048px CleanHigh→LOD0 tangent normal/detail and 2048px LOD0 AO.
- [x] Record raw -> clean-high -> LOD0 counts/bounds, UV before/after metrics, bake settings, visibility and exact scripted operations in the v6 final reports.

**Acceptance:** completed under the best-defensible policy. Scan-derived Raw/CleanHigh/LOD0 remain geometry-equivalent to the accepted reconstruction lineage and no manually/reference-assisted geometry was constructed; known upper-anatomy failures remain explicit.

---

## Task 10 — POST-BLENDER OWNER: ChatGPT + Blender MCP — Build automated project-image-derived brass appearance

- [x] Use the verified fail-closed 158-image uncoated appearance-statistics pipeline and bind the final material to its report/source-manifest hashes.
- [x] Treat automated appearance-camera localization/projection as unverified for this final; no camera-alignment result is claimed or substituted with manual matching.
- [x] Because no verified appearance-camera solution exists, do not claim reprojection/silhouette alignment metrics or photographic texture projection.
- [x] Skip photo projection rather than accepting unverified views; no hand-picked reference, background/cloth/wood projection, coated marker coloration, or fabricated photo-texture claim is used.
- [x] Produce Roughness from the uncoated project-image luminance statistics; use scan-baked Normal/detail and AO plus a fixed brass-conductor metallic prior.
- [x] Set `photographic_projection_verified=false` and derive BaseColor/Roughness reproducibly from the complete eligible 158-image set; no hand tuning/reference matching is used.
- [x] Render and inspect eight authoring plus eight fresh-reimport lookdev views; final appearance does not reproduce the coated black-marker/dry-shampoo vertex colors.

**Acceptance:** material provenance is explicit, automated, project-data-derived and verifiable; no manual reference matching, artist-authored fallback, or fabricated photo-texture claim.

---

## Task 11 — POST-BLENDER OWNER: ChatGPT + Blender MCP — Final Blender gate, GLB export and fresh re-import

- [x] Strengthen the final Blender/GLB gate to require the frozen Poisson lineage, preserved Raw/CleanHigh/LOD0 state, usable UV area, project-derived material provenance, finite/sane transforms, required texture connections and explicit anatomy findings.
- [x] Save and hash the versioned v6 repaired master first.
- [x] Export only repaired LOD0 to the versioned GLB; raw/source/debug objects, cameras and lights are excluded and vertex colors are disabled.
- [x] Start fresh factory-empty Blender 5.2 and import the GLB.
- [x] Verify exactly one intended final mesh, finite positions/normals, UVs, BaseColor/Roughness/Normal/AO, sane bounds/transforms and no hidden nonmesh/debug export.
- [x] Render eight fresh-reimport views plus eight authoring views and compare them; mean 8-bit MAE is ~0.000637 and minimum PSNR is ~78.56 dB.
- [x] Reconfirm anatomy after export/reimport. Lower/body/pedestal remains stable, but upper neck/lid/finial defects remain visible; the strict anatomy failure is preserved rather than relabeled.
- [x] Preserve/hash the previous canonical `Thai_Libation_Vessel_V4_FINAL.blend` and `.glb`, record rollback commit `625168138471dce0c8f01eda284dab7acd9be55f`, then promote exact verified v6 bytes to the canonical names.

**Acceptance:** fresh GLB re-import is materially equivalent to the verified v6 authoring model and passes the technical final-asset gate. Strict anatomical completeness is not claimed; completion is best-defensible with the frozen upper-anatomy limitations carried forward unchanged.

---

## Task 12 — POST-BLENDER OWNER: ChatGPT + Blender MCP — Full verification, documentation and final publication

- [x] Run all relevant V4 repair tests, including sparse integrity, graph selection, dense provenance, depth/coverage/contamination, mesh continuity, appearance provenance and Blender/GLB gate tests.
- [x] Run `py -3 -m pytest -q tests/test_v4_*.py tests/test_v4_repair.py -p no:cacheprovider`: 106 passed in the final verification run; no separate `tests/test_v4_appearance.py` exists.
- [x] No additional shared-module regression suite was required by the post-Blender changes; verification stayed scoped to V4 plus the final Blender/GLB gates.
- [x] Compile the touched finalizer/verifier Python paths successfully with `py -3 -m compileall -q`.
- [x] Recompute SHA-256 for accepted repaired sparse report/model evidence, dense cloud, raw Poisson, repaired `.blend`, `.glb`, textures and final reports in `final_hash_manifest.json`.
- [x] Verify every final report points to the exact current artifact bytes, including refreshed eight-view render hashes.
- [x] Remove only task-created temporary Blender audit/probe/contact-sheet residue; preserve failed reconstruction candidates and historical diagnostic evidence.
- [x] Update `docs/memory-bank/active-context.md`, `docs/memory-bank/progress.md`, `LESSONS.md`, `README.md`, and the canonical V4 repair plan with measured final state.
- [x] Inspect final Git status/diff and stage only intended finalization code/docs/curated evidence/artifacts; unrelated user-owned changes remain untouched.
- [x] Create and push the focused final Blender/GLB milestone commit; Git LFS uploaded the 344 MB final Blender object successfully.
- [x] Fetch and verify `origin/main` against the pushed local commit and run `git lfs fsck` successfully.

## V7 scan-preserving refinement addendum — 2026-09-17

The V6 export remains historical provenance in Git history, but its superseded standalone authoring/export files are removed from the active finished tree. The V7 scan-preserving V139 refinement is the completed canonical release under the user's explicit repair and final export authorization.

- [x] Reject the V113/V114 replacement-looking geometry/material direction.
- [x] Keep V94 as the dense-derived geometry base after V96 measured worse on the 3 cm ghost metric.
- [x] Remove exactly the identified 70-face synthetic planar bottom cap and preserve one physical 72-edge bottom opening.
- [x] Keep the bottom plane level while reducing measured centerline lean through the V119 scan-preserving shear.
- [x] Use only guarded same-object CleanHigh lower-band donor displacement; protect the physical opening and reject large donor jumps.
- [x] Clean and patch the terminal only through existing reconstructed vertices and same-scan evidence; do not add a sphere primitive or rebuilt finial.
- [x] Continue bounded all-visible-surface cleanup through V139, with segmented eight-angle audits across holder, bowl, globe, tower, upper ring and terminal, while locking the physical bottom boundary.
- [x] Verify V139 topology: 451,312 vertices / 902,838 triangular faces, one connected component, one 72-edge boundary loop, zero other non-manifold edges, zero zero-area faces, no loose vertices, finite positions and identity transforms.
- [x] Review final eight-angle clay surface bands, full turntable, top-oblique/underside views, and the final brass turntable/closeups.
- [x] Save and hash `Thai_Libation_Vessel_V4_V7_SCAN_PRESERVING_FINAL_V139.blend` (604,340,108 bytes; SHA-256 `951d11585d3b459251bb349acc5515877f4f223fb55189e37d5910f1716ee4f3`) and verify no missing external files.
- [x] Keep `MAT_V4_V115_BrassStatsOnly`; no generic decoration or V6 topology-specific maps are attached to the V7 mesh.
- [x] Promote V139 to `Thai_Libation_Vessel_V4_FINAL.blend` and export `Thai_Libation_Vessel_V4_FINAL.glb` after explicit user authorization.
- [x] Fresh-import the canonical GLB in Blender 5.2 factory-startup and verify exactly one mesh with matching 451,312 vertices / 902,838 triangles, material, transforms and dimensions.
- [x] Hash the canonical final files: current `.blend` `5eacf012b9940d1997291c4c1e0720776e2a658bf2c60b3a7e567947695e3436`; GLB `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`. The canonical `.blend` was re-saved after promotion; its semantic V139 mesh SHA-256 `82348c02e61529d1f19d6e0a11e2c3fcf3228a354d1186c73691d84beb5a393b` matches the versioned V139 master, while the previous canonical byte hash `6663d83303fefac34bbbee85132ad64322e2e9b0bc4db73776f2a882a9d07a0c` remains in Git history.

### Final professor submission package — 2026-09-19

- [x] Create root-level `submission/` and copy the verified canonical GLB to `Thai_Libation_Vessel_Final.glb`.
- [x] Export `Thai_Libation_Vessel_Final.ply` from `SM_V4_V139_SCAN_PRESERVING_FINAL` as binary triangulated geometry with normals.
- [x] Fresh-import the submission PLY and verify one mesh, 451,312 vertices / 902,838 triangles, identity transforms, and matching dimensions.
- [x] Fresh-import the submission GLB and verify one mesh, 451,312 vertices / 902,838 triangles, one `MAT_V4_V115_BrassStatsOnly` material, identity transforms, and matching dimensions.
- [x] Preserve the user-approved ready image set: eight turntable views plus one turntable contact sheet, with `README.txt` and regenerated `SHA256SUMS.txt`.
- [x] Record final submission hashes: GLB `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`; PLY `42ab10ebec3b69dc8dc7ce28463a9d04fd3f0ecb3b35f888b62bc3c222eb9ff1`.

### Finished-project cleanup — 2026-09-20

- [x] Preserve the ready submission, final DOCX/PDF report, immutable source/derived image sets, selected sparse/dense/Poisson lineage, scripts/tests, V139 authoring master/final QA, canonical `.blend`/GLB, and setup photographs.
- [x] Remove superseded V6 authoring/export assets, the standalone V128 checkpoint, pre-V139 V7 diagnostic/repair renders, historical V6 comparison renders and V6-only verification reports, transient `reconstruction/v4/work/` logs, old interim authoring/AO artifacts, and the duplicate `.blend1` backup.
- [x] Regenerate the submission checksum manifest for the ready contents and update current-state documentation/instructions to the finished-project boundary.
- [x] Verify final report/package integrity, ready-submission checksums, protected asset hashes/topology, full test suite, staged diff hygiene, and Git LFS integrity.
- [ ] Commit the verified cleanup, push `main`, and prove `HEAD == origin/main == GitHub main` with no pending Git/LFS changes.

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

For the finished project, no further reconstruction, candidate generation, Blender geometry/material work, or export changes are in scope unless the user explicitly reopens them. Completion now means the protected final V139/canonical deliverables and user-approved submission remain intact and independently verifiable, current documentation matches the active tree, obsolete residue is removed without sacrificing required evidence, relevant tests/integrity checks pass, and the final commit is synchronized to GitHub `main` with no pending Git/LFS changes.
