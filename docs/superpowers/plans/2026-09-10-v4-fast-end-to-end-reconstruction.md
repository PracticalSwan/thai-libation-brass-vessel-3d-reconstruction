# V4 Fast End-to-End Reconstruction Implementation Plan

> **Execution state:** historical/superseded for new execution as of 2026-09-14. Preserve this plan as prior requirements/context only. The active implementation plan is `docs/superpowers/plans/2026-09-14-v4-full-repair.md`; later investigation proved upstream `geo_g10` sparse-pose inconsistencies and duplicate dense reference writes, so the former dense/post-fusion continuation checkpoint below is not the current starting point. As of 2026-09-15, the current Codex/local executor stops at the verified Poisson pre-Blender handoff; ChatGPT + Blender MCP owns Blender and everything after Blender. Do not use Codex CLI.

**Goal:** Convert the immutable final V4 still-image set into one scan-derived Thai libation vessel, verify the real reconstruction before manual cleanup, and deliver a canonical editable Blender file plus a cleanly re-importable GLB.

**Spec:** `docs/superpowers/specs/2026-09-10-v4-fast-end-to-end-reconstruction-design.md`

**Canonical source:** `CSX4213_Project_V4_Images/`

**Canonical final outputs:**

```text
reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend
reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb
```

## 0. Fixed evidence and execution rules

The final media cannot be retaken. Do not ask for recapture and do not turn acquisition imperfections into blockers when they can be mitigated in software.

The planning audit accounted for every supplied image:

```text
688 JPEG files total
158 uncoated appearance/reference
107 empty-board/background
423 coated/marked object-bearing geometry
3072 x 4080 for all files
OPPO Reno12 F rear 26mm-equivalent camera for all files
focal length 3.98 mm, f/1.8, digital zoom 1 for all files
geometry/empty captures: ISO 100, 1/100 s, fixed reported lens state, manual exposure/WB metadata
```

The six object-bearing source-pass ranges are authoritative starting evidence:

```text
geo_g7   IMG20260912141912.jpg -> IMG20260912142404.jpg      142 files
geo_g8   IMG20260912143336.jpg -> IMG20260912143718.jpg       72 files
geo_g9   IMG20260912144413.jpg -> IMG20260912144629.jpg       55 files
geo_g10  IMG20260912145120.jpg -> IMG20260912145347.jpg       59 files
geo_g11  IMG20260912150136.jpg -> IMG20260912150348.jpg       58 files
geo_g12  IMG20260912150514.jpg -> IMG20260912150627_01.jpg    37 files
```

The known empty-board ranges are:

```text
empty_g6   IMG20260912133528.jpg -> IMG20260912133652.jpg      51 files
empty_g8   IMG20260912143726.jpg -> IMG20260912143825.jpg      31 files
empty_g9   IMG20260912144636.jpg -> IMG20260912144719.jpg      25 files
```

Everything from 13:18:08 through 13:33:17 is uncoated appearance/reference evidence, totaling 158 images. Some close/detail reference images contain tripod/green-sheet intrusion or clipped highlights; they are references, not geometry.

Do not assume `geo_g7` is one 142-view revolution. Its count and visual recurrence indicate repeated phase coverage. Detect revolution wrap(s) before assigning `phase_01`. Do not blindly split exactly in half.

There are no byte-identical SHA-256 duplicates, but same-second `_01` pairs include near-duplicates. Never delete originals. Mark redundant selected views in the manifest and keep provenance.

The white cloth is stationary and textured. The wooden board rotates with the vessel and is highly feature-rich. **Board leakage is the dominant reconstruction-contamination risk.** Every sparse/dense gate must explicitly prove that the vessel, not the board, supports the result.

### Global implementation constraints

- [ ] Keep `CSX4213_Project_V4_Images/` read-only. Never write derived files into it.
- [ ] Keep historical `IMG20260826122949/` immutable and out of V4 geometry.
- [ ] Preserve the private checkpoint under `analysis/ml/checkpoints/`; never publish it.
- [ ] Preserve `.codegraph/` as local ignored state; do not commit it.
- [ ] Preserve the large intentional V1/V2/V3 cleanup diff; do not restore deleted historical products.
- [ ] Preserve unrelated `.ai-bridge/` and other user-owned work unless the exact task requires it.
- [ ] Use one approved reconstruction stack only: Grounding DINO-T + SAM 2.1 -> ALIKED-N16Rot + LightGlue -> COLMAP/pyCOLMAP -> CUDA PatchMatch geometric -> Poisson -> Blender.
- [ ] Do not use Codex CLI.
- [ ] Before modifying reusable source, use CodeGraph for definitions/references/callers/tests/impact, then read the exact implementation.
- [ ] Git milestone authorization is now explicit: whenever a **major verified V4 milestone** is completed, create one focused descriptive commit, stage only milestone-related files, push normally to the configured upstream branch, and verify the remote result. Do not commit every small edit. Do not create branches/tags/releases, rewrite history, force-push, or include unrelated/temp/secret material without separate authorization.
- [ ] Record restart state in `reconstruction/v4/work/stage_state.json` and invalidate only downstream stages when an upstream identity changes.
- [ ] Do not start Blender geometry cleanup until the real raw dense cloud and raw Poisson mesh pass their visual gate.
- [ ] Do not replace failed scan evidence with synthetic/reference-modeled geometry.

### Live continuation checkpoint — authoritative for the next work

The shortest evidence-backed completion path is now:

```text
fix cross-camera depth-consistency metric
-> focused distinct-pose regression
-> re-evaluate existing true3 ring evidence with corrected 372/372 masks
-> accept best real dense cloud if essential contamination/identity gate passes
   OR rerun only a genuinely broken localized transition
-> one preserved raw Poisson mesh
-> front/quarter/side/top-oblique raw gate
-> Blender conservative scan cleanup
-> final LOD0 / UV / useful detail bake
-> brass PBR from uncoated references
-> canonical .blend / .glb
-> fresh GLB re-import verification
```

Rules for this checkpoint:

- [ ] Do **not** launch another PatchMatch run before correcting `v4_postfusion.py::_depth_pair_consistency` so source depth is compared to reprojected **source-camera Z**, not reference-camera `d`.
- [ ] Add a regression with distinct camera poses; a same-pose test is insufficient to catch the coordinate-frame defect.
- [ ] Re-run the ring-transition audit against the existing true3 geometric depth maps and the corrected COLMAP-resolvable mask set. Metric re-evaluation alone must not invalidate/recompute correct PatchMatch maps.
- [ ] Preserve the COLMAP mask filename contract `<image_name>.png` (for example `foo.jpg.png`) and the 372/372 resolved-mask evidence.
- [ ] Treat `mean_consistent_fraction_at_1pct >= 0.50` as a diagnostic, **not** an authoritative blocker by itself. The authoritative dense gate remains a real recognizable mask-constrained vessel cloud with no dominant board/background contamination.
- [ ] If corrected continuity evidence identifies an obvious real break, localize and rerun only the smallest affected transition/chunks. Do not restart the full healthy/true2/true3 dense workspaces unnecessarily.
- [ ] If the corrected-mask cloud is visually plausible, finite/rank-3, recognizable, and free of dominant contamination, advance immediately to one canonical raw Poisson mesh. Do not add new report/audit layers merely to increase metric coverage.
- [ ] Preserve prior dense candidates/maps/hashes and preserve `poisson_raw.ply` unchanged once created.
- [ ] After raw Poisson passes the four-view visual gate, move directly through Blender finishing and final export; prioritize final-model quality over additional diagnostic infrastructure.

## 1. Establish V4 implementation modules, path safety, and restart state

**Create:**

```text
v4_config.py
v4_ingest.py
v4_isolation.py
v4_matching.py
v4_sparse.py
v4_dense.py
v4_mesh.py
v4_blender.py                 # only repeatable helpers needed later
run_v4.py

tests/test_v4_paths.py
tests/test_v4_ingest.py
tests/test_v4_isolation.py
tests/test_v4_matching.py
```

Reuse existing helpers instead of copying them when the API can be generalized narrowly:

```text
quality_check.py
preprocess_images.py
external_learned_recovery.py
sparse_reconstruction.py
local_reconstruction.py
local_reconstruction_io.py
```

### 1.1 Inspect structure before edits

- [ ] Call the code-intelligence status surface and confirm CodeGraph is current.
- [ ] If and only if the index is stale after workspace changes, rebuild from repository root with `codegraph index --force --quiet .` using an ordinary project shell, never Codex CLI.
- [ ] Query definitions/references/impact for `build_verified_match_database`, `generate_sequential_pairs`, `run_command`, and any config/dataclass reused by V4.
- [ ] Read their direct callers and related tests before changing their signatures or semantics.
- [ ] Prefer V4-specific wrappers/new parameters over weakening historical checks globally.

### 1.2 Define immutable and writable roots in `v4_config.py`

- [ ] Define the canonical source root exactly as `CSX4213_Project_V4_Images/`.
- [ ] Define writable roots only under `capture_v4/` and `reconstruction/v4/`.
- [ ] Reject output paths resolving inside the immutable V4 source, historical raw folder, repository parent, `.codegraph/`, or private checkpoints.
- [ ] Resolve paths before safety comparisons; do not rely on string-prefix checks alone.
- [ ] Add `tests/test_v4_paths.py` cases proving raw-source/historical/protected paths cannot be output targets and valid V4 derived paths are accepted.

### 1.3 Implement `stage_state.json`

Store per stage:

```text
stage
status
input_hashes_or_manifest_hash
config_fingerprint
tool_model_identities
output_paths
output_hashes
started_at
completed_at
failure_category
retry_decision
```

- [ ] Mark `running` before expensive work and `complete` only after stage verification.
- [ ] A completed stage is reusable only when all identity inputs and required output hashes still match.
- [ ] Input/config changes invalidate only that stage and downstream stages.
- [ ] Never automatically delete accepted upstream evidence during invalidation.

### 1.4 Refresh runtime/tool identities

- [ ] Record CPU count, RAM, CUDA device and VRAM, Python, PyTorch/CUDA, and pyCOLMAP versions.
- [ ] Verify `exiftool -ver`; planning evidence was 13.59.
- [ ] Verify `ffmpeg -version`; planning evidence was 9.0.1.
- [ ] Verify `colmap -h` or `colmap version`; planning evidence was COLMAP 4.2.0 commit `be5e291` with CUDA.
- [ ] Confirm the installed COLMAP exposes `geometric_verifier`, `mapper`, `bundle_adjuster`, `image_undistorter`, `patch_match_stereo`, `stereo_fusion`, and `poisson_mesher`.
- [ ] Record exact GroundingDINO, SAM 2.1, ALIKED/LightGlue code/weight identities when those environments are initialized.
- [ ] Do not install alternate 3D reconstruction stacks.

**Task 1 acceptance:** path protection tests pass; `stage_state.json` contract exists; runtime/tool identities are recorded; no reconstruction has been run yet.

---

## 2. Build the authoritative 688-file manifest and logical sequence model

**Outputs:**

```text
capture_v4/manifests/source_manifest.csv
capture_v4/manifests/sequences.json
capture_v4/manifests/media_audit.json
capture_v4/manifests/exclusions.csv
```

### 2.1 Inventory every source exactly once

- [ ] Enumerate JPEG files only from `CSX4213_Project_V4_Images/`.
- [ ] Require exactly 688 files before continuing. If count differs, stop ingest and report the mismatch; do not silently adapt to a different set.
- [ ] For every file compute SHA-256 and byte size using streaming reads.
- [ ] Decode every file with Pillow/OpenCV; record decode success and dimensions.
- [ ] Batch ExifTool metadata into the manifest.
- [ ] Record the fields defined by the spec, including camera/lens/focal/zoom/exposure/WB/timestamp information.
- [ ] Confirm every decoded image is 3072x4080 and EXIF orientation 1, or explicitly flag the exact exceptions. Do not rewrite the whole dataset just to make redundant oriented copies.

### 2.2 Apply source roles from audited ranges and verify visually/content-wise

- [ ] Mark the 158 images through 13:33:17 as `appearance_reference`.
- [ ] Mark `empty_g6`, `empty_g8`, and `empty_g9` ranges exactly as `empty_board`.
- [ ] Mark the six `geo_g*` ranges exactly as `geometry`.
- [ ] Reconcile role totals to exactly `158 appearance + 107 empty + 423 geometry = 688`.
- [ ] Run an object-presence sanity check over each range so a timestamp typo cannot silently assign an object frame to empty or vice versa.
- [ ] Write role totals/ranges to `media_audit.json`.
- [ ] Add a regression test that fails when role accounting no longer sums to the audited 688-file baseline.

### 2.3 Compute image-quality diagnostics without arbitrary rejection

For all 423 geometry images record:

```text
Laplacian/edge sharpness signal
mean/percentile luminance
clipping fractions
contrast
vessel cutoff flag once provisional detection exists
```

- [ ] Do not reject a frame from one scalar threshold alone.
- [ ] Reject only corrupt/unreadable, catastrophic blur, catastrophic clipping, or a redundant near-duplicate carrying no useful phase evidence.
- [ ] Preserve warning-level views if they improve continuity.
- [ ] Store exclusions with exact reason and source hash in `exclusions.csv`.

### 2.4 Detect exact and near duplicates

- [ ] Confirm there are no SHA-identical duplicates.
- [ ] Group same-timestamp and perceptually near-identical candidates, including `_01` pairs.
- [ ] Explicitly evaluate `IMG20260912143525.jpg` and `_01`, already known to be a near/perceptual duplicate.
- [ ] For each redundant geometry pair, choose the sharper/cleaner image as `selected_for_geometry=true` and mark the other selected=false with the same duplicate-group ID.
- [ ] Never delete, rename, or overwrite either source.

### 2.5 Detect logical revolution wraps before phase assignment

For each `geo_g*` source pass:

- [ ] Preserve filename/timestamp order.
- [ ] Determine rotation direction from ordered vessel/board motion.
- [ ] Compute visual recurrence cues over ordered frames; use more than one cue/frame neighborhood so repeated texture or one similar view cannot create a false wrap.
- [ ] Detect candidate points where a full revolution returns to its start phase.
- [ ] Confirm each wrap by comparing several frames before/after the candidate and by checking continued motion direction.
- [ ] `geo_g7` must receive special handling; do not treat 142 images as one uniformly sampled circle and do not hardcode a 71/71 split.
- [ ] If G7 contains two complete revolutions, create separate logical ring IDs (for example `g7_r1`, `g7_r2`) and choose one primary ring or deduplicate same-phase views based on sharpness/coverage. Keep supplementary unique evidence only when it adds real support.
- [ ] If any pass contains a partial repeated cycle, keep only useful unique phase support and record why repeated frames were excluded.

### 2.6 Assign phase without fabricating physical angle

- [ ] Assign `frame_index_within_logical_ring` after duplicate/repeated-cycle decisions.
- [ ] Assign `phase_01` monotonically around each complete ring using ordered phase position.
- [ ] Leave `angle_deg` null because no measured turntable angle log exists.
- [ ] Store source pass, logical ring, direction, phase, and selection state in `sequences.json`.
- [ ] Do not claim exact degrees from timestamp/index alone.

### 2.7 Classify elevation roles descriptively

- [ ] Use visual camera elevation/framing to order logical rings from lower/horizontal through upper/steep-high.
- [ ] Keep the stable `geo_g*`/logical-ring IDs independent of descriptive role so later reclassification cannot break cache identities.
- [ ] Confirm G12 supplies steep-high/top/interior evidence.
- [ ] Do not invent camera elevation angles.

**Task 2 acceptance:** all 688 files appear once in the manifest; exact role totals reconcile; every selected geometry frame belongs to one auditable logical ring with direction and normalized phase; G7 repeated coverage is explicitly resolved; excluded views retain provenance.

---

## 3. Build full-resolution vessel masks with explicit board suppression

**Outputs per selected geometry image:**

```text
capture_v4/derived/masks/<image>.png
capture_v4/derived/feature_masks/<image>.png
capture_v4/derived/mvs_images/<image>.jpg or lossless equivalent
mask diagnostics in reconstruction/v4/reports/
```

### 3.1 Initialize official pretrained segmentation

- [ ] Use Grounding DINO-T/Swin-T for a vessel ROI/box.
- [ ] Use SAM 2.1 Hiera-small for full-resolution segmentation.
- [ ] Run on CUDA when supported.
- [ ] Pin/record model/code identities and prompt/config values in stage state.
- [ ] Do not train a custom detector/segmenter and do not use the private SmallSeg checkpoint as the V4 primary method.

### 3.2 Generate masks ring by ring

- [ ] Start each logical ring from a high-confidence Grounding DINO localization.
- [ ] Use SAM 2 ordered/temporal propagation only while the outline stays stable.
- [ ] Refresh the box/prompt when the mask drifts, collapses, grows onto the board, or clips a real vessel part.
- [ ] Independent still segmentation is acceptable where propagation is less reliable than per-frame prompting.
- [ ] Save a binary full-resolution mask with dimensions exactly matching 3072x4080 source geometry.

### 3.3 Preserve all critical vessel parts

Every mask review must explicitly verify:

```text
finial ball and stepped/ridged tip
lid/top tiers
long neck and lip/opening
shoulder/globe
bowl rim
visible bowl interior boundary
outer bowl
pedestal rings/transitions
base/foot
```

- [ ] Do not erode the thin finial or rim merely to make a smooth mask.
- [ ] Do not close the visible bowl opening with a fabricated filled disk.
- [ ] Keep actual interior support where it is visible.

### 3.4 Use G8/G9 empty-board sweeps as negative evidence

For G8 and G9 only, where same-setup empty tails are known:

- [ ] Compute board/background descriptors outside the provisional vessel mask for each geometry view.
- [ ] Find the best compatible empty-board phase candidate from the corresponding empty suffix.
- [ ] Require sufficient similarity/confidence before using that empty reference.
- [ ] Use the aligned empty view to identify wood pixels erroneously included at/under the vessel base.
- [ ] Remove only supported leakage; never add foreground based on differencing.
- [ ] Record geometry image -> empty reference mapping and confidence.
- [ ] Ignore transient hand pixels when evaluating empty/reference compatibility.

For `empty_g6`:

- [ ] Test camera/framing compatibility with geometry passes before using it.
- [ ] If incompatible, do not force an association; retain it as audited negative/reference evidence only.

### 3.5 Apply deterministic board-leak diagnostics

For every mask:

- [ ] Compute area, bbox, centroid, and largest connected-component ratio.
- [ ] Flag sudden frame-to-frame mask-area/centroid changes within a ring.
- [ ] Flag mask support extending broadly along the board plane below/around the foot.
- [ ] Flag abnormally wide horizontal mask support at base height.
- [ ] Inspect the base-contact region at native/high resolution on all flagged images.
- [ ] Prefer conservative exclusion of ambiguous board pixels over absorbing wood grain into the vessel.
- [ ] Do not use outward mask dilation near the base if it reaches board texture.

### 3.6 Produce feature masks without creating mask-boundary features

- [ ] Derive the feature mask from the accepted full mask.
- [ ] Use only a small inward safety margin where it does not remove thin real geometry.
- [ ] Add an edge/boundary exclusion band for keypoint filtering so artificial segmentation boundaries do not become correspondences.
- [ ] Preserve original full-image coordinates.

### 3.7 Produce MVS inputs

- [ ] Keep object photometry unchanged inside the accepted vessel support.
- [ ] Neutralize or mask pixels outside the MVS mask without cropping the image.
- [ ] Do not bake dry-shampoo/marker appearance into later material; these images are geometry inputs only.

### 3.8 Verify every selected mask visually

- [ ] Generate deterministic overlay/contact sheets covering **every selected geometry frame**, not a random subset.
- [ ] Review all overlays for finial/rim/base clipping and board/cloth/hand leakage.
- [ ] Re-run/refine only the failing frames/ring portions.
- [ ] Require zero imported feature keypoints outside the feature mask in a focused regression test.
- [ ] Add a test proving source/mask/feature-mask/MVS image dimensions and coordinate geometry match.

**Task 3 acceptance:** every selected geometry frame has an audited full mask, feature mask, and MVS support; the complete vessel is preserved; board/cloth/hands are excluded; the base does not include a wide wood patch.

---

## 4. Generalize ALIKED/LightGlue integration and choose one production resource configuration

### 4.1 Inspect reusable frontend blast radius

- [ ] Use CodeGraph references/impact for `ExternalLearnedConfig`, `generate_sequential_pairs`, feature-cache preparation, pair matching, and database import/verification helpers.
- [ ] Read exact source and directly related tests.
- [ ] Remove V4 dependence on `expected_images=288`, Step-13 naming/contracts, old V3 source/gate assumptions, and exactly-one-camera acceptance logic.
- [ ] Preserve historical behavior where it is still used by historical tests; introduce generic parameters/V4 wrappers rather than silently changing unrelated semantics.

### 4.2 Run one bounded ALIKED resource preflight

Known input is 3072x4080 and distant rings make the vessel relatively small.

- [ ] Select one representative distant geometry frame with an accepted feature mask.
- [ ] Inspect free VRAM through PyTorch or another working CUDA interface.
- [ ] Attempt ALIKED-N16Rot at max image size 3072 if live VRAM headroom makes that reasonable.
- [ ] If that one bounded run OOMs or leaves unsafe headroom, use 2048.
- [ ] Keep maximum keypoints at 4096 unless a concrete implementation defect proves this impossible; do not sweep keypoint counts.
- [ ] Use CUDA device 0, mixed precision when supported, and LightGlue adaptive depth/width.
- [ ] Lock the resulting config before bulk feature extraction and fingerprint it.
- [ ] Do not compare SIFT or another feature family.

### 4.3 Extract and cache only vessel-supported ALIKED features

- [ ] Cache by source-image SHA + feature-mask SHA + ALIKED model/config identity.
- [ ] Map any feature-only scaled coordinates exactly back to original 3072x4080 image coordinates before database import.
- [ ] Drop every keypoint outside the feature mask.
- [ ] Drop keypoints inside the explicit segmentation-boundary exclusion band.
- [ ] Record per-image raw keypoint count and retained on-mask count.
- [ ] If an entire ring has unexpectedly low retained counts, inspect mask/scaling first rather than switching algorithms.

**Task 4 acceptance:** one locked ALIKED resource config exists; every selected geometry image has a provenance-safe, vessel-only feature cache in original image coordinates.

---

## 5. Estimate cross-ring phase offsets and build deterministic acquisition-aware pairs

### 5.1 Replace fixed-index sequential assumptions

- [ ] Implement a generic pair builder in `v4_matching.py` using `phase_01`, not raw frame index.
- [ ] Keep or wrap `generate_sequential_pairs` only if its API can represent circular unequal-sized rings correctly.
- [ ] Pair immediate phase neighbors within each logical ring.
- [ ] Add a wider local phase band sufficient to create longer tracks.
- [ ] Include explicit wrap closure across phase 1 -> 0.
- [ ] Never create self-pairs or duplicate unordered pairs.

Do not use one fixed `±N indices` definition across all rings because real source counts vary from 37 to 72 per ordinary pass and G7 contains repeated coverage.

### 5.2 Estimate cross-ring circular offsets from vessel-only learned evidence

For each adjacent elevation pair:

- [ ] Build a small deterministic candidate set of relative phase offsets.
- [ ] For representative masked image pairs, run/reuse ALIKED + LightGlue and COLMAP/two-view geometric verification.
- [ ] Score each candidate from geometrically verified vessel-only support, not raw descriptor match count alone.
- [ ] Choose one circular offset with the strongest coherent support and store score/confidence in `sequences.json` or a dedicated pairing report.
- [ ] If two offsets are ambiguous because of near-symmetry, retain a slightly wider local cross-ring phase neighborhood rather than inventing an exact orientation.
- [ ] Do not exhaustively all-pairs every ring.

### 5.3 Create final pair schedule

- [ ] For each frame, pair nearest same-ring phase neighbors in both directions.
- [ ] Add wider same-ring phase neighbors.
- [ ] Pair to nearest offset-corrected phase in adjacent elevation ring plus a small neighboring phase window.
- [ ] Ensure every logical ring has at least one cross-ring connection candidate.
- [ ] Ensure removed near-duplicates do not create a gap that breaks ring closure; bridge by phase distance.
- [ ] Serialize the exact ordered/unordered pair set and hash it for restart identity.

### 5.4 Focused deterministic tests

- [ ] Same inputs produce the same pair set/order.
- [ ] No self/duplicate unordered pairs.
- [ ] Phase 0/1 closure works.
- [ ] Unequal frame-count rings pair by nearest phase, not same index.
- [ ] Missing/removed phase views are bridged deterministically.
- [ ] Cross-ring offset is applied consistently.

**Task 5 acceptance:** one deterministic vessel-aware circular/cross-ring pair schedule exists, based on real logical rings and estimated cross-ring offsets.

---

## 6. Build the verified V4 COLMAP match database

### 6.1 Create a clean V4 database

- [ ] Use only selected geometry images.
- [ ] Do not import appearance-reference or empty-board images as reconstruction cameras.
- [ ] Create `reconstruction/v4/sparse/database.db` from scratch for V4.
- [ ] Populate image/camera records from the chosen camera-group contract.

### 6.2 Camera grouping default

EXIF supports one shared geometry intrinsic group:

```text
same OPPO Reno12 F rear lens
3.98 mm / 26 mm-equivalent
3072x4080
orientation 1
digital zoom 1
f/1.8
```

- [ ] Default to one shared intrinsic group for selected geometry views.
- [ ] Before freezing this, confirm there is no hidden crop/rotation metadata or obvious image-dimension inconsistency.
- [ ] Moving the physical camera between rings changes extrinsics, not lens intrinsics; do not split by ring for that reason alone.
- [ ] Split camera groups only if actual metadata or later calibration behavior proves a different intrinsic contract.

### 6.3 Import ALIKED/LightGlue evidence

- [ ] Import only vessel-mask-filtered keypoints.
- [ ] Match only the Task 5 pair schedule with LightGlue.
- [ ] Cache matches by pair schedule hash + feature-cache identities + LightGlue identity/config.
- [ ] Import matches into the V4 database.
- [ ] Run COLMAP/pyCOLMAP geometric verification for imported pairs.
- [ ] Record verified inlier counts/distribution by same-ring and cross-ring category.

### 6.4 Board-contamination pre-map check

- [ ] Verify imported keypoints visually on representative frames, including base-contact regions.
- [ ] Confirm no keypoints lie on wood, white cloth, hands, or mask boundary.
- [ ] If board keypoints are found, fix masks/filtering and rebuild only affected feature/match/database state before mapping.

**Task 6 acceptance:** `database.db` contains only selected geometry cameras, vessel-supported ALIKED keypoints, scheduled LightGlue matches, and geometrically verified pair evidence.

---

## 7. Build and gate the V4 sparse reconstruction

### 7.1 Generalize sparse helpers narrowly

- [ ] Use CodeGraph before editing `sparse_reconstruction.py`.
- [ ] Remove V4-facing dependence on `expected_images=288` and the old >=274 registration threshold.
- [ ] Keep historical semantics isolated if historical code/tests still depend on them.

### 7.2 Choose the initial camera model

- [ ] Start the V4 model with `SIMPLE_RADIAL` shared intrinsics.
- [ ] Run one incremental pyCOLMAP mapping + bundle-adjustment pipeline using the verified learned database.
- [ ] Do not run a SIFT reconstruction comparison.
- [ ] If the sparse solution has clearly implausible focal/distortion calibration despite good correspondences, diagnose camera model once and change only on evidence; do not sweep camera models.

### 7.3 Rank/select connected sparse models

If pyCOLMAP emits multiple models:

- [ ] Prefer the model with broad logical-ring coverage, cross-elevation connectivity, strong tracks/observations, finite reprojection behavior, and coherent geometry.
- [ ] Do not choose purely by registered-image count.
- [ ] Record omitted rings/views and reasons.

### 7.4 Generate sparse diagnostics

Produce `reconstruction/v4/reports/sparse_gate.json` plus preview images containing:

```text
registered selected / selected total by logical ring
unregistered filenames by ring
camera centers/frusta colored by ring
track/observation statistics
reprojection summary
sparse point cloud views
estimated intrinsics/distortion
board/background contamination assessment
```

### 7.5 Sparse visual acceptance

Proceed only when all are true:

- [ ] virtual cameras form coherent circular/orbital trajectories rather than pose islands;
- [ ] elevation rings connect into one reconstruction;
- [ ] no gross pose jump/explosion exists;
- [ ] sparse points concentrate on the vessel silhouette/surface;
- [ ] no large flat wood-board cloud/plane dominates beneath the vessel;
- [ ] no cloth fold/background structure dominates;
- [ ] bowl/globe/neck/finial/pedestal support is recognizable enough to justify dense MVS.

If this gate fails, fix the causal upstream issue in this order: role/ring/phase -> masks/board leakage -> pair schedule/offset -> matching import -> camera model only if diagnostics implicate it. Do not jump to an alternate reconstruction stack.

**Task 7 acceptance:** one accepted sparse model exists under `reconstruction/v4/sparse/best/` and passes the vessel-centered/board-free visual gate.

---

## 8. Prepare dense workspace and prove CUDA PatchMatch with a real smoke run

### 8.1 Inspect process-helper impact

- [ ] Use CodeGraph for `local_reconstruction.run_command` and direct tests/callers.
- [ ] Reuse process execution, timeout, and failure-classification helpers where suitable.
- [ ] Do not inherit the old `reconstruction/local_dense` path policy.

### 8.2 Verify exact COLMAP 4.2 options against the installed executable

Before constructing production commands:

- [ ] Run `colmap image_undistorter -h`.
- [ ] Run `colmap patch_match_stereo -h`.
- [ ] Run `colmap stereo_fusion -h`.
- [ ] Record exact supported option names for max image size, geometric consistency, filtering, GPU selection, and fusion masks.
- [ ] Use the installed help output rather than copying stale CLI spelling from old code/docs.

### 8.3 Undistort accepted geometry images

- [ ] Use only selected images in the accepted sparse model.
- [ ] Use the accepted MVS image source and sparse calibration.
- [ ] Baseline max image size is 2000.
- [ ] Output to `reconstruction/v4/dense/workspace/` in COLMAP format.
- [ ] Preserve mapping from original filename to undistorted dense filename.

### 8.4 Transform masks through the same geometry

- [ ] Do not merely resize original masks.
- [ ] Apply the accepted sparse-camera undistortion mapping to masks with nearest-neighbor binary resampling.
- [ ] Write aligned masks under `capture_v4/derived/undistorted_masks/` or the exact COLMAP-compatible mask path.
- [ ] Overlay representative masks on undistorted images from low/middle/high rings.
- [ ] Check especially finial, bowl rim/interior, base contact, and image borders.

### 8.5 Run a real bounded CUDA PatchMatch smoke

This gate is mandatory even though `colmap -h` reports `with CUDA`.

- [ ] Create a bounded smoke workspace/config using a small representative connected subset containing views from at least two elevations and neighboring phase support.
- [ ] Run `patch_match_stereo` with geometric consistency on GPU 0 using the exact installed 4.2 syntax.
- [ ] Verify the process completes without CUDA/runtime failure and produces readable depth/normal output for the smoke views.
- [ ] Record the command/config/tool identity and smoke output status in the dense gate report.
- [ ] Remove smoke-only residue after it has served verification, unless it is deliberately kept as a tiny reproducibility report; do not mix smoke outputs with production workspace.

**Task 8 acceptance:** accepted sparse images and aligned masks are undistorted correctly, and an actual multi-view CUDA PatchMatch smoke run has succeeded.

---

## 9. Run production geometric PatchMatch and mask-aware fusion

### 9.1 Lock one dense configuration

Start with:

```text
max_image_size = 2000
geom_consistency = true
filter = true
gpu_index = 0
source selection = COLMAP automatic/accepted default
fusion input_type = geometric
fusion mask path = aligned undistorted vessel masks
```

- [ ] Fingerprint sparse model, dense images, masks, COLMAP version, and config before starting.
- [ ] Run only one CUDA-heavy reconstruction process at a time.

### 9.2 Run/resume PatchMatch

- [ ] Start full `patch_match_stereo` only after Task 8 passes.
- [ ] Preserve completed per-view outputs across interruption when the dense fingerprint is unchanged.
- [ ] On restart, verify identity first and resume rather than deleting/recomputing valid views.
- [ ] If CUDA OOM is diagnosed, change only max image size `2000 -> 1600`, update the config fingerprint, and rerun the same algorithm.
- [ ] Do not switch to photometric-only, OpenMVS, Meshroom, or another dense stack.

### 9.3 Run geometric stereo fusion with vessel masks

- [ ] Use geometric depth maps.
- [ ] Pass the correctly aligned undistorted mask path using the exact COLMAP 4.2 option discovered in Task 8.
- [ ] Write only the canonical fused cloud to `reconstruction/v4/dense/fused.ply` after successful fusion.
- [ ] Compute SHA-256 once accepted and treat the fused cloud as immutable upstream evidence.

### 9.4 Dense evidence report

Record:

```text
production dense size
completed depth/normal count
fused point count
finite fraction
bounds
source sparse hash
mask-set hash/config
front / quarter / side / top-oblique cloud previews
```

- [ ] Inspect for a large flat board plane, curtain surfaces, or pedestal-to-board webbing.
- [ ] If board geometry dominates, fail the dense gate and return to mask/fusion diagnostics; do not crop the resulting cloud blindly to hide an upstream masking defect unless the crop is a defensible vessel-support operation explicitly recorded.

**Task 9 acceptance:** `reconstruction/v4/dense/fused.ply` is a real mask-constrained geometric-consistency vessel cloud without dominant board/background contamination.

**Task 9 continuation note (2026-09-13):** cross-ring continuity is supporting diagnostic evidence, not a replacement acceptance criterion. A hard `mean_consistent_fraction_at_1pct >= 0.50` rule is not authorized by this plan. Correct the coordinate-frame bug first; if the corrected audit shows no actual structural break and the dense cloud satisfies the acceptance sentence above, proceed to Task 10 without another broad dense rerun.

---

## 10. Create and visually gate the raw Poisson mesh

### 10.1 Select one Poisson depth from actual point density

- [ ] Inspect fused point count/density.
- [ ] Use depth 13 when appropriate; choose one lower/higher value only if the actual density makes 13 unreasonable.
- [ ] Do not sweep Poisson parameters or meshing algorithms.

### 10.2 Generate canonical raw mesh

- [ ] Run COLMAP `poisson_mesher` or the project-approved Poisson invocation against the full fused cloud.
- [ ] Write `reconstruction/v4/mesh/poisson_raw.ply`.
- [ ] Hash it and treat it as immutable raw evidence.
- [ ] Record vertices, faces, connected components, dominant-component fraction, finite bounds, and fused-cloud identity.

### 10.3 Mandatory raw visual gate

Before any Blender cleanup:

- [ ] render/view the fused cloud from front, quarter, side, and top-oblique;
- [ ] render/view the unmodified Poisson mesh from the same views;
- [ ] verify recognizable bowl/rim/interior, globe/shoulder, neck, lid/finial, pedestal/rings, and base;
- [ ] verify the result is not primarily a board slab or curtain-backed object;
- [ ] preserve the raw previews/review report.

If completing the object would require inventing most geometry, stop here and report the actual scan limitation. Do not procedurally model a replacement vessel and call it reconstructed.

**Task 10 acceptance:** the unmodified fused cloud and raw Poisson mesh are visually plausible enough to justify conservative finishing.

---

## 11. Initialize Blender only after raw acceptance

Before Blender mutation:

- [ ] Read `/mnt/data/Blender_Skills_1_2026-09-06.zip` guidance beginning with `CHATGPT_START_HERE.md`, `SKILLS_INDEX.md`, and `blender-director/SKILL.md` when that bundle is available in the execution environment.
- [ ] Inspect live Blender MCP/tool schemas and current scene state; live capabilities override legacy examples.
- [ ] Saving, rendering, exporting, and overwriting canonical V4 Blender outputs are in scope only as required by this plan; do not touch unrelated scenes/assets.

### 11.1 Create scene collections

```text
COL_V4_Source
COL_V4_Work
COL_V4_Final
COL_V4_Lookdev
```

- [ ] Import `poisson_raw.ply` as `SM_V4_Poisson_Raw` into `COL_V4_Source`.
- [ ] Keep the source object unchanged/locked from destructive editing.
- [ ] Capture front/quarter/side/top-oblique raw review images in Blender before cleanup.

### 11.2 Create clean-high working copy

- [ ] Duplicate the raw object as `SM_V4_Scan_CleanHigh` under `COL_V4_Work`.
- [ ] Perform all destructive cleanup on this copy only.

### 11.3 Conservative cleanup order

- [ ] Remove clearly disconnected floaters/noise unsupported by photographs/raw surface continuity.
- [ ] Keep ambiguous nearby components until reference evidence resolves them.
- [ ] Merge near-duplicate vertices only where appropriate.
- [ ] Recalculate/fix normals and straightforward non-manifold scan defects.
- [ ] Remove impossible internal debris that is clearly scan noise, not bowl/interior surface.
- [ ] Repair holes only when surrounding observed real surface bounds the repair.
- [ ] For porous regions, use controlled voxel-remesh on a working duplicate only when necessary, then project/shrinkwrap toward the accepted scan.
- [ ] Use local smooth/relax/sculpt only to restore continuity supported by the scan/reference images.
- [ ] Never erase/rebuild the characteristic bowl rim/interior, globe, shoulder steps, long neck/opening, lid tiers/finial, pedestal transitions, or base because a synthetic primitive looks cleaner.

### 11.4 Scale policy

- [ ] Search existing project docs/evidence for an actual physical vessel measurement before final scale.
- [ ] If a reliable measurement exists, apply and document it.
- [ ] If none exists, do not block completion and do not invent dimensions; keep a consistent normalized scale and state that absolute scale is unverified.

### 11.5 Save the working Blender master checkpoint

- [ ] Save to `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend` once the source/raw and cleaned-high objects are correctly organized.
- [ ] Reopen/read the saved file or verify scene state after save before claiming it exists correctly.

**Task 11 acceptance:** a recognizable scan-derived cleaned-high vessel exists in Blender beside the preserved raw source; no synthetic replacement has occurred.

---

## 12. Build production mesh, UVs, and evidence-backed detail bake

### 12.1 Create final LOD0

- [ ] Duplicate the accepted clean-high object into `COL_V4_Final` as `SM_V4_Vessel_LOD0`.
- [ ] Reduce geometry while protecting silhouette and real openings/transitions.
- [ ] Use roughly 250k-350k triangles as a target, not a mandatory count; justify a materially different budget from actual scan complexity/performance.
- [ ] Compare front/quarter/side/top silhouettes to `SM_V4_Scan_CleanHigh` after reduction.

### 12.2 UV unwrap

- [ ] Align the vessel's physical vertical axis with Blender Z without changing scan shape.
- [ ] Place main seams on low-visibility areas and logical ring/opening boundaries.
- [ ] Preserve separate topology around the bowl opening/interior and neck/lip.
- [ ] Pack one practical 0-1 UV set with adequate bake margins.
- [ ] Verify no gross overlap exists where unique baked detail is expected.

### 12.3 Bake only useful scan detail

- [ ] Compare production mesh to clean-high.
- [ ] Bake tangent-space normal and/or AO only when decimation removed visible useful scan detail.
- [ ] Verify bake projection/cage does not project across bowl opening, rim, neck, or neighboring thin surfaces.
- [ ] Do not bake marker/dry-shampoo color as final material information.

**Task 12 acceptance:** `SM_V4_Vessel_LOD0` has a preserved silhouette, usable UVs, and only necessary geometry-detail bakes.

---

## 13. Build final brass material from the 158 uncoated references

### 13.1 Select reference evidence

- [ ] Use the early broad uncoated clean rings as the primary stable brass appearance reference.
- [ ] Use close/detail uncoated views for local ornament/wear context only.
- [ ] Mask/ignore tripod, green sheet, white/wood background, hands, and other non-vessel pixels.
- [ ] Reject clipped highlight pixels when estimating base brass color/roughness.
- [ ] Treat strong specular reflections as lighting observations, not literal diffuse color.
- [ ] Never use coated/black-marker geometry images for final base color/roughness texture.

### 13.2 Create `MAT_V4_Brass`

Start from:

```text
Principled BSDF
Metallic = 1.0
Base Color = robust reference-supported brass tone
Roughness = restrained reference-supported range
Normal/AO = accepted bake where useful
Wear/variation = only visible/reference-supported variation
```

- [ ] Keep the material explainable and portable to GLB.
- [ ] Avoid procedural dirt/ornament invented merely to hide scan defects.
- [ ] If image-derived texture maps are used, store canonical assets under `reconstruction/v4/blender/textures/` and ensure paths are portable/packed as appropriate.

### 13.3 Neutral lookdev verification

- [ ] Use one neutral lookdev rig in `COL_V4_Lookdev`.
- [ ] Inspect front/quarter/side/top-oblique views.
- [ ] Confirm the object reads as brass rather than dry-shampoo/marker-coated geometry.
- [ ] Do not reopen geometry reconstruction for purely cosmetic material tuning unless a real geometric defect is revealed.

**Task 13 acceptance:** final production mesh has a portable brass PBR material supported by the uncoated V4 reference set.

---

## 14. Save canonical Blender master, export GLB, and independently verify portability

### 14.1 Final Blender audit

Before final save/export verify:

```text
SM_V4_Poisson_Raw preserved
SM_V4_Scan_CleanHigh present
SM_V4_Vessel_LOD0 present and active final geometry
expected collections only for V4 work
valid normals
nonzero mesh
sane transforms
UV map when material/bakes require it
MAT_V4_Brass assigned
texture paths resolve
no raw/debug object accidentally selected for export
```

- [ ] Save canonical master to `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend`.
- [ ] Verify the saved `.blend` reopens with the expected collections/objects/material.

### 14.2 Export final GLB

- [ ] Export only `SM_V4_Vessel_LOD0` and required material/texture assets.
- [ ] Apply/export transforms consistently without silently changing shape.
- [ ] Write `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb`.
- [ ] Record output SHA-256 and file size.

### 14.3 Fresh-process GLB re-import gate

- [ ] Open a fresh Blender scene/process, not the authoring scene.
- [ ] Import the canonical GLB.
- [ ] Confirm mesh exists and has nonzero vertices/faces.
- [ ] Confirm material exists and texture references resolve.
- [ ] Confirm object bounds/scale are sane and correspond to the source final mesh.
- [ ] Confirm no gross normal/shading breakage.
- [ ] Confirm raw/source/debug geometry was not accidentally exported.
- [ ] Capture a final front/quarter/side/top-oblique inspection contact sheet from the re-imported GLB.

**Task 14 acceptance:** both canonical final files exist, the `.blend` opens correctly, and the GLB independently re-imports as the same usable vessel.

---

## 15. Documentation, cleanup, final verification, and plan closure

### 15.1 Update only required project state/docs

- [ ] Update `README.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md` with verified V4 completion facts and canonical final paths.
- [ ] Record real tool/model versions, selected logical-ring counts, selected-image counts, sparse coverage, dense/mesh metrics, and final asset verification results without fabricating numbers.
- [ ] Keep professor-facing narrative concise unless separately requested.

### 15.2 Focused code verification

Run only relevant checks:

- [ ] compile/import new V4 Python modules;
- [ ] `tests/test_v4_paths.py`;
- [ ] `tests/test_v4_ingest.py`;
- [ ] `tests/test_v4_isolation.py`;
- [ ] `tests/test_v4_matching.py`;
- [ ] direct focused tests for any reused helper whose behavior was modified;
- [ ] do not claim broad historical tests passed unless they were actually run.

### 15.3 Final defect-focused review

- [ ] Inspect changed source with bounded/path-specific diffs; avoid one giant diff that can hit ENOBUFS.
- [ ] Check for accidental V1/V2/V3 restoration.
- [ ] Check source images are unchanged by comparing manifest hashes.
- [ ] Check no private checkpoint, `.codegraph/`, temp contact sheet, cache, test residue, debug log, or unrelated artifact is staged/published.
- [ ] Remove only task-created temporary residue that is no longer required; preserve canonical reports/checkpoints/evidence.
- [ ] Verify final stage-state identities and outputs.

### 15.4 Close this plan only from evidence

- [ ] Mark each implementation checkbox complete only after its required output/gate is verified.
- [ ] Do not mark downstream tasks complete because a file merely exists.
- [ ] If a hard external blocker remains, leave the corresponding checkbox unchecked and document exact evidence/blocker.

**Task 15 acceptance:** requested V4 scope is complete, verification is evidence-backed, unrelated user work remains intact, and no temporary planning/implementation residue remains.

---

## Essential gate matrix

| Gate | Required evidence | Failure action |
| --- | --- | --- |
| Source/manifest | 688 hashes/decode; 158/107/423 roles reconcile; logical rings resolved | fix ingest/classification; do not reconstruct |
| Mask/board | every selected mask reviewed; full vessel retained; board/cloth/hands excluded | refine segmentation/negative evidence |
| Learned database | vessel-only ALIKED keypoints; deterministic ring-aware pairs; verified LightGlue geometry | fix masks/pairs/import |
| Sparse | coherent cross-connected virtual rings; vessel-centered cloud; no dominant board plane | fix causal upstream stage |
| CUDA dense smoke | real PatchMatch subset produces depth/normal output | diagnose CUDA/options before full run |
| Dense | geometric fused cloud exists with aligned vessel masks; no dominant board/background | return to mask/fusion diagnostics |
| Raw mesh | unmodified Poisson is recognizable enough to finish without inventing most geometry | stop before Blender replacement modeling |
| Blender | canonical `.blend` opens with preserved source + clean high + final LOD0/material | repair master state |
| GLB | fresh import contains correct mesh/material/textures/bounds/shading | fix export/material portability |

## Execution sequence

Execute strictly in this order unless a failed gate sends work back to its causal upstream stage:

```text
1  paths / tools / restart state
2  688-file manifest / roles / duplicates / revolution wraps / phases
3  Grounded-SAM2 masks + board suppression
4  ALIKED resource preflight + vessel-only feature caches
5  cross-ring phase offsets + deterministic pair schedule
6  LightGlue + COLMAP geometric verification database
7  pyCOLMAP sparse SfM/BA + board-free sparse gate
8  image+mask undistortion + real CUDA PatchMatch smoke
9  full geometric PatchMatch + masked geometric fusion
10 Poisson raw mesh + mandatory raw visual gate
11 Blender preserve/import + conservative clean-high
12 production mesh + UV + useful detail bake
13 clean-reference brass material
14 canonical .blend + .glb + fresh GLB re-import
15 docs / cleanup / focused verification / checkbox closure
```

Do not pause for minor missing views, benign warnings, or cosmetic defects when an essential gate still passes. Do stop before expensive downstream work when an essential gate fails.

## Planning self-review — completed 2026-09-12

- [x] Every final supplied image was accounted for: 688 total.
- [x] Source roles were resolved from real media: 158 appearance, 107 empty-board, 423 object-bearing geometry.
- [x] Exact G8/G9 geometry-to-empty boundaries were verified visually.
- [x] All source dimensions/orientations/device/lens/focal/zoom metadata were audited.
- [x] Geometry exposure settings were found consistent; variable early exposure/WB belongs to appearance/detail refs.
- [x] No byte-identical duplicate exists; same-second near-duplicate handling is planned.
- [x] G7 repeated-revolution risk is explicitly handled instead of assuming one 142-frame ring.
- [x] Cross-ring phase-offset estimation is explicit because start angles are not measured/aligned.
- [x] Wood-board leakage is treated as a first-class failure mode in segmentation, sparse, and dense gates.
- [x] Same-setup rotating empty-board tails are used as negative refinement evidence without becoming a second reconstruction method.
- [x] High-angle/top/interior coverage exists in the supplied media.
- [x] Final clean/uncoated references exist and are reserved for material/lookdev rather than geometry matching.
- [x] Actual metadata supports one shared intrinsic group by default, while allowing evidence-backed split/model correction.
- [x] A real bounded CUDA PatchMatch smoke gate is required before production dense work.
- [x] Raw dense/Poisson evidence remains mandatory before Blender cleanup or synthetic intervention.
- [x] The plan does not request recapture and does not depend on unavailable future images.
- [x] No V4 reconstruction implementation, Blender mutation, final export, commit, or push was performed during this planning pass.
