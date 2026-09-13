# V4 Fast End-to-End Reconstruction Design

**Status:** Approved execution design, updated 2026-09-13 for the current dense/post-fusion checkpoint. Earlier V4 ingest, masking, matching, sparse reconstruction, dense PatchMatch, and fusion work now exist; continue from the newest verified state rather than replaying the plan from Task 1.

**Goal:** Reconstruct the Thai libation brass vessel from the supplied final fixed-camera turntable capture and deliver a visually accurate, inspectable Blender master plus a working GLB using one evidence-bound reconstruction path. The supplied images are final and cannot be retaken; software must adapt to the media that exists.

### Current continuation checkpoint — 2026-09-13

The selected V4 geometry set has reached 372/372 sparse registration and dense reconstruction has already produced preserved 2000-pixel geometric PatchMatch maps and multiple fused candidates. The current work is **not** to restart upstream reconstruction. It is to make the dense acceptance evidence geometrically correct, accept the best real fused cloud when the authoritative visual/contamination gates pass, then finish Poisson and Blender.

The current corrected-mask fusion contract resolves all 372 masks using COLMAP's `<image_name>.png` rule. Post-fusion evidence reports no detected dominant board slab, cloth/background structure, or pedestal-board webbing and confirms vessel identity. Preserve those fused-cloud reports, maps, candidates, and hashes.

One known audit defect must be corrected before its cross-ring scores are used: `v4_postfusion.py::_depth_pair_consistency` reprojects reference depth samples into the source camera but currently discards the source-camera Z returned by `_project_points`, then compares the source depth map against the original reference-camera depth. That mixes depth values from different camera coordinate frames. The corrected metric must compare source depth against reprojected source-camera Z and must have a regression with genuinely distinct camera poses. Re-evaluating this metric uses the existing true3 depth maps and corrected masks; it does not require a new PatchMatch run.

The implementation value `mean_consistent_fraction_at_1pct >= 0.50` is **diagnostic only** unless later source evidence proves that it corresponds to a real geometry failure. It is not an authoritative replacement for the essential Task 9/10 acceptance: a real mask-constrained finite/rank-3 vessel cloud without dominant board/background contamination, followed by direct fused-cloud/raw-Poisson inspection. If corrected continuity evidence exposes a genuine transition break, fix only that localized transition. Otherwise proceed to one preserved raw Poisson mesh, then Blender.

## 1. Scope and non-negotiable boundaries

V1, V2, and V3 are rejected historical work. Their reconstruction artifacts are not V4 geometry inputs. Capture-independent utilities may be generalized and reused only after their V3-specific assumptions are removed from V4-facing paths.

The final incoming V4 source is:

```text
CSX4213_Project_V4_Images/
```

It contains exactly 688 JPEG files. This directory is an immutable source boundary. Do not rename, move, rotate, crop, resize, recompress, overwrite, or delete these files. Do not unpack/copy the sibling ZIP merely to duplicate the same source data. V4 manifests and derived products live under `capture_v4/`; reconstruction products live under `reconstruction/v4/`.

Historical `IMG20260826122949/` remains immutable and is not V4 geometry input.

The single V4 geometry route remains:

```text
final immutable V4 still images
-> deterministic manifest / role classification / QA
-> Grounding DINO-T + SAM 2.1 vessel masks
-> board/background leakage suppression
-> conservative geometry-preserving matching inputs
-> mask-filtered ALIKED-N16Rot
-> LightGlue
-> COLMAP geometric verification
-> pyCOLMAP incremental SfM + bundle adjustment
-> accepted image + mask undistortion
-> bounded CUDA PatchMatch smoke check
-> CUDA COLMAP PatchMatch with geometric consistency
-> mask-aware geometric stereo fusion
-> Poisson raw mesh
-> mandatory raw dense/mesh visual gate
-> Blender conservative scan cleanup
-> production mesh / UV / useful detail bake
-> brass PBR from uncoated V4 references
-> editable .blend
-> final .glb
-> clean GLB re-import verification
```

No SIFT-vs-ALIKED, VGGT, MASt3R, OpenMVS, Meshroom, NeRF, Gaussian-splatting, dense-method, or mesher bake-off is part of V4. An alternate reconstruction stack is not a fallback for a normal failure; diagnose the approved path first.

## 2. Final-media audit: authoritative evidence

### 2.1 Complete inventory

All 688 images were visually reviewed through contact sheets and numerically inspected from decoded pixels/EXIF. Natural timestamp clusters use gaps greater than 20 seconds only as an audit aid; implementation must classify roles from the manifest and image content rather than treating every timestamp cluster as one reconstruction ring.

The source-role accounting is complete:

| Role | Count | Audit ranges / notes |
| --- | ---: | --- |
| Uncoated appearance/reference | 158 | 13:18:08-13:33:17, five capture clusters; includes two broad clean rings plus close/detail references |
| Empty-board/background | 107 | 51 images at 13:35:28-13:36:52; 31 images at 14:37:26-14:38:25; 25 images at 14:46:36-14:47:19 |
| Object-bearing coated/marked geometry | 423 | six geometry passes: 142 + 72 + 55 + 59 + 58 + 37 images |
| Total | **688** | every supplied image is accounted for |

The exact mixed-pass boundaries were visually confirmed:

```text
G8 geometry ends: IMG20260912143718.jpg
G8 empty begins:   IMG20260912143726.jpg

G9 geometry ends: IMG20260912144629.jpg
G9 empty begins:   IMG20260912144636.jpg
```

The six object-bearing geometry pass ranges are:

```text
G7  IMG20260912141912.jpg  -> IMG20260912142404.jpg   142 files
G8  IMG20260912143336.jpg  -> IMG20260912143718.jpg    72 files
G9  IMG20260912144413.jpg  -> IMG20260912144629.jpg    55 files
G10 IMG20260912145120.jpg  -> IMG20260912145347.jpg    59 files
G11 IMG20260912150136.jpg  -> IMG20260912150348.jpg    58 files
G12 IMG20260912150514.jpg  -> IMG20260912150627_01.jpg 37 files
```

G7 has roughly twice the intended per-revolution frame count and visual recurrence evidence. Treat it as a capture pass containing one or more revolution wraps until deterministic phase/wrap analysis confirms the logical circles. Do not assign `142/360` degrees per frame or blindly split at index 71. G8-G12 visually behave as single ordered circles, but exact phase remains inferred rather than physically measured.

### 2.2 Camera and EXIF consistency

ExifTool 13.59 inspected every image. The complete set reports:

```text
width x height              = 3072 x 4080 for all 688
EXIF orientation            = 1 for all 688
Make                         = OPPO for all 688
Model                        = OPPO Reno12 F for all 688
LensModel                    = OPPO Reno12 F back camera 26mm f/1.8 for all 688
FocalLength                  = 3.98 mm for all 688
FocalLengthIn35mmFormat      = 26 mm for all 688
DigitalZoomRatio             = 1 for all 688
FNumber                      = f/1.8 for all 688
ExposureCompensation         = 0 for all 688
```

The geometry and empty-board captures are highly consistent: ISO 100, 1/100 s, fixed reported lens/focal state, and manual-exposure/manual-WB metadata. The 36 images with auto-exposure/auto-WB behavior belong to the early uncoated close/detail reference groups, not geometry. Focus-mode/focus-distance and color-temperature tags are not present, so focus/WB lock cannot be proven from those absent EXIF fields; visual continuity is the secondary evidence.

This strongly supports one shared intrinsic group across the selected V4 geometry views, subject to one implementation-time verification that no hidden crop/orientation state or sparse-calibration behavior contradicts it. Split camera groups only for evidence, not by ring by default. `SIMPLE_RADIAL` remains the first camera-model candidate, not a frozen requirement.

### 2.3 Exposure, sharpness, clipping, and framing

No unreadable/corrupt source was observed. No object-bearing geometry frame was found with catastrophic vessel cutoff or catastrophic motion blur in the all-frame review.

Representative downsampled Laplacian-variance distributions for the geometry-bearing clusters were healthy relative to the empty-board tails. The globally lowest sharpness scores belonged to empty-board frames, not vessel-bearing geometry. Geometry highlight clipping is low numerically even though the metal remains visibly reflective; the largest clipping values occur in early uncoated close-detail reference images.

The vessel remains centered with useful margin in all geometry passes. G11 is the closest broad geometry pass and still contains the complete vessel. Occasional hands/fingers appear near the image edge in a few geometry/reference frames; these are background intrusions and must never survive the vessel feature mask. They are not a reconstruction blocker.

### 2.4 Coating, markers, glare, and surface coverage

The dry-shampoo treatment reduced but did not eliminate brass reflectivity. Geometry images still show moving gold/specular highlights. Random black dots/dashes are dense and non-periodic over the outer bowl, globe/shoulder, pedestal/base, and much of the neck. They supply substantially better correspondence texture than the rejected polished capture.

Higher-risk regions are:

```text
finial ball and stepped/ridged tip
some polished neck/lid transition bands
bowl rim
visible bowl interior
base-contact boundary
locally glossy shoulder/globe regions
```

These regions have less marker support and/or more reflection. The reconstruction must use multi-ring geometric consistency and must not manufacture missing surface detail. The steep-high pass supplies important top/bowl/finial evidence.

### 2.5 Background and rotating-board risk

The white cloth backdrop is stationary and contains strong folds, seams, and tonal texture. Those features are invalid for rotating-object SfM.

More importantly, the wooden square board rotates with the vessel and is highly textured. Board leakage is more dangerous than cloth leakage because the board obeys the same relative rotation as the vessel and can create apparently consistent sparse tracks and dense geometry. A reconstruction can therefore look numerically well-connected while actually being board-supported.

The final set also contains unusually useful negative evidence: rotating empty-board sweeps. G8 and G9 contain same-setup empty-board suffixes after the vessel is removed. G6 is a 51-frame empty-board sweep from an earlier setup and may be paired with another setup only if camera/framing compatibility is proven, not assumed.

The segmentation design must use phase-compatible empty-board evidence when available to veto wood leakage near the foot/base. It must never turn the board itself into a foreground prior.

### 2.6 Appearance-reference evidence

The 158 uncoated images provide real brass appearance references. The early broad clean rings provide the most stable material evidence; later close/detail references provide ornament/detail context but include variable exposure, stronger clipping, and occasional tripod/green-sheet intrusion. Final material sampling must reject clipped specular pixels and unrelated background/green regions rather than averaging them into the brass material.

The coated/marked geometry photographs are not final material sources.

## 3. Actual capture topology and phase model

Do not assume the idealized four-ring acquisition literally occurred. The real media contains six object-bearing geometry passes and multiple empty-board sequences.

Use stable source pass IDs first:

```text
geo_g7
geo_g8
geo_g9
geo_g10
geo_g11
geo_g12
```

Then derive logical `ring_id` values only after wrap/revolution detection. Preserve the source pass in metadata so a logical split is auditable.

Each selected geometry frame records:

```text
source_pass_id
ring_id
provisional visual role/elevation class
frame_index_within_source_pass
frame_index_within_logical_ring
rotation_direction
phase_01 in [0, 1)
angle_deg = null unless genuinely measured
physical_camera_group
source_role = geometry
empty_board_sequence when compatible
source SHA-256
```

No turntable angle log exists. `phase_01` is therefore inferred from ordered visual phase, not treated as measured physical angle.

### 3.1 Revolution/wrap detection

For every object-bearing source pass:

1. Preserve filename/timestamp order as the initial sequence.
2. Detect near-duplicate adjacent/same-second frames before phase normalization.
3. Measure visual recurrence using geometry-safe cues from the vessel/board phase and ordered continuity.
4. Identify candidate 0/1 wrap points where the scene returns to an earlier phase.
5. Validate each candidate wrap against several neighboring frames; do not decide from one perceptual-hash match.
6. Split a pass into logical rings only when a complete repeated revolution is supported.
7. If a pass contains a partial extra revolution, keep useful unique phase coverage but do not interleave duplicate phases as though they were new angles.
8. G7 requires this procedure explicitly because 142 images are consistent with repeated phase coverage.

### 3.2 Near-duplicate policy

There are no byte-identical SHA-256 duplicate files. Ten same-timestamp `_01` pairs exist, and perceptual inspection found several near-duplicates; `IMG20260912143525.jpg` and `_01` are visually/perceptually identical at thumbnail hash level despite differing bytes.

Never delete source files. The selected-geometry manifest marks redundant views as `selected=false` with a reason and keeps their hashes. Prefer the sharper/cleaner member when two files carry no meaningful new angular evidence.

### 3.3 Cross-ring phase alignment

Different geometry passes do not share a guaranteed start angle. Do not pair cross-ring views by raw index.

Before production cross-ring matching:

1. use a sparse set of phase candidates between adjacent elevation rings;
2. evaluate mask-restricted learned match/inlier support;
3. estimate the circular phase offset that maximizes consistent vessel correspondences;
4. store the offset and confidence;
5. pair each frame with nearest phase neighbors in the adjacent ring, allowing a small neighborhood for uncertainty.

This is phase-offset estimation inside the approved ALIKED/LightGlue route, not a matcher comparison.

## 4. V4 directory and provenance contract

The actual source is already in place, so V4 does not need a second raw-media copy.

```text
CSX4213_Project_V4_Images/          # immutable final incoming source

capture_v4/
    manifests/
        source_manifest.csv
        sequences.json
        media_audit.json
        exclusions.csv
    derived/
        oriented/                    # only if orientation normalization is actually required
        masks/
        feature_masks/
        matching_images/
        mvs_images/
        undistorted_masks/
        previews/

reconstruction/v4/
    sparse/
        database.db
        best/
    dense/
        workspace/
        fused.ply
    mesh/
        poisson_raw.ply
        scan_clean_high.ply
        export_mesh.ply
    blender/
        Thai_Libation_Vessel_V4_FINAL.blend
        Thai_Libation_Vessel_V4_FINAL.glb
        textures/
    reports/
    previews/
    work/
        stage_state.json
```

Rules:

- `CSX4213_Project_V4_Images/` is read-only by policy.
- Every one of the 688 raw files is hashed and appears exactly once in `source_manifest.csv`.
- Every source has exactly one primary role: `geometry`, `empty_board`, or `appearance_reference`.
- Geometry selection is a separate boolean; exclusions never remove source provenance.
- V1-V3 derived media never enters V4 geometry.
- Canonical reconstruction images preserve original geometry unless a transformation is represented consistently in the camera model.
- Feature-only rescaling/cropping is allowed only if keypoints are mapped back exactly to original full-image coordinates before COLMAP import.
- All generated V4 products remain outside the immutable source directory.

## 5. Ingest and deterministic media manifest

The V4 set consists of still JPEGs; no frame extraction is needed. FFmpeg is installed but is not required for still ingestion. ExifTool is the metadata authority, with Pillow/OpenCV used for decoding and numerical QA.

`source_manifest.csv` records at least:

```text
relative_path
sha256
byte_size
width
height
format
orientation
make
model
lens_model
focal_length_mm
focal_length_35mm_eq
f_number
iso
exposure_time
exposure_compensation
digital_zoom_ratio
white_balance
exposure_program
datetime_original
source_cluster
source_role
source_pass_id
logical_ring_id
selected_for_geometry
selection_reason
near_duplicate_group
frame_index
phase_01
camera_group
```

The generated manifest must reconcile to the planning audit totals `158 + 107 + 423 = 688`. A mismatch stops ingest because it means the implementation is not operating on the audited final set.

Role classification starts from the approved timestamp/file ranges in Section 2 and verifies object presence/content rather than silently trusting timestamps.

Minimal QA records decode status, blur signal, luminance/contrast, clipping fraction, object cutoff, and duplicate/near-duplicate evidence. Exclude only unreadable/catastrophic frames or truly redundant near-duplicates. Warning-level geometry views remain eligible when they preserve phase continuity.

Because EXIF orientation is 1 for all images, do not rewrite 688 files merely to create “oriented copies.” Use originals directly for geometry unless a later decode inconsistency proves a derived normalization is necessary.

## 6. Object isolation and board-leakage suppression

### 6.1 Primary segmentation

Use pretrained Grounding DINO-T/Swin-T to localize the vessel and SAM 2.1 Hiera-small to produce full-resolution masks. No custom segmentation training is planned. Ordered-ring propagation may be used only while stable; refresh the detection/prompt when the mask visibly drifts.

Required preserved real geometry includes:

```text
finial ball and stepped tip
lid/top tiers
long neck and opening/lip
shoulder/globe
bowl rim and visible interior boundary
outer bowl
pedestal transitions/rings
base/foot
```

### 6.2 Three mask products

For each selected geometry image produce:

```text
full object mask   # preserves full vessel silhouette/support
feature mask       # conservative inward boundary safety where safe
MVS mask           # full vessel support for dense fusion
```

All masks retain source width/height and original pixel coordinates. The feature mask must not erode thin finial/rim geometry merely to obtain a cleaner edge.

### 6.3 Empty-board negative evidence

For G8/G9, use the same-setup empty-board suffix as a refinement/veto cue:

1. estimate board phase for each geometry image against empty-board candidates using only background/board regions outside the provisional vessel mask;
2. choose a compatible empty-board phase only when similarity/confidence is sufficient;
3. use the matched empty view to identify wood pixels that SAM incorrectly absorbed near the base;
4. treat this as negative evidence only; never add foreground from image differencing;
5. ignore hands or transient intrusions in empty/reference frames;
6. record the selected empty reference or `none` per geometry image.

For G6, first verify camera/framing compatibility before associating it with any geometry pass. If incompatible, keep it as provenance/reference only.

For geometry passes without a compatible empty sequence, use Grounded-SAM2 plus temporal silhouette consistency, connected-component checks, and base-contact diagnostics rather than inventing an empty reference.

### 6.4 Board-leak diagnostics

Each ring receives quantitative and visual checks:

```text
mask area / bbox / centroid continuity
largest connected-component fraction
frame-to-frame area jump
pixels extending below the visible foot/base
suspicious broad horizontal mask support at board level
feature-keypoint count outside accepted vessel mask = 0
manual overlay/contact-sheet review of every selected frame
```

Do not use mask dilation around the base if it absorbs wood grain. If the foot/base is ambiguous, preserve the real foot conservatively and prefer slight under-inclusion of board-contact pixels over a wide wood patch.

No white-threshold segmentation is a primary path.

## 7. Learned features and acquisition-aware pairing

The only matcher is ALIKED-N16Rot + LightGlue.

Reuse/generalize the proven primitives in `external_learned_recovery.py` for feature structures, extraction, caching, matching, COLMAP database import, and geometric verification. V4 must not inherit `expected_images=288`, Step-13 labels, old path/gate assumptions, or exactly-one-camera assertions as frozen legacy behavior.

### 7.1 Resource configuration

Source resolution is now known: 3072x4080 portrait for every image. Some geometry passes frame the vessel relatively small, so implementation must do one bounded CUDA resource preflight before bulk extraction:

1. inspect live VRAM;
2. run ALIKED on a representative distant geometry frame at the preferred practical inference size;
3. start with 3072 max image size if it fits comfortably; if it OOMs or leaves unsafe VRAM headroom, lock 2048 instead;
4. use `aliked-n16rot`, maximum 4096 keypoints, LightGlue `features=aliked`, CUDA device 0, mixed precision when supported, and adaptive LightGlue depth/width;
5. lock the chosen production configuration and do not sweep alternatives.

If the selected configuration yields abnormally low on-mask feature counts in a whole ring, diagnose mask/scaling first. Do not switch feature families.

### 7.2 Pair schedule by phase, not index

For each logical ring, create pairs using circular phase distance:

```text
nearest immediate phase neighbors
+ a wider local phase band for track length
+ explicit 0/1 wrap closure
```

Because rings have unequal frame counts (approximately 37-72 per normal pass, with G7 repeated coverage), do not use a fixed raw index offset as the meaning of angular proximity.

Across adjacent elevation rings:

```text
estimated circular phase offset
+ nearest corresponding phase
+ a small neighboring phase window
```

The schedule must tolerate removed near-duplicates and missing phases without breaking continuity. No self-pairs, duplicate unordered pairs, or exhaustive all-pairs matching.

Every keypoint imported into COLMAP must be inside the accepted feature mask. Every scheduled LightGlue match must pass COLMAP geometric verification before mapping.

## 8. Sparse SfM and camera strategy

Use pyCOLMAP incremental SfM plus bundle adjustment as the only sparse solver.

The actual EXIF strongly supports a shared intrinsic contract across selected geometry: same phone, back camera, 3.98 mm / 26 mm-equivalent focal length, digital zoom 1, 3072x4080, orientation 1, f/1.8. Therefore the planned default is one shared geometry camera/intrinsic group, subject to explicit verification of hidden crop state and stable bundle-adjustment behavior.

Do not create one camera per ring merely because the physical camera moved. Camera extrinsics change by view; shared intrinsics do not require shared physical pose.

Start with `SIMPLE_RADIAL`. If calibration becomes implausible or residual/distortion behavior clearly contradicts that model, make one evidence-backed camera-model correction rather than a broad model sweep.

Sparse acceptance requires all of the following:

```text
useful registration across all necessary logical rings
coherent virtual orbital/ring trajectories
cross-elevation connectivity
no gross pose jumps/explosions
sparse points visually concentrated on the vessel
no dominant wood-board plane/cloud
no stationary cloth-supported structure
recognizable vessel support sufficient for dense MVS
```

Add a board-contamination diagnostic using feature masks/projected observations and sparse-cloud visualization. A high registration count is not sufficient evidence of success.

## 9. Dense CUDA reconstruction

Only the accepted sparse model and selected geometry frames enter dense reconstruction.

Production path:

```text
COLMAP image_undistorter
-> transform vessel masks through the same accepted camera geometry
-> verify representative image/mask overlays
-> bounded CUDA PatchMatch smoke subset
-> full CUDA patch_match_stereo with geometric consistency
-> geometric stereo_fusion using aligned vessel masks
-> fused.ply
```

Before any full dense run, query the installed COLMAP 4.2.0 help for exact option spelling. Current verified executable reports:

```text
COLMAP 4.2.0 (Commit be5e291 on 2026-08-31 with CUDA)
```

Baseline production size remains 2000 pixels. The only resource fallback is 2000 -> 1600 after a diagnosed CUDA OOM.

The CUDA smoke check must actually execute PatchMatch on a bounded representative subset/workspace; a version string containing `with CUDA` is necessary but not sufficient runtime proof. The smoke subset includes views from more than one elevation and verifies that depth/normal output is produced without a CUDA/runtime failure.

Dense fusion uses only vessel masks aligned to the undistorted image geometry. Do not mask by simply resizing original binary masks if COLMAP undistortion changes projection.

### 9.1 Current post-fusion acceptance policy

At the current project checkpoint, do not launch another full PatchMatch run merely because a post-fusion ring-continuity report fails the implementation-only 1%/0.50 diagnostic. First correct the cross-camera depth comparison described in the continuation checkpoint and re-run the audit against the already-computed true3 geometric depth maps.

Dense acceptance is based on the authoritative evidence hierarchy:

1. the fused cloud is real output from the approved geometric PatchMatch/fusion route;
2. all intended fusion masks resolve under the actual COLMAP filename contract;
3. the cloud is finite/rank-3 and visibly represents the vessel;
4. direct and measured evidence shows no **dominant** board slab, curtain/background structure, or pedestal-board webbing;
5. corrected ring-continuity evidence does not reveal an obvious genuine structural break.

If those conditions are met, preserve/hash the accepted fused cloud and advance to Poisson. If condition 5 fails for a real visible transition, rerun only the smallest affected dense chunks with evidence-backed local/cross-ring source support. Preserve all previous maps/candidates; do not restart the entire dense workspace without a concrete cause.

## 10. Poisson mesh and raw-geometry gate

Use Poisson reconstruction only. Preserve:

```text
reconstruction/v4/dense/fused.ply
reconstruction/v4/mesh/poisson_raw.ply
```

as immutable accepted upstream evidence.

Start from Poisson depth 13 only if the actual fused point density supports it; choose one appropriate depth, not a sweep.

Before Blender cleanup, inspect front/quarter/side/top-oblique views of both fused cloud and raw Poisson surface. The real reconstruction must plausibly contain the bowl/interior/rim, globe/shoulder, neck, lid/finial, pedestal transitions, and base. A technically valid mesh that is board-shaped or identity-poor fails this gate.

If completing the object would require inventing most of the vessel, stop at this gate and report the actual reconstruction limitation. Do not replace failed scan evidence with a synthetic/reference-modeled vessel.

## 11. Blender finalization

Blender begins only after the raw-geometry gate passes. Read the preferred Blender skill bundle first and use live Blender capabilities rather than assuming a legacy schema.

Scene contract:

```text
COL_V4_Source
    SM_V4_Poisson_Raw
COL_V4_Work
    SM_V4_Scan_CleanHigh
COL_V4_Final
    SM_V4_Vessel_LOD0
COL_V4_Lookdev
    CAM_V4_Lookdev
    LGT_V4_*
```

Cleanup sequence:

1. Import and preserve the raw Poisson mesh unchanged.
2. Capture raw review views before destructive cleanup.
3. Duplicate to `SM_V4_Scan_CleanHigh`.
4. Remove only unsupported floating/noise components.
5. Fix normals, duplicate vertices, impossible internal debris, and straightforward scan defects.
6. Repair holes only where surrounding observed surface makes the repair defensible.
7. Use controlled voxel remesh only when necessary for porous/noisy regions, followed by shrinkwrap/projection toward accepted scan evidence.
8. Use local sculpt/relax only to restore evidence-supported continuity, never to redesign from memory.
9. Preserve bowl interior/rim, globe, shoulder transitions, neck/opening, lid/finial, pedestal rings, and base as distinct real forms.
10. Build `SM_V4_Vessel_LOD0`, nominally about 250k-350k triangles unless evidence warrants another budget.
11. UV unwrap and bake normal/AO only where reduction loses useful scan detail.

Absolute real-world scale is not allowed to block completion. First search existing project evidence for a reliable measurement. If none exists, preserve a documented normalized/consistent scale and mark absolute dimensions unverified rather than inventing a measurement or requesting image recapture.

## 12. Brass appearance and material

The final V4 set already contains sufficient uncoated material/reference imagery to avoid using the marked geometry frames as the final texture source.

Use the broad uncoated clean rings for stable brass color/roughness context. Use close/detail references only after masking out background, green-sheet/tripod intrusion, and clipped/specular pixels. Do not project black markers or dry-shampoo residue into the final material.

Create one primary Principled brass material:

```text
MAT_V4_Brass
Metallic = 1.0 baseline
Base Color = robustly sampled/reference-supported uncoated brass
Roughness = restrained reference-supported variation
Normal/AO = accepted scan/detail bake where useful
Wear/variation = only where supported by clean references
```

No invented decorative texture is required to hide reconstruction defects.

## 13. Runtime/tool evidence

Planning-session verification on 2026-09-12 established:

```text
ExifTool 13.59
FFmpeg 9.0.1 full build
COLMAP 4.2.0, official Windows CUDA build, commit be5e291
required COLMAP commands present: mapper, bundle_adjuster, geometric_verifier,
image_undistorter, patch_match_stereo, stereo_fusion, poisson_mesher
CodeGraph provider available and current
```

Earlier verified Python state supplied by the project handoff is:

```text
Python 3.14.2
PyTorch 2.13.0+cu130
CUDA available through PyTorch
NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB
pyCOLMAP 4.2.0
```

Implementation must refresh one bounded runtime snapshot rather than assuming these versions are unchanged. Do not treat `nvidia-smi`/NVML failure alone as proof CUDA is unusable when PyTorch/COLMAP runtime checks succeed.

## 14. Restart/checkpoint strategy

`reconstruction/v4/work/stage_state.json` records:

```text
stage
status = pending | running | complete | failed
input hashes / manifest hash
configuration fingerprint
tool/model identities
output paths and hashes
start/completion timestamps
failure category
retry decision
```

Downstream-only invalidation is mandatory. Reuse a completed stage only when source/config/tool/output identities still match. Preserve valid ALIKED features, LightGlue matches, sparse model, and completed PatchMatch views across interruptions when their upstream fingerprints are unchanged.

Never delete valid upstream evidence merely because a later stage fails.

## 15. Verification gates

Only evidence-bearing gates that prevent wasted work are mandatory:

1. **Manifest/media gate:** all 688 files hash/decode and reconcile to 158 appearance + 107 empty + 423 object-bearing geometry before optional near-duplicate selection; logical rings/revolution wraps are documented.
2. **Mask/board gate:** complete vessel retained; cloth, board, hands, and other background excluded; no broad wood patch at the base.
3. **Sparse gate:** coherent cross-connected virtual rings; vessel-centered points; no dominant board/background reconstruction.
4. **CUDA dense gate:** a real bounded PatchMatch CUDA smoke succeeds before full dense execution.
5. **Raw geometry gate:** fused cloud and raw Poisson mesh are visually plausible before Blender.
6. **Blender gate:** canonical `.blend` opens with intended source/work/final collections and material.
7. **Portability gate:** final GLB cleanly re-imports with geometry, material/textures, sane transforms/bounds, and no gross shading failure.

Focused deterministic tests cover source-path immutability, source-role accounting, mask coordinate preservation, off-mask keypoint exclusion, board-leak checks where deterministic, revolution/phase handling, circular pair determinism, and downstream stage invalidation. Do not run broad V1-V3 regression suites unless a specific reused helper requires a focused compatibility check.

## 16. CodeGraph-informed reuse and blast radius

CodeGraph 1.6.0 is current. Existing navigation identified these reusable seams:

```text
external_learned_recovery.build_verified_match_database
external_learned_recovery.generate_sequential_pairs
local_reconstruction.run_command
local_reconstruction_io PLY helpers
```

Legacy assumptions to remove/generalize on V4-facing paths include:

```text
expected_images = 288
>=274 registration gates
Step-13 labels/contracts
exactly one-camera assertions used as old acceptance logic
reconstruction/local_dense fixed output policy
```

Before modifying reusable code, inspect CodeGraph callers/references/tests and then exact source. CodeGraph is structural navigation evidence, not runtime proof.

## 17. Observed risk register and required mitigation

| Risk | Actual evidence | Required mitigation |
| --- | --- | --- |
| Wood-board contamination | rotating board is richly textured and object-consistent | strict vessel masks; phase-compatible empty-board negative evidence; sparse/dense board-contamination gate |
| Static cloth contamination | folds/seams visible in every setup | mask before ALIKED; mask dense fusion |
| Remaining brass glare | coating did not fully matte object | geometric consistency; multiple elevations; no texture-based geometry invention |
| Sparse marker areas | finial/rim/interior/transitions less marked | retain high-angle evidence; do not over-erode mask; inspect raw coverage |
| Repeated revolution(s) | G7 has 142 frames and recurrence evidence | explicit wrap detection; logical-ring split/dedup before phase assignment |
| Unequal ring counts | geometry passes range 37-142 source frames | phase-distance pairing, not fixed index offsets |
| Unknown cross-ring start angle | no physical angle log | learned-inlier circular phase-offset estimation |
| Near-duplicates | same-second `_01` pairs; at least one perceptual duplicate | keep source, exclude redundant selected views deterministically |
| Edge hands/intrusions | a few geometry/reference frames show hands at image edge | vessel-only mask; ignore transient background in empty/refinement matching |
| Material reference clipping | early close clean refs have stronger highlight clipping | robust material sampling; reject clipped/background pixels |

## 18. Remaining data-dependent decisions

The final-media audit resolves image count, source resolution, source roles, clean-reference availability, and high-angle coverage. The following remain implementation-time decisions because they require computed evidence rather than visual guessing:

```text
exact G7 revolution wrap point(s) and whether any partial repeated cycle is retained
rotation direction per logical ring
exact phase_01 values after near-duplicate handling
cross-ring circular phase offsets/confidence
whether G6 empty-board sweep is camera-compatible with a geometry pass
final ALIKED max image size: 3072 if bounded CUDA preflight is safe, otherwise 2048
final camera model after sparse calibration behavior
Poisson depth if real fused point density makes depth 13 unsuitable
absolute physical scale if a reliable existing measurement is found
```

Everything else is fixed by this design. Do not request recapture; do not reopen algorithm selection for ordinary quality problems. Use the supplied final media and diagnose the approved pipeline stage that fails.
