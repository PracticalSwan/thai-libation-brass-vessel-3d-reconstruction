# Steps 14-17 Local Dense Reconstruction, Mesh, and Texture Design

**Status:** Approved for implementation planning on 2026-09-06
**Canonical project:** `C:\Assumption University\CSX4213\Project`
**Baseline commit:** `eef926c9d47db9f42a2b363cee77d68edb298a08`
**Execution scope:** implement Steps 14-17 in one continuous run after reading this design and the integrated implementation plan.

## 1. Purpose

Steps 14-17 convert the already-selected **Step 10 local sparse reconstruction** into the best defensible local dense model, mesh, and photo-textured asset that the current capture arc can support.

The accepted downstream sparse source is fixed:

```text
reconstruction/sparse/best
registered images     73 / 288
sparse points         6,099
observations          21,351
mean reproj. error    1.2373052447638215 px
camera model          SIMPLE_RADIAL
camera params         3542.7206959261907, 1536.0, 2040.0, -0.01641661056124677
```

The retained 266-image Step 13 model remains evidence only and shall not silently become the dense source. Sparse-recovery experimentation is closed.

The four implementation phases are:

1. **Step 14 — Local dense preparation and acceptance contract**
2. **Step 15 — Dense stereo and fusion**
3. **Step 16 — Mesh reconstruction and bounded cleanup**
4. **Step 17 — Photo texturing**

Blender artistic/manual cleanup is a later phase. Step 17 may use Blender only for an automated non-destructive validation render if needed; it shall not begin manual modeling or aesthetic cleanup.

## 2. Current capability evidence

Planning inspection on 2026-09-06 established:

```text
Python                  3.14.2
pyCOLMAP                 4.2.0
pycolmap.has_cuda        false
GPU previously verified NVIDIA GeForce RTX 5050 Laptop GPU
Blender                  5.2 installed at C:\Program Files\Blender Foundation\Blender 5.2\blender.exe
COLMAP CLI in PATH       absent
OpenMVS in PATH          absent
```

The installed pyCOLMAP exposes:

```text
undistort_images
patch_match_stereo       # requires CUDA
stereo_fusion
poisson_meshing
simplify_mesh
```

However, this Windows pyCOLMAP wheel has no CUDA support, so its `patch_match_stereo` path is not expected to execute successfully on the live environment.

COLMAP 4.2.0 officially provides Windows binaries, CUDA dense stereo, Poisson/Delaunay/advancing-front meshing, mesh simplification, and `mesh_texturer`. The implementation may use the official **COLMAP 4.2.0 Windows CUDA command-line distribution** as the dense backend while retaining pyCOLMAP as the sparse-model authority and Python orchestration layer.

Primary official references:

```text
https://colmap.github.io/tutorial
https://colmap.github.io/cli.html
https://colmap.github.io/install.html
https://colmap.github.io/changelog.html
https://github.com/colmap/colmap/releases/tag/4.2.0
```

## 3. Design principles

### 3.1 Frozen invariants

These are not flexible:

- Raw photographs under `IMG20260826122949/` remain immutable.
- Selected PREPROCESSED images under `preprocessing/pycolmap_input/images/` remain immutable.
- Step 10 sparse evidence under `reconstruction/sparse/` is read-only.
- Step 11/12/13 evidence is read-only.
- The dense source is exactly the 73 registered image names from `reconstruction/sparse/best`.
- Do not re-run sparse extraction, sparse matching, incremental mapping, or learned recovery.
- Do not substitute the 266-image Step 13 model because it appears visually or numerically attractive.
- Do not fabricate depth maps, dense points, mesh geometry, UVs, textures, renders, or metrics.
- Do not manually edit measured reports to change a gate outcome.
- Do not commit model caches, downloaded tool archives, transient depth maps, or private CNN checkpoints.
- Do not begin manual Blender cleanup/modeling in Steps 14-17.

### 3.2 Flexibility envelope

The implementation is deliberately allowed bounded engineering flexibility because dense MVS is hardware- and scene-sensitive.

The agent may:

- choose the simplest supported COLMAP 4.2 dense invocation path after a live capability probe;
- use pyCOLMAP for undistortion/meshing where it works, or the official COLMAP 4.2 CLI for the full dense chain when that keeps the workflow simpler and more reproducible;
- install/extract an official COLMAP 4.2 Windows CUDA binary outside the repository if no working dense executable already exists;
- choose a practical dense `max_image_size` based on the RTX 5050 8 GB VRAM and measured runtime behavior;
- reduce dense resolution once if the first practical configuration fails from GPU memory, TDR, or resource pressure;
- reduce the number of PatchMatch source views once if needed for memory/runtime stability;
- disable PatchMatch geometric consistency only as a bounded fallback when the preferred geometrically consistent run is blocked by resource/runtime limitations; filtering must remain enabled if this fallback is used;
- choose Poisson or Delaunay as the primary mesh based on the actual fused cloud and visual evidence, with at most one alternative mesh candidate retained for comparison;
- use mesh simplification only when it materially improves downstream usability without visibly destroying vessel geometry;
- choose texture atlas resolution/packing parameters based on actual mesh size, VRAM/RAM, and image coverage;
- use one controlled retry per expensive phase when the first result is unusable for a clearly measured reason.

The agent shall **not** turn this flexibility into parameter sweeps, repeated trial-and-error, or matcher/reconstruction experimentation.

## 4. Proposed architecture

Add small, explainable Python modules rather than one large script:

```text
local_dense_reconstruction.py     capability, sparse-source validation, dense commands, PLY metrics
run_local_dense_reconstruction.py staged Steps 14-15 orchestration and reports
local_mesh_reconstruction.py      mesh metrics, candidate ranking, simplification checks
run_local_mesh_reconstruction.py  Step 16 orchestration and reports
local_texturing.py                texturer invocation, texture/UV/material validation
run_local_texturing.py            Step 17 orchestration and reports
```

Exact file split may be adjusted if a smaller coherent module boundary is clearer. Do not create abstractions merely to satisfy this sketch.

Recommended tests:

```text
tests/test_local_dense_reconstruction.py
tests/test_run_local_dense_reconstruction.py
tests/test_local_mesh_reconstruction.py
tests/test_run_local_mesh_reconstruction.py
tests/test_local_texturing.py
tests/test_run_local_texturing.py
```

If a runner/domain split would create trivial wrappers, combine the tests/modules rather than adding ceremony.

## 5. Artifact layout

All new reconstruction outputs live under:

```text
reconstruction/local_dense/
```

Recommended durable layout:

```text
reconstruction/local_dense/
  dense/
    fused.ply
  mesh/
    primary_mesh.ply
    final_mesh.ply
    optional_alternative_mesh.ply
  texture/
    ... final texturer outputs ...
  reports/
    step14_capability.json
    step14_source_manifest.csv
    step14_summary.json
    step15_dense_attempts.json
    step15_summary.json
    step16_mesh_attempts.json
    step16_summary.json
    step17_texture_summary.json
    steps14_17_summary.json
  previews/
    step14_01_local_sparse_source.png
    step15_01_dense_cloud.png
    step16_01_mesh.png
    step16_02_mesh_comparison.png       # only if there are two real candidates
    step17_01_textured_preview.png
```

Heavy transient workspace:

```text
reconstruction/local_dense/work/
```

The work directory may contain:

- undistorted images;
- stereo configuration;
- depth maps;
- normal maps;
- consistency graphs;
- temporary meshes;
- downloaded/extracted tool-local runtime metadata;
- temporary render files.

The `work/` tree is not automatically a publication artifact. Preserve it only while needed to continue Steps 14-17 or to diagnose a failed stage. After durable outputs and reports are verified, remove only transient files that are no longer required.

## 6. Step 14 — Local dense preparation and acceptance contract

### 6.1 Source contract

Step 14 shall reopen `reconstruction/sparse/best` and derive the exact registered image names from the model itself.

Expected measured source:

```text
73 registered images
first capture: IMG20260826122949.jpg
last capture:  IMG20260826123824.jpg
```

Do not assume the registered set is exactly sequence indices 1-73 without verifying the names from the model.

### 6.2 Capability gate

Step 14 shall record:

- pyCOLMAP version and `has_cuda`;
- available pyCOLMAP dense functions;
- GPU/CUDA runtime evidence;
- detected COLMAP CLI path/version if present;
- whether that CLI exposes `image_undistorter`, `patch_match_stereo`, `stereo_fusion`, mesh commands, and `mesh_texturer`;
- Blender path/version as an optional validation tool, not the main reconstruction backend;
- exact dense backend chosen and why.

### 6.3 Dense backend policy

Preferred order:

1. **Working COLMAP 4.2 CUDA CLI** — preferred on this Windows machine because the current pyCOLMAP wheel has `has_cuda=false` and COLMAP 4.2 includes the complete dense-to-texture command chain.
2. **CUDA-enabled pyCOLMAP 4.2** — acceptable if the environment changes and a verified build becomes available without destabilizing the project.
3. **Official COLMAP 4.2 CUDA Docker image** — allowed only if Docker GPU passthrough is already working and this is simpler than installing the Windows binary.

Do not introduce OpenMVS, Meshroom, RealityCapture, proprietary cloud reconstruction, or another MVS stack unless all COLMAP 4.2 paths are genuinely blocked and a separate documented architecture decision is required.

### 6.4 Preparation outputs

The dense workspace shall be generated only from:

```text
sparse model: reconstruction/sparse/best
images:       preprocessing/pycolmap_input/images/<registered names only>
```

Preferred undistortion output type is COLMAP workspace format.

### 6.5 Step 14 hard acceptance

Step 14 passes only if:

- Step 10 model reopens with the expected 73-image / 6,099-point metrics;
- every registered image exists and matches the frozen selected-image manifest;
- the dense workspace contains exactly the 73 registered views, not all 288;
- camera/intrinsics remain consistent with the source model;
- undistorted images are readable;
- no source image is modified;
- a backend capable of PatchMatch stereo is proven available before Step 15 begins.

If these cannot be established, Steps 15-17 must not generate fake partial outputs.

## 7. Step 15 — Dense stereo and fusion

### 7.1 Preferred dense configuration

Start with a conservative quality/performance configuration appropriate for an 8 GB laptop GPU rather than full 3072x4080 dense processing.

Suggested starting range:

```text
PatchMatch max_image_size     1600-2000
StereoFusion max_image_size   same or compatible
geom_consistency              true
PatchMatch source images      automatic; bounded if memory requires it
GPU index                     0 unless runtime detects otherwise
```

The exact first value may be selected after the capability probe. Record it before execution.

### 7.2 One controlled fallback

A single fallback is allowed if the preferred attempt fails due to OOM, Windows GPU timeout/TDR, or clearly excessive resource pressure.

Examples of allowed fallback changes:

- reduce `max_image_size` one level, e.g. 2000 -> 1600 or 1600 -> 1200;
- reduce source views, e.g. automatic 30 -> automatic 10-20;
- disable geometric consistency while retaining PatchMatch filtering.

Choose the smallest change that addresses the measured failure. Do not sweep combinations.

### 7.3 Dense metrics

At minimum report:

- number of reference images requested;
- number of depth maps produced/readable;
- number of normal maps produced/readable where exposed;
- fusion runtime;
- fused point count;
- finite XYZ count/fraction;
- color availability;
- bounding-box extents;
- nearest-neighbor/sample spacing summary if feasible without a new heavy dependency;
- dense/sparse point ratio;
- exact attempt configuration;
- exact backend version/path.

### 7.4 Dense acceptance

Hard gates:

- fused cloud exists and reopens;
- all stored XYZ values are finite;
- fused cloud contains more points than the 6,099-point sparse source;
- point cloud has non-zero spatial extent in X/Y/Z;
- real color or other backend-supported appearance data is present when expected;
- no catastrophic coordinate explosion relative to the source sparse model;
- a visual preview shows a coherent vessel-related structure rather than only noise/background.

Quality target, not an absolute hard gate:

```text
>= 10x sparse point count (~60,990 points)
```

A coherent but smaller dense cloud may continue to Step 16 if the hard gates pass and the report explains why the target was not met. A huge but obviously invalid point cloud shall fail despite its point count.

## 8. Step 16 — Mesh reconstruction and bounded cleanup

### 8.1 Candidate policy

Primary choice should be evidence-driven:

- **Poisson** is preferred when the fused cloud has reasonably complete normals and a smooth vessel surface is the priority.
- **Delaunay** is preferred when visibility-aware filtering better suppresses background/outlier surfaces or Poisson closes unsupported regions excessively.

At most two meaningful mesh candidates may be generated for comparison. This is not a mesh-parameter sweep.

### 8.2 Cleanup policy

Allowed bounded cleanup:

- keep the dominant vessel-supporting connected component if the separation is algorithmically defensible;
- remove isolated tiny components using an explicit size rule derived from the actual component distribution;
- simplify with QEM when the original mesh is unnecessarily dense for texturing/downstream Blender use;
- preserve colors/attributes where supported;
- preserve the original unsimplified primary candidate for evidence.

Not allowed:

- hand-sculpting missing vessel geometry;
- arbitrary hole filling that invents unsupported surfaces;
- manually deleting inconvenient geometry without an explicit reproducible rule;
- smoothing until vessel features are materially altered.

### 8.3 Mesh metrics

Report:

- vertices;
- triangular faces;
- finite coordinate fraction;
- connected-component count and face distribution if practical;
- mesh bounding box;
- dense-cloud versus mesh bounding-box relationship;
- simplification ratio if used;
- file size;
- algorithm/options;
- visual review result.

### 8.4 Mesh acceptance

Hard gates:

- mesh file exists and reopens;
- non-zero vertices and faces;
- finite geometry;
- one dominant component corresponds visually to the vessel reconstruction;
- mesh does not collapse to a plane/line/point;
- mesh scale/bounds are plausible relative to the fused cloud;
- no obvious giant artificial enclosing shell is accepted as the final mesh.

Soft target:

- retain enough triangles to preserve vessel silhouette and major surface detail while remaining practical for Step 17 texturing and later Blender cleanup.

The final mesh is chosen by validity and visual plausibility first, not by maximum face count.

## 9. Step 17 — Photo texturing

### 9.1 Preferred path

Use COLMAP 4.2 `mesh_texturer` when the selected dense backend exposes it. This keeps geometry, calibrated cameras, undistorted images, and texturing in one toolchain.

The texturer shall consume:

```text
workspace: Step 14/15 COLMAP dense workspace
mesh:      Step 16 final mesh
images:    calibrated undistorted local views
```

### 9.2 Flexibility

The agent may adjust texturing parameters such as atlas size, downsampling, or seam-related options based on the actual mesh and available resources.

At most one controlled retry is allowed if the first attempt fails for a measurable memory/packing/configuration reason.

If `mesh_texturer` is unexpectedly unavailable in the verified COLMAP 4.2 binary, a narrow fallback may use Blender 5.2 in **headless scripted mode** to project/bake the same calibrated local photographs. Such use is limited to automated UV/texture generation and preview validation; manual Blender cleanup remains Step 18/later.

### 9.3 Texture validation

Validate the backend's actual output contract rather than assuming a specific extension. Where applicable verify:

- textured mesh reopens;
- UV coordinates exist;
- material references resolve;
- texture atlas files exist and are readable;
- texture dimensions are sensible;
- no missing material/texture path;
- mapped texture coverage is substantial rather than a tiny accidental patch;
- a real preview render shows image-derived appearance aligned with the mesh.

### 9.4 Texture acceptance

Hard gates:

- geometry survives texturing unchanged except for format conversion/UV data;
- at least one valid texture/material asset is referenced by the final textured mesh;
- texture image(s) reopen successfully;
- UV/material assignment is non-empty;
- visual preview shows recognizably photo-derived vessel appearance;
- texture artifacts/limitations caused by reflective brass are documented rather than hidden.

If the geometry is valid but photo texturing is impossible because of backend capability, the step is incomplete and must be reported as such; do not rename vertex colors as a completed photo texture unless the plan is explicitly revised.

## 10. Unified reporting

Every stage writes a machine-readable report before downstream execution.

Final integrated summary:

```text
reconstruction/local_dense/reports/steps14_17_summary.json
```

It shall include:

- sparse source identity and immutable hash/provenance;
- exact registered-image list hash;
- dense backend/version/path;
- capability result;
- selected dense attempt and any one fallback reason;
- dense metrics;
- selected mesh algorithm and metrics;
- simplification decision;
- texture backend and output manifest;
- visual-status fields per stage;
- known limitations;
- next downstream artifact path;
- `blender_manual_cleanup_started=false`.

## 11. Staged runner semantics

Recommended top-level stages:

```text
prepare
stereo
mesh
texture
finalize
all
```

`all` may execute Steps 14-17 continuously, but it must respect the hard gate between stages.

```text
prepare fail -> stop
prepare pass -> stereo
stereo fail after allowed fallback -> stop
stereo pass -> mesh
mesh fail after allowed alternative -> stop
mesh pass -> texture
texture fail after allowed retry -> truthful partial result
texture pass -> finalize
```

A failed upstream stage must never be hidden by generating placeholder downstream assets.

## 12. Verification strategy

Use TDD for deterministic contracts, parsers, path safety, command construction, metrics, report logic, and stage gates.

Do not mock the entire real pipeline and then claim runtime success. Real dense/MVS/mesh/texture execution is required before completion.

Verification should include:

- focused new tests;
- relevant Step 10-13 regression tests;
- full project tests after implementation stabilizes;
- Python compilation;
- raw/selected image integrity verification;
- Step 10 protected model/report integrity;
- real artifact reopening;
- visual inspection of every final preview;
- transient/cache cleanup check;
- Git diff and publication preflight if publication is authorized.

## 13. Publication/storage policy

Do not automatically commit the full dense workspace. Depth maps and intermediate MVS files can be very large.

Default publication candidates:

- source/tests;
- design/plan/measured docs;
- compact JSON/CSV reports;
- final preview images;
- final fused point cloud if reasonably sized;
- final mesh if reasonably sized;
- final textured asset only if its total size is appropriate for the repository.

If final geometry/textures are large, keep them local and commit a manifest/report with exact hashes, dimensions/counts, and reproducibility instructions instead of forcing large binary assets into Git.

Private/local-only paths remain excluded:

```text
.codegraph/
analysis/ml/checkpoints/
model/download caches
external tool archives/install trees
reconstruction/local_dense/work/
```

## 14. EARS-style requirements

### Universal

- U-1401: The pipeline **shall** use `reconstruction/sparse/best` as the only dense sparse source.
- U-1402: The pipeline **shall** derive the dense image set from the sparse model's registered image names.
- U-1403: The pipeline **shall** preserve raw and selected image bytes unchanged.
- U-1404: The pipeline **shall** record exact backend versions and options for every expensive reconstruction stage.
- U-1405: The pipeline **shall** retain measured failure evidence instead of fabricating downstream success.

### State-driven

- S-1401: **When** the current pyCOLMAP runtime has no CUDA PatchMatch support, the pipeline **shall** select a verified supported COLMAP CUDA backend before dense stereo.
- S-1501: **If** the preferred dense attempt fails from resource pressure, the pipeline **may** run one bounded fallback with a smaller resource footprint.
- S-1601: **If** the primary mesh is visibly invalid, the pipeline **may** generate one alternative COLMAP mesh candidate and choose by measured validity/visual plausibility.
- S-1701: **If** COLMAP mesh texturing is unavailable despite a valid mesh, the pipeline **may** use headless Blender texturing as one narrow fallback without starting manual cleanup.

### Unwanted behavior

- N-1401: The pipeline **shall not** restart sparse-recovery experimentation.
- N-1402: The pipeline **shall not** use the Step 13 266-image model as dense input.
- N-1501: The pipeline **shall not** perform unbounded dense parameter sweeps.
- N-1601: The pipeline **shall not** hand-sculpt or invent unsupported vessel geometry.
- N-1701: The pipeline **shall not** label vertex colors alone as completed photo texturing unless the design is explicitly revised.
- N-1702: The pipeline **shall not** begin manual Blender cleanup/modeling in Steps 14-17.

### Agent requirements

- A-1401: The agent **shall** recover the live project state and read the full Steps 14-17 plan before implementation.
- A-1402: The agent **shall** choose within the documented flexibility envelope and record every deviation from the suggested defaults.
- A-1403: The agent **shall** continue automatically through Steps 14-17 when gates pass and shall not stop for routine approvals.
- A-1404: The agent **shall** stop only at a genuine blocker, destructive authorization boundary, or hard acceptance failure that cannot be resolved within the permitted fallback.
- A-1405: The agent **shall** perform spec-compliance review before code-quality review and final verification.

## 15. Completion definition

Steps 14-17 are complete only when:

- Step 14 has a verified 73-view dense workspace and working dense backend;
- Step 15 has a real accepted fused dense cloud;
- Step 16 has a real accepted final mesh;
- Step 17 has a real accepted photo-textured asset;
- all reports match real artifacts;
- visual evidence has been inspected;
- protected inputs/evidence remain unchanged;
- transient work is bounded/cleaned appropriately;
- documentation clearly states that the result represents the **local 73-image capture arc**, not the complete 288-image sequence;
- manual Blender cleanup has not yet started.
