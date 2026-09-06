# Steps 14-17 Local Dense Reconstruction, Mesh, and Texture Implementation Plan

> Execute Steps 14-17 as one continuous implementation after recovering the live repository state. This plan intentionally contains a bounded flexibility envelope for dense-resolution, backend invocation, mesh choice, and texturing parameters. Use that flexibility to solve real runtime constraints, not to run parameter sweeps.

**Goal:** Starting from the frozen Step 10 local sparse source, produce and verify one defensible local dense point cloud, one final mesh, and one photo-textured asset, with truthful reports and real visual evidence.

**Design:** `docs/superpowers/specs/2026-09-06-steps-14-17-local-dense-mesh-texture-design.md`

**Baseline:** `eef926c9d47db9f42a2b363cee77d68edb298a08` on `main`.

## Project mapping

The user's requested Steps 1-4 map to project Steps 14-17:

| User step | Project step | Outcome |
| --- | --- | --- |
| 1 | Step 14 | verified local sparse source, dense capability/backend, 73-view undistorted workspace, explicit local acceptance contract |
| 2 | Step 15 | accepted fused dense point cloud from the local capture arc |
| 3 | Step 16 | accepted final mesh with bounded reproducible cleanup/simplification |
| 4 | Step 17 | accepted photo-textured mesh/material/atlas assets |

## Frozen inputs and boundaries

```text
Dense sparse source                 reconstruction/sparse/best
Expected registered images          73
Expected sparse points              6,099
Expected observations               21,351
Expected mean reprojection error    1.2373052447638215 px
Expected camera                     1 x SIMPLE_RADIAL
Selected image root                 preprocessing/pycolmap_input/images/
Selection manifest SHA-256          79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91
Raw image count                     297 immutable JPEGs
Step 13 evidence model              read-only, never dense input
Sparse recovery                     closed
Manual Blender cleanup              outside Steps 14-17
```

Do not modify or rewrite Step 10-13 evidence.

## Flexibility rules

The following are **allowed implementation decisions** without returning for routine approval:

1. **Dense backend:** use the simplest verified COLMAP 4.2 CUDA path. Current pyCOLMAP has `has_cuda=false`, so an official COLMAP 4.2 Windows CUDA binary is expected to be the practical primary path. A CUDA-enabled pyCOLMAP 4.2 or official COLMAP CUDA Docker path is also acceptable if already functional and simpler.
2. **Dense resolution:** choose a first `max_image_size` in the 1600-2000 range after live capability/VRAM inspection. One lower-resource retry is allowed.
3. **PatchMatch resource fallback:** one retry may lower resolution and/or source-view count; geometric consistency may be disabled only if resource pressure is the proven blocker and filtering remains enabled.
4. **Mesher:** choose Poisson or Delaunay as primary from real dense-cloud evidence. At most one alternative mesh candidate may be generated if the primary is invalid or clearly worse.
5. **Simplification:** simplify only when the mesh is impractically dense or materially harder to texture. Preserve the original primary mesh.
6. **Texture parameters:** choose atlas/packing/downsampling settings based on the final mesh and runtime resources. One texturing retry is allowed for a measured packing/memory/configuration failure.
7. **Code organization:** the design's suggested module split is not mandatory if a smaller and clearer implementation results.

The following are **not flexible**:

- no sparse retry/rematching/remapping;
- no Step 13 dense source substitution;
- no OpenMVS/Meshroom/proprietary cloud fallback without a separately documented architecture decision;
- no repeated tuning/sweeps;
- no manual geometry invention;
- no manual Blender cleanup/modeling;
- no fake placeholder downstream artifacts after an upstream gate fails.

---

# Phase 0 — Recover live state and freeze protection baseline

## Task 0.1 — Recover repository state

1. Read `AGENTS.md`, `CLAUDE.md`, `LESSONS.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md`.
2. Read this plan and the Steps 14-17 design in full.
3. Confirm Git root, branch, `HEAD`, `origin/main`, and working-tree state.
4. Preserve intentional local-only state:
   - `.codegraph/`
   - `analysis/ml/checkpoints/`
5. Inspect current dependencies and runtime without installing anything yet.

**Acceptance:** implementation starts from the actual current repository rather than a stale handoff.

## Task 0.2 — Freeze protected evidence

Record pre-implementation hashes/metrics for:

```text
reconstruction/reports/step10_summary.json
reconstruction/sparse/best/*
reconstruction/bridging/reports/step11_summary.json
reconstruction/learned_recovery/reports/step12_summary.json
reconstruction/external_learned_recovery/reports/step13_summary.json
```

Reverify:

- raw manifest: 297/297 unchanged;
- selected manifest: 288/288 exact;
- Step 10 model: 73 images / 6,099 points / one SIMPLE_RADIAL camera / 1.2373052447638215 px.

Extract the exact 73 registered filenames from the Step 10 model and persist a deterministic list hash for later reports.

**Acceptance:** protected source/evidence is internally consistent before any dense output is created.

---

# Step 14 — Local dense preparation and acceptance contract

## Task 14.1 — Add deterministic local-dense domain contracts test-first

Create the smallest coherent domain module(s) and tests for:

1. fixed sparse source path and expected Step 10 metrics;
2. exact registered-name extraction from a pyCOLMAP reconstruction;
3. registered-name list hashing;
4. safe workspace-boundary checks;
5. dense backend capability representation;
6. command argument construction without shell-string injection;
7. subprocess result capture/timeout/error reporting;
8. output-manifest hashing;
9. PLY header parsing and basic point/mesh metrics where feasible without a new dependency;
10. deterministic report serialization;
11. hard stage-gate functions;
12. cleanup allowlists for transient local-dense paths.

Use small fake command runners in unit tests. Do not make tests download COLMAP or execute real MVS.

**Expected source shape, flexible:**

```text
local_dense_reconstruction.py
run_local_dense_reconstruction.py
```

If one module is clearer, combine them.

## Task 14.2 — Implement live dense-backend capability discovery

Inspect in this order:

1. current pyCOLMAP 4.2 runtime;
2. existing COLMAP command/script installations in PATH and common external tool locations;
3. official COLMAP 4.2 Windows CUDA binary availability;
4. optional Docker GPU path only if already operational and simpler.

Record:

```text
pycolmap_version
pycolmap_has_cuda
pycolmap_dense_functions
colmap_cli_path
colmap_cli_version
colmap_cli_commands
cuda_dense_supported
gpu_identity
backend_kind
backend_reason
```

Current expected finding: pyCOLMAP 4.2 exposes dense APIs but `has_cuda=false`, so direct `pycolmap.patch_match_stereo` is not a valid dense-stereo backend.

## Task 14.3 — Install/extract official COLMAP 4.2 CUDA CLI only if needed

If no working compatible dense CLI exists:

1. use current official COLMAP 4.2 release/installation sources;
2. prefer the official prebuilt Windows CUDA binary over source compilation;
3. verify release identity/version before use;
4. install/extract outside the Git repository, preferably under a stable external tool root such as `C:\Tools\COLMAP-4.2.0`;
5. record the executable/batch path and archive/file checksum when practical;
6. do not modify project Python package versions merely to install the CLI;
7. do not add the binary/archive to Git.

If the prebuilt binary is unavailable or incompatible, the agent may choose one of these bounded paths:

- existing functional official CUDA Docker image;
- CUDA-enabled COLMAP 4.2 build via official supported Windows tooling **only if** the prerequisites are already available and the build is a reasonable continuation.

Do not spend the whole task building a new photogrammetry toolchain if official binary/Docker paths are blocked. In that case write a truthful capability blocker.

## Task 14.4 — Prove command surface before real data execution

Run harmless capability commands only:

- version/help;
- command inventory;
- confirm these commands exist in the chosen backend:
  - `image_undistorter`
  - `patch_match_stereo`
  - `stereo_fusion`
  - at least one supported mesher;
  - `mesh_simplifier` if simplification is intended;
  - `mesh_texturer`.

Confirm GPU-capable PatchMatch rather than assuming CUDA from the filename/package.

## Task 14.5 — Build exact 73-view source workspace

Do not copy all 288 selected images into dense work.

Use the Step 10 model's registered image names to create a deterministic local-view input set using the least wasteful safe method supported by COLMAP:

- preferred: pass the original selected image root plus the 73-view sparse model if the undistorter naturally exports only registered images;
- otherwise create a Step 14-local 73-image reference/copy set under `reconstruction/local_dense/work/`.

Never modify the source selected images.

## Task 14.6 — Run image undistortion / dense workspace initialization

Preferred COLMAP workspace output:

```text
reconstruction/local_dense/work/dense_workspace/
```

Run undistortion using the verified Step 10 sparse model.

Choose an undistortion max image size that remains compatible with the Step 15 starting dense size. If no limit is needed, retain full undistorted geometry but let PatchMatch downscale internally.

After undistortion validate:

- exact number of local images;
- image readability;
- COLMAP sparse workspace files;
- stereo config existence where expected;
- no files outside the Step 14 work root were modified.

## Task 14.7 — Generate Step 14 reports/preview

Write:

```text
reconstruction/local_dense/reports/step14_capability.json
reconstruction/local_dense/reports/step14_source_manifest.csv
reconstruction/local_dense/reports/step14_summary.json
```

Render a Step 14 sparse-source preview using the real Step 10 local camera/point geometry. It must visibly label this as the **73-image local source**, not a global model.

## Task 14.8 — Step 14 gate

Pass only if:

- Step 10 metrics match the protected baseline;
- exact 73 registered views are accounted for;
- dense workspace initialized successfully;
- a verified PatchMatch-capable backend is available;
- source integrity remains exact.

If fail: do not continue to Step 15.

---

# Step 15 — Dense stereo and fusion

## Task 15.1 — Implement dense-attempt configuration and reporting test-first

Add deterministic contracts for:

- preferred dense configuration;
- one fallback configuration derived from a measured failure category;
- allowed failure categories: OOM/resource, TDR/timeout, capability, corrupt output, subprocess failure;
- no second retry after the fallback;
- selected-attempt ranking that prioritizes hard validity/visual plausibility before point count;
- report format and timestamps/runtime;
- dense output path safety.

## Task 15.2 — Choose first dense configuration from live resources

Inspect actual GPU/driver/available memory if possible.

Select and record **one** preferred configuration before running it.

Recommended initial range:

```text
PatchMatchStereo.max_image_size = 1600..2000
StereoFusion.max_image_size     = compatible value
PatchMatchStereo.geom_consistency = true
PatchMatchStereo.filter = true
GPU index = 0 unless detected otherwise
source images = __auto__ default unless memory analysis justifies a smaller count
```

For the RTX 5050 8 GB laptop GPU, do not start at full 3072x4080 dense resolution unless the live backend proves this is practical.

## Task 15.3 — Run real PatchMatch stereo

Execute against the Step 14 COLMAP dense workspace.

Requirements:

- capture stdout/stderr to bounded task logs;
- record start/end/runtime;
- do not delete successful partial PatchMatch results if the backend supports resume;
- detect OOM/TDR/resource failure distinctly from algorithmic no-output failure;
- never run a second unrelated configuration while the first is still active.

## Task 15.4 — Controlled fallback only when justified

If preferred PatchMatch fails for a measured resource reason, derive one smaller configuration.

Prefer minimal changes in this order:

1. lower `max_image_size` one level;
2. reduce auto source views;
3. disable geometric consistency only if the above is insufficient or the failure specifically indicates it is necessary; keep filtering enabled.

Record exactly what changed and why.

No third attempt.

## Task 15.5 — Run stereo fusion

After a successful PatchMatch attempt:

1. run `stereo_fusion` using geometric input when geometric consistency was used;
2. choose the compatible fusion input mode if the fallback disabled geometric consistency;
3. write durable fused point cloud to:

```text
reconstruction/local_dense/dense/fused.ply
```

Keep transient depth/normal maps in the work tree.

## Task 15.6 — Parse and validate dense cloud

Implement a lightweight PLY parser/metric path sufficient to verify the actual output. Do not add Open3D/PyMeshLab merely for point counting if the PLY header/data can be read directly.

Measure:

```text
point_count
finite_xyz_count/fraction
has_color
bounding_box_min/max/extents
dense_to_sparse_ratio
file_size_bytes
```

Optionally compute a bounded sampled spacing estimate using NumPy/SciPy already installed; do not make this mandatory.

## Task 15.7 — Render dense preview

Generate a real dense-cloud preview from the fused points.

The preview should:

- make the local capture limitation clear;
- avoid cropping out outliers just to make the model look cleaner;
- optionally include the Step 10 camera arc for context;
- use real point colors when practical.

## Task 15.8 — Dense acceptance gate

Hard pass requirements:

- fused PLY reopens;
- point count > 6,099;
- finite XYZ fraction = 1.0 or any invalid records are explicitly removed by a deterministic rule and reported;
- non-zero X/Y/Z extent;
- bounds are plausible relative to the Step 10 sparse model;
- preview shows coherent vessel-related structure, not only noise/background.

Quality target:

```text
>= 60,990 points (10x sparse)
```

Below-target can still pass if hard gates and visual plausibility pass. Explain the decision.

Write:

```text
reconstruction/local_dense/reports/step15_dense_attempts.json
reconstruction/local_dense/reports/step15_summary.json
reconstruction/local_dense/previews/step15_01_dense_cloud.png
```

If the dense gate fails after the single allowed fallback, stop Steps 16-17 and document the failure.

---

# Step 16 — Mesh reconstruction and bounded cleanup

## Task 16.1 — Add mesh metric/selection tests first

Implement tests for:

- PLY mesh vertex/face parsing;
- finite-geometry validation;
- bounding-box comparison to dense cloud;
- connected-component summary if implemented;
- candidate validity classification;
- deterministic primary/alternative selection;
- simplification ratio reporting;
- final mesh output safety.

Do not make tests depend on real giant meshes.

## Task 16.2 — Choose primary mesher from dense evidence

Before execution, decide and record:

- Poisson primary if the dense cloud/normals support smooth surface reconstruction;
- Delaunay primary if visibility filtering is more appropriate for the measured outlier/background structure.

Current project context favors trying Poisson first unless the dense preview shows obvious broad background sheets or sparse visibility makes Poisson likely to close unsupported regions.

The agent may make the opposite choice with a short measured rationale.

## Task 16.3 — Run primary mesh

Write the original primary candidate to a durable evidence path, e.g.:

```text
reconstruction/local_dense/mesh/primary_mesh.ply
```

Record:

- algorithm;
- full options;
- runtime;
- output size;
- vertex/face counts;
- bounds;
- finite geometry.

## Task 16.4 — Visual and metric gate on primary mesh

Reject a primary mesh when any of these occurs:

- zero/no faces;
- non-finite geometry;
- plane/line/point collapse;
- giant enclosing shell unrelated to the measured cloud;
- vessel geometry is dominated by disconnected background/outlier surfaces;
- bounds are catastrophically inconsistent with the dense cloud.

## Task 16.5 — One alternative candidate if primary is invalid or clearly worse

If the primary fails or is visibly unsuitable, generate exactly one meaningful alternative with the other COLMAP meshing family.

Do not vary many options on the same mesher.

Write optional candidate to:

```text
reconstruction/local_dense/mesh/optional_alternative_mesh.ply
```

Select the final candidate by:

1. hard geometric validity;
2. visually plausible vessel structure;
3. dominant supported object component;
4. appropriate bounds;
5. sufficient detail;
6. only then face count/file size.

## Task 16.6 — Deterministic cleanup and simplification

If the selected mesh contains many small isolated components:

1. measure component face counts;
2. derive a defensible threshold from the distribution;
3. remove only small isolated components when the rule clearly separates noise from the main reconstruction;
4. report every removed component aggregate.

If mesh density is excessive for texturing/later Blender:

- run one QEM simplification;
- choose a target face ratio based on real face count, not a fixed arbitrary 0.1;
- default target band may be 0.25-0.75 depending on the input size;
- verify silhouette/detail after simplification;
- preserve the unsimplified selected candidate.

Final durable mesh:

```text
reconstruction/local_dense/mesh/final_mesh.ply
```

## Task 16.7 — Mesh reports and previews

Write:

```text
reconstruction/local_dense/reports/step16_mesh_attempts.json
reconstruction/local_dense/reports/step16_summary.json
reconstruction/local_dense/previews/step16_01_mesh.png
```

Only render a comparison figure if two real candidates exist:

```text
reconstruction/local_dense/previews/step16_02_mesh_comparison.png
```

## Task 16.8 — Step 16 hard gate

Pass only if final mesh:

- reopens;
- has non-zero vertices/faces;
- is finite;
- has plausible 3D extent relative to the fused cloud;
- has a dominant vessel-supporting component;
- does not contain an accepted giant artificial shell;
- visually preserves the local vessel silhouette/detail sufficiently for texturing.

If fail after one alternative candidate, stop before Step 17.

---

# Step 17 — Photo texturing

## Task 17.1 — Add texturing contract tests first

Tests should cover:

- chosen backend capability;
- safe texturer command construction;
- output manifest discovery;
- OBJ/MTL or backend-specific material-reference validation where applicable;
- texture image readability/dimensions;
- UV/material non-emptiness checks where practical;
- geometry count/bounds preservation after format conversion;
- one-retry gate;
- final integrated summary semantics.

## Task 17.2 — Use COLMAP 4.2 mesh_texturer as primary

Verify live CLI help for the actual option surface instead of guessing undocumented flags.

Input:

```text
workspace = Step 14/15 dense workspace
mesh      = Step 16 final_mesh.ply
output    = reconstruction/local_dense/texture/<final output root>
```

Select conservative atlas/options based on the actual mesh and available resources.

Do not copy all original 288 images into texturing; use the 73 calibrated local views already represented in the dense workspace.

## Task 17.3 — One controlled texturing fallback

If COLMAP texturing fails for a measured configuration/memory issue, change only the smallest relevant parameter once.

If the verified COLMAP 4.2 executable unexpectedly lacks `mesh_texturer`, the single architectural fallback permitted by the design is:

- Blender 5.2 headless scripted photo projection/baking using the same calibrated local views;
- no manual editing;
- no sculpting/remeshing;
- no aesthetic material redesign;
- output remains a reproducible Step 17 texture pipeline.

If that fallback would require inventing camera calibration/poses rather than importing the verified COLMAP model, stop and report blocked instead.

## Task 17.4 — Validate textured asset

Discover the actual generated outputs and build a manifest containing:

```text
relative_path
role (mesh/material/texture/other)
size_bytes
sha256
image_dimensions if texture
```

Validate as applicable:

- textured mesh readable;
- UV data non-empty;
- material references resolve;
- texture files readable;
- texture atlas dimensions non-zero;
- material/texture assignment covers a substantial part of the mesh;
- mesh geometry counts/bounds remain materially unchanged by texturing.

## Task 17.5 — Real textured preview

Create a real rendered preview of the textured result.

Preferred methods:

- backend-native preview/export if available;
- otherwise Blender 5.2 headless import/render used strictly as a validator.

The preview must not include manual cleanup that belongs to the next phase.

Reflective brass may cause seams, highlight inconsistencies, or view-dependent appearance. Document those limitations; do not hide them with procedural replacement materials.

## Task 17.6 — Step 17 hard gate

Pass only if:

- final textured mesh/material output exists;
- at least one readable photo-derived texture is referenced;
- UV/material assignment is non-empty;
- geometry remains valid;
- preview shows recognizably image-derived vessel appearance;
- any untextured regions/seams are disclosed.

Write:

```text
reconstruction/local_dense/reports/step17_texture_summary.json
reconstruction/local_dense/previews/step17_01_textured_preview.png
```

---

# Phase 5 — Integrated finalization, review, and documentation

## Task F.1 — Build one integrated final summary

Write:

```text
reconstruction/local_dense/reports/steps14_17_summary.json
```

Required top-level fields:

```text
source_sparse_model
source_registered_images
source_model_metrics
source_manifest_hash
capability
selected_dense_attempt
dense_acceptance
selected_mesh
mesh_acceptance
texture_output
texture_acceptance
steps14_17_success
known_limitations
next_phase
blender_manual_cleanup_started=false
```

`steps14_17_success` is true only if Steps 14, 15, 16, and 17 all pass their hard gates.

## Task F.2 — Measured documentation

Create:

```text
docs/geometry-ml/local-dense-mesh-texture.md
```

Update only after real execution:

```text
README.md
AGENTS.md
CHANGELOG.md
LESSONS.md                    only for durable lessons
docs/memory-bank/active-context.md
docs/memory-bank/progress.md
```

Document:

- Step 10 local source and why Step 13 is evidence-only;
- chosen COLMAP dense backend and capability reason;
- preferred and fallback dense attempts, if any;
- actual dense metrics;
- mesh algorithm and cleanup/simplification decision;
- texturing backend and asset manifest;
- all gate results;
- visual limitations from incomplete local coverage and reflective brass;
- next phase: Blender cleanup/final presentation model.

Do not claim full 360-degree/global reconstruction.

## Task F.3 — Spec-compliance review first

Check every frozen invariant:

- Step 10 source only;
- exact registered views only;
- raw/selected immutable;
- sparse recovery not reopened;
- bounded dense retries;
- at most two meaningful mesh candidates;
- no manual geometry invention;
- photo texture really exists;
- no manual Blender cleanup;
- no fabricated metrics/artifacts.

Fix any violations before code-quality review.

## Task F.4 — Code-quality review second

Review:

- subprocess safety and quoting on Windows;
- capability detection robustness;
- exact path handling;
- interrupted/resume behavior;
- PLY parsing correctness;
- output/reports consistency;
- cleanup allowlists;
- duplicated code with existing sparse helpers;
- deterministic ordering/hashes;
- no hidden external downloads during tests;
- explainability for coursework.

Add only narrow regressions for actual risks discovered.

## Task F.5 — Automated verification

Run:

1. focused Steps 14-17 tests;
2. Step 10-13 regression tests;
3. full project test suite;
4. Python compilation for changed source/test files;
5. documentation link/whitespace checks.

Do not report "all tests passed" unless the full relevant set actually ran.

## Task F.6 — Real artifact verification

Reopen/inspect:

- Step 10 sparse source;
- 73-view dense workspace manifest;
- fused PLY;
- final mesh;
- final textured asset/material/atlas;
- every generated preview.

Reverify:

- 297 raw images unchanged;
- 288 selected images unchanged;
- protected Step 10-13 reports/models unchanged;
- no unexpected source-image writes;
- no manual Blender output beyond optional automated validation render.

## Task F.7 — Cleanup

After all durable outputs are secured:

Remove task-created residue that is no longer needed:

- Python caches;
- failed-attempt temporary files;
- extracted/downloaded archives inside workspace if any accidentally landed there;
- duplicate render scratch files;
- obsolete temp pair/config files.

Dense `work/` handling is evidence-sensitive:

- remove depth/normal/transient workspace if all downstream steps are complete and reproducibility is fully captured by reports;
- otherwise preserve the minimum workspace needed to allow texture reproducibility, but keep it local-only and clearly documented.

Never delete source evidence or unrelated user-owned files.

## Task F.8 — Publication boundary

Do **not** commit/push automatically unless the active user request/session explicitly authorizes Git publication.

If publication is authorized:

1. fetch `origin/main`;
2. inspect divergence/status;
3. preflight intended files;
4. exclude `.codegraph/`, private CNN checkpoint, external tool installation, download caches, and transient dense workspace;
5. inspect binary sizes before staging final dense/mesh/textured assets;
6. keep oversized outputs local and publish manifests/hashes instead;
7. commit with a scoped conventional commit;
8. push normally, never force;
9. verify local `HEAD == origin/main == remote main`.

---

# Expected runner interface

The implementation may expose separate runners or one integrated runner. Prefer the simplest interface that remains restartable.

A good integrated CLI shape is:

```text
python -B run_local_reconstruction.py --stage prepare
python -B run_local_reconstruction.py --stage stereo
python -B run_local_reconstruction.py --stage mesh
python -B run_local_reconstruction.py --stage texture
python -B run_local_reconstruction.py --stage finalize
python -B run_local_reconstruction.py --stage all
```

Alternatively, separate `run_local_dense_reconstruction.py`, `run_local_mesh_reconstruction.py`, and `run_local_texturing.py` are acceptable.

Whichever architecture is chosen:

- each expensive stage must be independently restartable;
- `all` must continue automatically through successful gates;
- downstream stages must not run after a failed hard gate;
- report semantics must be owned by their stage rather than duplicated in `all` orchestration.

# Final acceptance checklist

## Step 14

- [ ] Live state recovered and protected baseline frozen.
- [ ] Exact Step 10 model reopened at 73 images / 6,099 points.
- [ ] Exact 73 registered filenames derived from the model.
- [ ] Working COLMAP PatchMatch-capable backend verified.
- [ ] Dense workspace initialized from exactly those 73 views.
- [ ] Raw/selected source integrity unchanged.
- [ ] Step 14 reports and real preview written.

## Step 15

- [ ] One preferred dense configuration recorded before execution.
- [ ] At most one bounded fallback used and justified.
- [ ] Real PatchMatch depth results produced.
- [ ] Real fused PLY produced and reopened.
- [ ] Fused point count > 6,099.
- [ ] Finite/plausible 3D extent verified.
- [ ] Dense preview visually inspected.
- [ ] Dense hard gate passed truthfully.

## Step 16

- [ ] Primary mesher choice justified from dense evidence.
- [ ] At most one alternative mesh candidate generated.
- [ ] Final mesh finite with non-zero vertices/faces.
- [ ] Dominant vessel-supporting component visually plausible.
- [ ] No giant artificial shell accepted.
- [ ] Any cleanup/simplification reproducible and recorded.
- [ ] Final mesh preview inspected.

## Step 17

- [ ] COLMAP `mesh_texturer` used if supported, otherwise one allowed narrow fallback documented.
- [ ] Final textured mesh/material/texture assets exist.
- [ ] UV/material/texture references validate.
- [ ] Real photo-derived texture image(s) reopen.
- [ ] Geometry remains valid.
- [ ] Textured preview visually inspected.
- [ ] Reflective/incomplete-coverage artifacts documented.

## Integrated completion

- [ ] `steps14_17_summary.json` agrees with real outputs.
- [ ] Focused + regression + full tests run.
- [ ] Touched Python compiles.
- [ ] Raw/selected/Step 10-13 evidence remains unchanged.
- [ ] Docs match measured outcome.
- [ ] Task-created residue cleaned or explicitly retained as local-only reproducibility state.
- [ ] Result described as **local 73-view reconstruction**, not full 288-view global reconstruction.
- [ ] `blender_manual_cleanup_started=false`.

# Final next phase after successful Steps 14-17

The next project phase is manual/controlled Blender cleanup and final presentation work:

1. import the accepted textured asset;
2. clean remaining reconstruction artifacts without inventing unsupported vessel geometry;
3. normalize scene orientation/scale where justified;
4. configure materials/lighting for presentation while preserving photo-texture evidence;
5. create final renders and model validation views;
6. complete final coursework report/presentation/submission packaging.
