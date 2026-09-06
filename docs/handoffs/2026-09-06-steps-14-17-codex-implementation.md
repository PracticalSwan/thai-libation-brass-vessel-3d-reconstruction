# Codex Implementation Handoff — Steps 14-17

Use the prompt below as the implementation task for Codex.

---

# ONLY TASK — Implement Steps 14-17 Completely in One Continuous Run

Continue the canonical **CSX4213 Thai Libation Brass Vessel 3D Reconstruction** project and implement the next four reconstruction phases completely:

1. **Step 14 — Local dense preparation and acceptance contract**
2. **Step 15 — Dense stereo and fusion**
3. **Step 16 — Mesh reconstruction and bounded cleanup**
4. **Step 17 — Photo texturing**

Do these four steps in one continuous implementation session. Proceed automatically when gates pass. Do not stop for routine approval or minor implementation choices. Use the bounded flexibility already authorized in the design/plan instead of asking me to choose every parameter.

## Canonical project

```text
C:\Assumption University\CSX4213\Project
```

Expected starting baseline when this handoff was written:

```text
branch       main
HEAD         eef926c9d47db9f42a2b363cee77d68edb298a08
origin/main  eef926c9d47db9f42a2b363cee77d68edb298a08
```

Recover the actual live state first. Newer authoritative state wins if the repository changed after this handoff.

## Mandatory first reads

Before implementation, read in this order:

```text
AGENTS.md
CLAUDE.md
LESSONS.md
docs/memory-bank/active-context.md
docs/memory-bank/progress.md
docs/superpowers/specs/2026-09-06-steps-14-17-local-dense-mesh-texture-design.md
docs/superpowers/plans/2026-09-06-steps-14-17-local-dense-mesh-texture.md
docs/geometry-ml/external-learned-global-recovery.md
```

Treat the Steps 14-17 design and integrated implementation plan as the authoritative implementation contract for this task.

## Core project state you must preserve

Sparse recovery is finished. Do **not** reopen it.

The downstream sparse source is frozen to:

```text
reconstruction/sparse/best
```

Expected source metrics:

```text
registered images             73 / 288
sparse points                 6,099
observations                  21,351
mean reprojection error       1.2373052447638215 px
camera count                  1
camera model                  SIMPLE_RADIAL
camera params                 3542.7206959261907, 1536.0, 2040.0, -0.01641661056124677
```

The Step 13 learned model under:

```text
reconstruction/external_learned_recovery/best
```

is evidence only. It reached 266/288 but failed the frozen >=274 global gate. **Never substitute it as the dense source.**

Raw and selected sources remain immutable:

```text
IMG20260826122949/
preprocessing/pycolmap_input/images/
```

Known local-only state that must remain excluded from publication:

```text
.codegraph/
analysis/ml/checkpoints/
```

## Goal

Produce the best defensible **local 73-view** dense reconstruction that the accepted Step 10 capture arc supports, then convert it into a validated mesh and a real photo-textured asset.

A complete success means:

```text
Step 14 passes -> exact 73-view dense workspace + working dense backend
Step 15 passes -> accepted real fused dense point cloud
Step 16 passes -> accepted real final mesh
Step 17 passes -> accepted real photo-textured mesh/material/atlas
```

Never describe the result as a complete 288-view global reconstruction.

# Execution requirements

## 1. Recover and freeze evidence before modifying code

First:

- identify the Git root/branch/HEAD/origin state;
- inspect working-tree changes and preserve unrelated contributor-owned work;
- reverify 297 raw images against the published baseline;
- reverify 288 selected images and the frozen selection-manifest hash;
- reopen the Step 10 sparse model and confirm the 73-image / 6,099-point metrics;
- derive the exact 73 registered image names from the model itself;
- record pre-task hashes for protected Step 10-13 reports/models so accidental changes are detectable.

Do not assume the registered images are indices 1-73 without verifying model names.

## 2. Implement test-first for deterministic logic

Use TDD for:

- local-source validation;
- registered-image manifest/hash;
- path/workspace safety;
- COLMAP backend capability representation;
- safe command construction/subprocess invocation;
- dense attempt/fallback gates;
- PLY point/mesh metric parsing;
- mesh candidate validation/ranking;
- texture output/material/UV/atlas validation;
- cleanup allowlists;
- report/final-summary semantics.

Do not make unit tests download COLMAP or run giant MVS jobs. Real MVS execution belongs to the real runtime stages.

Keep implementation simple and course-explainable. You may adjust the suggested module split if a smaller coherent architecture is clearer.

## 3. Step 14 — Dense backend and local workspace

The current planning inspection found:

```text
pyCOLMAP 4.2.0
pycolmap.has_cuda = false
pyCOLMAP exposes undistort_images / patch_match_stereo / stereo_fusion /
poisson_meshing / simplify_mesh
COLMAP CLI not currently in PATH
Blender 5.2 installed
```

`pycolmap.patch_match_stereo` requires CUDA, so the current wheel is not expected to be a viable dense-stereo backend.

### Preferred live backend

Use the simplest verified **COLMAP 4.2 CUDA** path.

Expected practical choice on this Windows machine:

- official COLMAP 4.2.0 Windows CUDA command-line binary;
- install/extract outside the repository if needed, preferably under a stable external tools directory such as `C:\Tools\COLMAP-4.2.0`;
- verify version/command surface before real execution;
- do not change the project Python dependency stack just to install the CLI;
- do not add tool binaries/archives to Git.

Official sources to re-check if installation is needed:

```text
https://colmap.github.io/install.html
https://colmap.github.io/cli.html
https://colmap.github.io/tutorial
https://github.com/colmap/colmap/releases/tag/4.2.0
```

The verified backend must expose at least:

```text
image_undistorter
patch_match_stereo
stereo_fusion
poisson_mesher and/or delaunay_mesher
mesh_simplifier if simplification is used
mesh_texturer
```

A CUDA-enabled pyCOLMAP 4.2 runtime is acceptable if one already exists by then. Official COLMAP CUDA Docker is allowed only if GPU passthrough is already working and it is simpler than the Windows binary.

Do **not** introduce OpenMVS/Meshroom/proprietary cloud reconstruction unless all approved COLMAP paths are genuinely blocked and the architecture has to be explicitly revised.

### Dense workspace

Use exactly the Step 10 registered views.

Preferred work root:

```text
reconstruction/local_dense/work/dense_workspace/
```

Undistort/initialize a COLMAP dense workspace from:

```text
sparse = reconstruction/sparse/best
images = preprocessing/pycolmap_input/images/
```

Validate that the dense workspace represents exactly the 73 registered views.

Write real Step 14 reports and a sparse-source preview before Step 15.

Step 14 hard gate must pass before stereo begins.

## 4. Step 15 — Dense stereo + fusion

Choose one preferred dense configuration after inspecting the live GPU/backend.

The design deliberately allows flexibility. For the RTX 5050 8 GB laptop GPU, choose a sensible first max image size in the **1600-2000** range rather than blindly using full 3072x4080.

Preferred principles:

```text
geom_consistency = true
filter = true
GPU index = detected working GPU, normally 0
auto source views unless memory evidence justifies a bounded count
```

Record the full preferred configuration **before** executing it.

### One controlled fallback only

If PatchMatch fails because of OOM, Windows TDR/timeout, or clearly measured resource pressure, you may run one lower-resource retry.

Use the smallest justified change, e.g.:

1. lower `max_image_size` one level;
2. reduce source image count;
3. only if needed, disable geometric consistency while keeping filtering enabled.

Do not sweep combinations. No third attempt.

After successful PatchMatch, fuse to:

```text
reconstruction/local_dense/dense/fused.ply
```

Measure at least:

```text
point count
finite XYZ fraction
color availability
bounding box/extents
dense/sparse point ratio
file size
attempt configuration/runtime
```

Hard dense requirements:

- PLY reopens;
- point count > 6,099;
- finite, non-degenerate 3D extent;
- bounds plausible relative to Step 10 sparse source;
- real preview shows coherent vessel-related structure.

Quality target, not mandatory hard threshold:

```text
>= 60,990 points (~10x sparse)
```

A smaller but coherent cloud may pass if hard gates and visual evidence pass. A huge invalid cloud must fail.

## 5. Step 16 — Mesh reconstruction

Use the real dense cloud to choose the primary mesher.

- Poisson is reasonable when the dense cloud/normals support a smooth surface.
- Delaunay is reasonable when visibility-based filtering better handles broad outliers/background.

Choose one primary and record why.

At most **one** alternative mesh candidate is allowed if the primary is invalid or clearly worse. Do not parameter-sweep meshers.

Allowed cleanup is bounded and reproducible:

- remove isolated tiny components only by an explicit size rule derived from actual component distribution;
- preserve the dominant vessel-supporting component when defensible;
- QEM simplify only if useful for texturing/later Blender;
- preserve the unsimplified primary candidate.

Never hand-sculpt, arbitrarily fill holes, or invent unsupported vessel geometry.

Final mesh:

```text
reconstruction/local_dense/mesh/final_mesh.ply
```

Hard mesh requirements:

- non-zero vertices/faces;
- finite geometry;
- plausible 3D extent relative to the dense cloud;
- dominant vessel-supporting component;
- no accepted giant artificial shell;
- real mesh preview is plausible enough for Step 17.

## 6. Step 17 — Photo texturing

Primary path: **COLMAP 4.2 `mesh_texturer`** using:

```text
workspace = Step 14/15 dense workspace
mesh      = Step 16 final mesh
views     = the calibrated 73 local views
```

Inspect live `mesh_texturer -h` / command help before choosing flags. Do not guess unsupported options.

Use conservative atlas/packing parameters based on the actual mesh/resources.

One texturing retry is allowed only for a measured packing/memory/configuration issue.

### Narrow fallback

If the verified COLMAP 4.2 backend unexpectedly lacks `mesh_texturer`, the design permits **Blender 5.2 headless scripted texturing** as a narrow fallback using the same calibrated cameras/views.

This fallback may:

- import verified geometry/cameras;
- generate UVs;
- project/bake the real photographs;
- render an automated validation preview.

It may **not**:

- manually model/sculpt/remesh the vessel;
- start aesthetic cleanup;
- replace photo appearance with a procedural brass material and call that Step 17 texturing.

Manual Blender cleanup is the next project phase, not this task.

Validate the actual texture output contract:

- mesh/material/texture files exist;
- texture images reopen;
- UV/material assignment is non-empty;
- material references resolve;
- geometry is materially unchanged by texturing;
- a real render shows recognizably image-derived vessel appearance.

Document reflection/seam/incomplete-coverage artifacts instead of hiding them.

## 7. Staged/restartable architecture

Prefer a restartable runner with stages roughly equivalent to:

```text
prepare
stereo
mesh
texture
finalize
all
```

You may use one runner or a few small runners, whichever is clearer.

Important orchestration rule:

- each stage owns its reports/gate semantics;
- `all` only sequences stages;
- do not duplicate gate interpretation in the top-level `all` path.

Continue automatically:

```text
prepare pass -> stereo
stereo pass -> mesh
mesh pass -> texture
texture pass -> finalize
```

On a hard gate failure after the allowed fallback, stop downstream generation and write a truthful partial result.

## 8. Durable artifact layout

Use:

```text
reconstruction/local_dense/
  dense/
  mesh/
  texture/
  reports/
  previews/
  work/          # heavy/transient/local-only by default
```

Expected reports include:

```text
step14_capability.json
step14_source_manifest.csv
step14_summary.json
step15_dense_attempts.json
step15_summary.json
step16_mesh_attempts.json
step16_summary.json
step17_texture_summary.json
steps14_17_summary.json
```

Expected final summary semantics:

```text
steps14_17_success = step14 && step15 && step16 && step17 hard gates
blender_manual_cleanup_started = false
```

## 9. Visual verification is mandatory

Generate and actually inspect real previews for:

- Step 14 local sparse source;
- Step 15 fused dense cloud;
- Step 16 final mesh;
- mesh comparison only if two real candidates exist;
- Step 17 textured result.

Do not claim visual plausibility from metrics alone.

Do not crop out major outliers solely to make evidence look cleaner.

## 10. Documentation after measured execution

Create:

```text
docs/geometry-ml/local-dense-mesh-texture.md
```

Then update authoritative docs to the measured final state:

```text
README.md
AGENTS.md
CHANGELOG.md
LESSONS.md only for durable lessons
docs/memory-bank/active-context.md
docs/memory-bank/progress.md
```

State clearly:

- this is a **local 73-view reconstruction**;
- Step 13's 266-image model remains evidence-only because it failed the prior global gate;
- which dense backend/version was used;
- whether the one dense fallback was needed;
- actual fused-point metrics;
- actual mesh algorithm/metrics;
- any cleanup/simplification;
- actual texturing output/limitations;
- next phase is Blender cleanup/final presentation.

## 11. Review order

After implementation, do two separate reviews:

### Review A — spec compliance

Check:

- Step 10 source only;
- exact 73 views;
- raw/selected immutable;
- no sparse recovery restarted;
- dense fallback bounded;
- at most one alternative mesh;
- no geometry invention;
- photo texture is real;
- no manual Blender cleanup;
- no fabricated evidence.

Fix any violation before moving on.

### Review B — code quality

Inspect/fix:

- Windows subprocess quoting;
- capability detection;
- path boundary safety;
- interruption/restart behavior;
- PLY parsing correctness;
- deterministic reports/manifests/hashes;
- cleanup safety;
- unnecessary dependencies;
- duplicate logic;
- coursework explainability.

Add only narrow tests for real defects/risks discovered.

## 12. Verification before completion

Run proportionate real verification:

1. focused Steps 14-17 tests;
2. relevant Step 10-13 regression tests;
3. full project tests;
4. Python compilation for changed files;
5. doc link/whitespace checks;
6. raw 297-image integrity check;
7. selected 288-image integrity check;
8. protected Step 10-13 report/model hash verification;
9. reopen final fused cloud/mesh/textured asset;
10. inspect all final previews;
11. cleanup task residue and confirm private/local-only state remains excluded.

Do not say all tests passed unless they actually ran.

## 13. Storage and cleanup

Do not commit giant intermediate MVS outputs by default.

`reconstruction/local_dense/work/` should remain local-only while needed.

Depth maps/normal maps/tool downloads/install trees should not enter Git.

For final dense/mesh/textured binaries:

- inspect size before any publication;
- publish compact final artifacts only when repository-appropriate;
- if too large, keep them local and publish exact manifests/hashes/metrics/reproduction instructions instead.

Do not delete work needed for Step 17 before texturing has completed.

After Step 17 and durable verification, remove only task-created transient state that is no longer needed.

## 14. Git boundary

This prompt authorizes implementation of Steps 14-17, not automatic publication by itself.

Do **not** commit or push unless the active user request/session separately authorizes publication.

If later authorized, stage only intended source/tests/docs/reports/previews and repository-appropriate final assets. Exclude:

```text
.codegraph/
analysis/ml/checkpoints/
reconstruction/local_dense/work/
external COLMAP install/archive
model/tool caches
unrelated contributor changes
```

Never force-push.

# Definition of done

Do not finish merely because code exists.

The requested implementation is complete only when, as far as the actual runtime permits:

- Step 14 has a verified 73-view dense workspace and working dense backend;
- Step 15 has a real accepted fused point cloud;
- Step 16 has a real accepted final mesh;
- Step 17 has a real accepted photo-textured asset;
- reports/previews reflect real outputs;
- both review passes are complete;
- focused/regression/full verification is complete;
- protected evidence remains unchanged;
- task residue is cleaned/bounded;
- docs reflect the measured state;
- the final output is described honestly as a local reconstruction.

If a genuine hard blocker prevents one of Steps 14-17 even after its explicitly allowed fallback, complete the strongest safe partial implementation, preserve all evidence, report the exact blocker and the remaining step, and do not fabricate the downstream asset.

Proceed now from the live repository state and continue automatically through all four steps.
