# Thai Libation Brass Vessel 3D Reconstruction

CSX4213 Computer Vision project for reconstructing a real Thai brass libation vessel (ชุดกรวดน้ำ / ที่กรวดน้ำ) as a 3D asset from photographs.

The project implements a complete image-to-3D computer vision workflow: controlled acquisition, object segmentation, learned local feature extraction and matching, sparse structure-from-motion, dense multi-view stereo, Poisson surface reconstruction, and scan-preserving 3D asset preparation.

## Project overview

The goal is to reconstruct the vessel from real captured images rather than manually model its shape. Geometry is derived from the coated/marked capture set, while a separate uncoated image set is used for brass appearance information. Blender is used only for technical mesh preparation, UV generation, baking, material setup, validation, and export; it is not used to invent missing vessel geometry.

The current V4 dataset contains **688 JPEG images**:

- **423 coated/marked geometry images** for reconstruction
- **158 uncoated appearance images** for brass color/material statistics
- **107 empty-board/background images** for background and board discrimination

The geometry sequence uses a fixed-camera turntable/object-rotation setup on a white background. The object was photographed from multiple elevation passes with a locked rear-camera configuration. A temporary matte coating reduces specular reflection, and high-contrast markers improve local feature detection on the brass surface.

For a fuller methodology and project summary, see **[Project Report](docs/PROJECT_REPORT.md)**.

## Computer vision pipeline

```text
Controlled multi-view image capture
        ↓
Media ingest, metadata audit, and image QA
        ↓
Grounding DINO-T object localization
        ↓
SAM 2.1 full-resolution vessel segmentation
        ↓
Geometry-preserving preprocessing and masks
        ↓
ALIKED-N16Rot local feature extraction
        ↓
LightGlue feature matching
        ↓
COLMAP two-view geometric verification
        ↓
pyCOLMAP / COLMAP sparse Structure-from-Motion
        ↓
Camera refinement and bundle adjustment
        ↓
Image and mask undistortion
        ↓
CUDA COLMAP PatchMatch Stereo
        ↓
Mask-aware dense stereo fusion
        ↓
Poisson surface reconstruction
        ↓
Scan-preserving mesh cleanup and LOD preparation
        ↓
UV generation + AO/detail-normal baking
        ↓
Brass PBR material derived from uncoated captures
        ↓
Blender master + GLB export
        ↓
Fresh GLB re-import and multi-view verification
```

## Pipeline components

| Stage | Main methods / tools | Purpose |
| --- | --- | --- |
| Acquisition | Fixed camera, rotating object, multiple elevation passes | Capture overlapping views around the full vessel |
| Media audit | FFmpeg, OpenCV, ExifTool | Verify decode, dimensions, metadata, and capture consistency |
| Segmentation | Grounding DINO-T + SAM 2.1 | Isolate the vessel from the stationary background and rotating board |
| Local features | ALIKED-N16Rot | Detect and describe repeatable local image features |
| Matching | LightGlue | Match learned local features between overlapping views |
| Geometry verification | COLMAP | Reject geometrically inconsistent correspondences |
| Sparse reconstruction | pyCOLMAP / COLMAP SfM | Estimate camera poses and a sparse 3D point cloud |
| Dense reconstruction | CUDA COLMAP PatchMatch Stereo | Estimate per-view depth and normal maps |
| Dense fusion | Mask-aware stereo fusion | Combine depth estimates into a dense colored point cloud |
| Meshing | COLMAP Poisson mesher | Convert the fused point cloud into a continuous scan-derived surface |
| Asset preparation | Blender | Prepare LOD0, UVs, baked detail, material, and export without manual shape creation |
| Validation | Geometry checks, hashes, multi-view renders, clean GLB re-import | Verify reproducibility and exported-asset integrity |

## Current reconstruction artifacts

The repository contains a verified V4 reconstruction baseline together with an active scan-preserving refinement workspace.

Key reconstruction artifacts include:

- **Sparse model:** 372 registered reconstruction views in the selected V4 sparse lineage
- **Dense fused cloud:** `reconstruction/v4/dense/fused_dense_best_defensible_v1_full_geometric_phase.ply`
- **Selected Poisson mesh:** `reconstruction/v4/mesh/poisson_best_defensible_v1_depth13_trim5.ply`
- **Canonical Blender asset:** `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend`
- **Canonical GLB asset:** `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb`
- **Current refinement workspace:** `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_repair/`

The selected dense reconstruction contains approximately **1.93 million fused points**. The selected Poisson surface contains approximately **4.95 million vertices** before production-mesh reduction. The verified production LOD0 baseline contains **151,547 vertices** and **294,713 faces**.

## Reconstruction evidence

Representative outputs are kept in the repository so the main stages can be inspected without retaining every temporary diagnostic artifact.

### Sparse reconstruction

- `reconstruction/v4/previews/sparse_model_0_contact.png`
- `reconstruction/v4/previews/sparse_model_0_trajectory.png`

### Dense reconstruction

- `reconstruction/v4/previews/dense_dense_best_defensible_v1_full_geometric_phase_semantic/`

### Poisson mesh

- `reconstruction/v4/previews/raw_poisson_best_defensible_v1_depth13_trim5/`

### Final asset verification

- `reconstruction/v4/previews/final_lod0_v1/`
- `reconstruction/v4/previews/final_glb_reimport_v1/`

## Appearance and material workflow

The final brass appearance is separated from geometry reconstruction. The geometry images use temporary coating and markers, so their visible color is not treated as the final material reference.

Instead, the project derives reproducible appearance statistics from the **158 uncoated photographs**. The production asset uses:

- brass-colored Base Color derived from the uncoated capture set
- metallic material response
- roughness derived from the appearance statistics
- scan-derived ambient-occlusion information
- baked tangent-space normal/detail information from the higher-detail scan geometry

This keeps the final appearance tied to the photographed project object while keeping geometry and appearance acquisition roles separate.

## Validation and reproducibility

The pipeline records manifests, reconstruction reports, model hashes, and validation outputs for the important stages. Validation includes:

- source-image and metadata auditing
- mask and feature-support checks
- geometric match verification
- sparse camera and track checks
- dense depth/normal-map accounting
- mask-aware fused-cloud inspection
- Poisson connected-component measurements
- finite geometry, normal, UV, and material checks
- deterministic multi-view render comparison
- clean GLB re-import into an empty Blender scene

The verified exported baseline contains one intended production mesh with the expected material and texture connections. The authoring and re-import renders were also compared numerically to confirm that export preserves the prepared asset.

## Repository structure

```text
capture_v4/                     V4 manifests, masks, and derived capture inputs
CSX4213_Project_V4_Images/      Immutable V4 source photographs
analysis/                       Computer-vision analysis and representative evidence
docs/                           Project documentation and detailed report
reconstruction/v4/              Sparse, dense, mesh, reports, previews, and Blender outputs
scripts/                        Reproducible reconstruction and verification utilities
tests/                          Regression and pipeline validation tests
```

Large temporary reconstruction workspaces, caches, and redundant diagnostics are intentionally not part of the final project record. The repository keeps the source data, selected reconstruction lineage, representative evidence, reproducible scripts, and final asset outputs needed to understand and verify the project.

## Main technologies

Python, PyTorch, OpenCV, Grounding DINO-T, SAM 2.1, ALIKED-N16Rot, LightGlue, pyCOLMAP, COLMAP 4.2, CUDA PatchMatch Stereo, Poisson surface reconstruction, Blender 5.2, FFmpeg, and ExifTool.

## Project report

A detailed description of the objective, acquisition design, computer vision methodology, implementation, reconstruction outputs, and validation is available in:

**[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)**
