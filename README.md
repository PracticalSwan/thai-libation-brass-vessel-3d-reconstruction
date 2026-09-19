# Thai Libation Brass Vessel 3D Reconstruction

CSX4213 Computer Vision project for reconstructing a real Thai brass libation vessel (ชุดกรวดน้ำ / ที่กรวดน้ำ) as a 3D asset from photographs.

The project implements a complete image-to-3D computer vision workflow: controlled acquisition, object segmentation, learned local feature extraction and matching, sparse structure-from-motion, dense multi-view stereo, Poisson surface reconstruction, and scan-preserving 3D asset preparation.

## Project overview

The goal is to reconstruct the vessel from real captured images rather than manually model its shape. Geometry is derived from the coated/marked capture set, while a separate uncoated image set is used for brass appearance information. Blender is used only for technical mesh preparation, UV generation, baking, material setup, validation, and export; it is not used to invent missing vessel geometry.

The current V4 dataset contains **688 JPEG images**:

- **423 coated/marked geometry images** for reconstruction
- **158 uncoated appearance images** for brass color/material statistics
- **107 empty-board/background images** for background and board discrimination

The immutable raw photographs are stored in `CSX4213_Project_V4_Images/`. The current derived reconstruction media are stored in `capture_v4/derived/` as **372 MVS images, 372 vessel masks, and 372 feature masks**. Both the raw and derived image trees are versioned with Git LFS; superseded pre-V4 captures and processing products are not part of the active repository.

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
| Asset preparation | Blender | Preserve the reconstructed scan, perform bounded evidence-backed cleanup, prepare materials, and validate/export without generic replacement modeling |
| Validation | Geometry checks, hashes, multi-view renders, clean GLB re-import | Verify reproducibility and exported-asset integrity |

## Current reconstruction artifacts

The repository contains the completed V4 reconstruction and its scan-preserving V7 final authoring lineage.

Key reconstruction artifacts include:

- **Sparse model:** 372 registered reconstruction views in the selected V4 sparse lineage
- **Dense fused cloud:** `reconstruction/v4/dense/fused_dense_best_defensible_v1_full_geometric_phase.ply`
- **Selected Poisson mesh:** `reconstruction/v4/mesh/poisson_best_defensible_v1_depth13_trim5.ply`
- **Canonical Blender asset:** `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend`
- **Canonical GLB asset:** `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb`
- **Final scan-preserving authoring master:** `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/Thai_Libation_Vessel_V4_V7_SCAN_PRESERVING_FINAL_V139.blend`
- **Professor submission package:** `submission/` — verified GLB and PLY deliverables, 14 final model images/contact sheets, README, and SHA-256 manifest

The selected dense reconstruction contains approximately **1.93 million fused points**. The selected Poisson surface contains approximately **4.95 million vertices** before production-mesh reduction. The completed V139 final remains a dense-derived scan mesh at **451,312 vertices / 902,838 triangular faces**. Cleanup was limited to the reconstructed vertices and same-object scan evidence; the physical bottom opening remains intentional, and the final material comes from the verified project-wide uncoated-image statistics. The canonical GLB was re-imported in a fresh Blender 5.2 process and reproduced one mesh with the same topology, material, identity transforms, and dimensions.

## Professor submission package

The root-level `submission/` folder is the compact handoff intended for the professor. It contains `Thai_Libation_Vessel_Final.glb`, `Thai_Libation_Vessel_Final.ply`, eight 45-degree turntable images, four closeups, two contact sheets, `README.txt`, and `SHA256SUMS.txt`. The GLB preserves the final brass material; the PLY is a binary triangulated geometry-and-normals export. Both were imported into fresh Blender 5.2 factory-startup scenes and reproduced the final V139 topology of **451,312 vertices / 902,838 triangles** and matching scene-space dimensions. The GLB SHA-256 is `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6`; the PLY SHA-256 is `42ab10ebec3b69dc8dc7ce28463a9d04fd3f0ecb3b35f888b62bc3c222eb9ff1`.

## Reconstruction evidence

Representative outputs are kept in the repository so the main stages can be inspected without retaining every temporary diagnostic artifact.

### Acquisition setup

- `private_images/` — coating, lighting, fixed-phone, rotating-board, and white-background setup photographs

### Sparse reconstruction

- `reconstruction/v4/previews/sparse_model_0_contact.png`
- `reconstruction/v4/previews/sparse_model_0_trajectory.png`

### Dense reconstruction

- `reconstruction/v4/previews/dense_dense_best_defensible_v1_full_geometric_phase_semantic/`

### Poisson mesh

- `reconstruction/v4/previews/raw_poisson_best_defensible_v1_depth13_trim5/`

### Final asset verification

- `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/previews/v139_final_surface_qa/`
- `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/previews/v139_final_brass_qa/`
- `reconstruction/v4/previews/final_lod0_v1/` — retained historical V6 verification
- `reconstruction/v4/previews/final_glb_reimport_v1/` — retained historical V6 verification

## Appearance and material workflow

The final brass appearance is separated from geometry reconstruction. The geometry images use temporary coating and markers, so their visible color is not treated as the final material reference.

Instead, the project derives reproducible appearance statistics from the **158 uncoated photographs**. The V139 production asset uses a single statistics-only brass material with:

- brass Base Color derived from the uncoated capture set
- metallic response appropriate to the photographed brass surface
- roughness derived from the appearance statistics
- surface relief supplied by the reconstructed mesh itself

The earlier V6 asset retained topology-specific AO/normal-map work, but those maps are not attached to V139 because its topology/UV lineage differs. This keeps the final appearance tied to project evidence without claiming a photographic projection or incompatible texture bake.

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
CSX4213_Project_V4_Images/      688 immutable V4 source photographs (Git LFS)
capture_v4/manifests/           Current capture roles, sequences, exclusions, and audit records
capture_v4/derived/             372 MVS images + 372 masks + 372 feature masks (Git LFS)
docs/                           Current project documentation and final formatted report
reconstruction/v4/              Sparse, dense, mesh, reports, previews, and Blender outputs
submission/                     Professor-facing GLB, PLY, final model images, README, and checksums
scripts/                        Reproducible V4 reconstruction and verification utilities
tests/                          Current pipeline regression and validation tests
```

Large temporary reconstruction workspaces, caches, and redundant diagnostics are intentionally not part of the final project record. The repository keeps the source data, selected reconstruction lineage, representative evidence, reproducible scripts, and final asset outputs needed to understand and verify the project.

## Main technologies

Python, PyTorch, OpenCV, Grounding DINO-T, SAM 2.1, ALIKED-N16Rot, LightGlue, pyCOLMAP, COLMAP 4.2, CUDA PatchMatch Stereo, Poisson surface reconstruction, Blender 5.2, FFmpeg, and ExifTool.

## Project report

A detailed description of the objective, acquisition design, computer vision methodology, implementation, reconstruction outputs, and validation is available in:

**[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)**

The final formatted paper is also included as `docs/Thai_Libation_Brass_Vessel_Final_Report.docx`.
