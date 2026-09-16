# Project Report — Thai Libation Brass Vessel 3D Reconstruction

## 1. Project summary

This CSX4213 Computer Vision project reconstructs a real Thai brass libation vessel (ชุดกรวดน้ำ / ที่กรวดน้ำ) from photographs and produces a usable 3D asset.

The work is designed as an end-to-end computer vision reconstruction system rather than a manual 3D modeling exercise. The vessel geometry is recovered from a controlled multi-view image capture using segmentation, learned feature matching, structure-from-motion, multi-view stereo, and Poisson surface reconstruction. The reconstructed scan is then prepared for practical use in Blender through geometry-preserving cleanup, level-of-detail preparation, UV generation, texture baking, material construction, export, and re-import verification.

A separate set of uncoated photographs is used for appearance information so that the final brass material does not inherit the temporary coating and marker colors used to improve geometry reconstruction.

## 2. Project objective

The project has four main objectives:

1. Capture enough overlapping views of the physical vessel to recover its three-dimensional structure.
2. Build a reproducible computer vision pipeline that estimates camera motion and vessel geometry directly from the captured images.
3. Convert the reconstructed dense scan into an editable and portable 3D asset without manually redesigning the vessel shape.
4. Preserve evidence and validation outputs for the major stages so that the reconstruction can be inspected and reproduced.

The project deliberately keeps geometric reconstruction and appearance acquisition separate. Coated and marked images are optimized for geometric correspondence; uncoated images are used for the visual appearance of the brass surface.

## 3. Object and acquisition setup

The subject is a Thai brass libation vessel with several visually distinct regions, including the pedestal and foot, bowl/body, shoulder and globe transitions, long neck, rim/opening, lid tiers, and upper finial structure.

Brass is challenging for photogrammetry because a polished metallic surface produces view-dependent highlights and often contains large regions with weak visual texture. To improve correspondence quality, the geometry capture uses:

- a temporary matte coating to reduce strong specular reflections;
- high-contrast, irregular marker dots to create non-repeating local texture;
- a fixed camera and rotating-object setup;
- several elevation passes to observe the vessel from lower, horizontal, upper, and steep-high viewpoints;
- a white background to simplify visual isolation;
- empty-board/background images to help distinguish the vessel from the rotating support and stationary background.

Setup photographs are retained in `private_images/` as practical evidence of the coating step, lighting arrangement, fixed-phone setup, rotating board, and white-background layout used during acquisition.

The audited geometry capture uses the OPPO Reno12 F rear camera with a consistent lens state. Geometry and empty-board images are 3072 × 4080 pixels, captured using the same rear-camera configuration with digital zoom 1 and consistent exposure metadata.

## 4. V4 image dataset

The authoritative V4 source dataset is stored in `CSX4213_Project_V4_Images/` and contains **688 JPEG images**.

| Image group | Count | Role |
| --- | ---: | --- |
| Coated/marked object images | 423 | Geometry reconstruction |
| Uncoated object images | 158 | Brass appearance and material statistics |
| Empty-board/background images | 107 | Background/support discrimination and acquisition evidence |
| **Total** | **688** | Complete V4 capture |

The raw source photographs are treated as immutable input. Processing takes place in derived directories and reconstruction workspaces so that the original capture remains unchanged. The 688 raw JPEGs and the current processed image trees are versioned with Git LFS so the repository contains the exact media used by the V4 pipeline without retaining superseded captures.

## 5. End-to-end computer vision pipeline

The final workflow is:

```text
Image acquisition
  -> media and metadata audit
  -> vessel localization and segmentation
  -> masked reconstruction inputs
  -> ALIKED-N16Rot local features
  -> LightGlue feature matching
  -> COLMAP geometric verification
  -> sparse Structure-from-Motion
  -> camera refinement / bundle adjustment
  -> image and mask undistortion
  -> CUDA PatchMatch Stereo
  -> mask-aware stereo fusion
  -> dense point cloud
  -> Poisson surface reconstruction
  -> scan-preserving mesh preparation and bounded repair
  -> project-image-derived brass material
  -> Blender master and GLB export
  -> clean re-import verification
```

The following sections describe each stage.

## 6. Stage 1 — Media ingest and capture audit

The capture is first inspected before reconstruction. FFmpeg and OpenCV are used for media handling and image validation, while ExifTool is used to inspect capture metadata.

The audit records information such as:

- image dimensions;
- capture sequence membership;
- camera/lens metadata;
- exposure consistency;
- geometry, appearance, and background roles;
- exclusions and derived reconstruction subsets.

The resulting manifests are kept under `capture_v4/manifests/`.

This stage ensures that later reconstruction steps use the intended photographs and consistent camera state.

## 7. Stage 2 — Vessel localization and segmentation

The background and rotating support contain strong visual features that must not become part of the vessel reconstruction. The project therefore performs explicit object isolation before feature matching and dense reconstruction.

The segmentation pipeline uses:

1. **Grounding DINO-T** to localize the vessel;
2. **SAM 2.1** to produce full-resolution vessel masks;
3. conservative mask refinement to preserve thin boundaries and the base region;
4. empty-board/background evidence to reduce support-board leakage.

The full-resolution masks are reused by later stages rather than relying on a simple color threshold. The current derived set is stored under `capture_v4/derived/` and contains 372 reconstruction images in `mvs_images/`, 372 vessel masks in `masks/`, and 372 feature masks in `feature_masks/`. These processed images are versioned with Git LFS alongside the raw V4 photographs.

## 8. Stage 3 — Geometry-preserving preprocessing

Preprocessing is intentionally conservative. The objective is to make the images suitable for reconstruction while preserving the original vessel geometry and camera relationships.

Operations include image decoding, manifest generation, mask alignment, reconstruction-input preparation, and validation of image/mask dimensions. No geometry is synthesized at this stage.

## 9. Stage 4 — Learned local feature extraction

Local image features are extracted with **ALIKED-N16Rot**.

ALIKED provides keypoints and descriptors designed for repeatable local matching across viewpoint changes. The reconstruction uses the vessel masks so that feature extraction and matching are concentrated on relevant object regions rather than the stationary cloth background or the rotating board.

This stage produces the feature representation used to establish image-to-image correspondences across the different turntable views and elevations.

## 10. Stage 5 — Feature matching with LightGlue

**LightGlue** matches the ALIKED descriptors between candidate image pairs.

The matching stage is responsible for connecting observations of the same physical vessel surface across different photographs. Pair scheduling uses acquisition locality and reconstruction evidence so that nearby turntable views receive strong coverage while useful cross-elevation connections can also be retained.

The project stores matching and pair metadata so that the sparse reconstruction lineage can be reproduced and audited.

## 11. Stage 6 — COLMAP geometric verification

Raw descriptor matches are not automatically accepted as geometry. COLMAP two-view geometry is used to verify correspondences using camera geometry and robust estimation.

This step filters inconsistent matches and converts learned image correspondences into geometrically meaningful constraints for structure-from-motion.

## 12. Stage 7 — Sparse Structure-from-Motion

The verified matches feed a **pyCOLMAP / COLMAP Structure-from-Motion** workflow.

Sparse reconstruction estimates:

- camera intrinsics;
- camera poses;
- multi-view feature tracks;
- sparse 3D points;
- reprojection relationships between images and reconstructed points.

Bundle adjustment refines the camera and point estimates jointly. The selected V4 sparse lineage contains **372 registered reconstruction views** and provides the camera geometry used by the dense stage.

Representative sparse outputs are available in:

- `reconstruction/v4/previews/sparse_model_0_contact.png`
- `reconstruction/v4/previews/sparse_model_0_trajectory.png`

## 13. Stage 8 — Image and mask undistortion

Before dense reconstruction, COLMAP undistorts the registered images using the selected sparse camera model.

The segmentation masks are transformed into the same coordinate system and resolution as the dense reconstruction images. This alignment is important because the masks are also used during dense fusion to prevent background and support-board pixels from entering the final point cloud.

## 14. Stage 9 — Dense multi-view stereo

Dense depth estimation uses **COLMAP 4.2 PatchMatch Stereo with CUDA acceleration**.

For each registered reference image, PatchMatch Stereo estimates:

- a dense depth map;
- a surface-normal map;
- multi-view geometric consistency.

The selected full dense reconstruction uses 372 registered views at a 2000-pixel maximum image size and geometric consistency processing on GPU 0.

## 15. Stage 10 — Mask-aware dense stereo fusion

The depth maps are fused into a dense colored 3D point cloud. Vessel masks remain active during fusion so that reconstructed points are concentrated on the object rather than the background or board.

The selected V4 fused cloud is:

`reconstruction/v4/dense/fused_dense_best_defensible_v1_full_geometric_phase.ply`

It contains approximately **1.93 million fused points**.

Representative dense views are stored in:

`reconstruction/v4/previews/dense_dense_best_defensible_v1_full_geometric_phase_semantic/`

The repository also records dense reconstruction reports that bind the selected cloud to its sparse model, camera set, masks, and runtime configuration.

## 16. Stage 11 — Poisson surface reconstruction

The dense oriented point cloud is converted into a triangle surface with the **COLMAP Poisson mesher**.

The selected scan-derived Poisson mesh is:

`reconstruction/v4/mesh/poisson_best_defensible_v1_depth13_trim5.ply`

The mesh contains approximately:

- **4.95 million vertices**;
- **9.90 million faces** before production-mesh reduction.

The selected mesh is strongly dominated by one connected surface component, which makes it suitable for downstream scan-preserving asset preparation.

Representative Poisson views are stored in:

`reconstruction/v4/previews/raw_poisson_best_defensible_v1_depth13_trim5/`

## 17. Stage 12 — Scan-preserving 3D asset preparation

Blender 5.2 is used as a technical post-processing environment after the image-derived reconstruction is complete.

The asset-preparation stage includes:

- preserving the raw Poisson/CleanHigh reconstruction as source evidence;
- keeping the final geometry traceable to the dense-derived mesh;
- removing only evidence-backed synthetic or defective scan geometry;
- bounded smoothing/fairing of reconstructed vertices without redesigning the vessel;
- material construction from project-derived appearance statistics;
- GLB export and fresh-process re-import verification.

An earlier V6 production LOD0 baseline contained **151,547 vertices / 294,713 faces** and retained its own UV/bake workflow. The completed V139 release instead keeps the higher-density repaired scan mesh because that lineage best preserves the reconstructed vessel.

The current canonical files are:

- `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.blend`
- `reconstruction/v4/blender/Thai_Libation_Vessel_V4_FINAL.glb`

The final V7 authoring pass stayed on the reconstructed dense mesh rather than replacing vessel anatomy. A synthetic planar bottom cap was removed after comparison with preserved raw/CleanHigh evidence, leaving the real 72-edge physical opening. Lower-pedestal repair used only same-object CleanHigh geometry, and subsequent smoothing/fairing operated on reconstructed vertices with the opening boundary locked. The holder-to-middle-ring transition, bowl, globe, complete tower, upper ring, and terminal were inspected in segmented eight-angle closeups. The terminal was repaired from its existing scan vertices using bounded same-scan profile correction, local fairing, and robust sphere fitting; no primitive sphere, rebuilt finial, lathed profile, generic side, or external reference geometry was added.

The completed object is `SM_V4_V139_SCAN_PRESERVING_FINAL`. It contains **451,312 vertices** and **902,838 triangular faces**, one connected component, one intentional 72-edge boundary loop, zero other non-manifold edges, no zero-area faces, no loose vertices, and identity transforms. Its dimensions are approximately **0.89744 × 0.89870 × 2.09221 m**. The physical bottom boundary remains level. Appearance uses only `MAT_V4_V115_BrassStatsOnly`, derived from the verified 158-image uncoated set; photographic texture projection was not claimed.

V139 was promoted to the canonical `.blend` and exported as the canonical GLB. A fresh Blender 5.2 factory-startup process imported the GLB into an empty scene and verified exactly one mesh object with **451,312 vertices / 902,838 triangles**, the expected brass material, identity transforms, and matching dimensions. Canonical SHA-256 values are `6663d83303fefac34bbbee85132ad64322e2e9b0bc4db73776f2a882a9d07a0c` for `Thai_Libation_Vessel_V4_FINAL.blend` and `38a38dc17d23a4a19ecd991b6c4ff8023814b9d17fa6fbb13d0c78fe3e3f1ad6` for `Thai_Libation_Vessel_V4_FINAL.glb`.

## 18. Stage 13 — Brass appearance reconstruction

Geometry and appearance are handled separately because the geometry images contain temporary coating and markers.

The final appearance workflow uses the complete **158-image uncoated capture set** to derive reproducible brass appearance statistics. V139 uses one statistics-only material, `MAT_V4_V115_BrassStatsOnly`, with dataset-derived Base Color, metallic response, and dataset-derived roughness. Surface relief comes from the reconstructed mesh itself.

The earlier V6 asset retained topology-specific AO/normal-map work. Those maps are not reused on V139 because the V139 topology and UV lineage are different, and photographic projection was not independently verified. This keeps the final appearance tied to project evidence without attaching incompatible bakes or hand-reconstructed decoration.

## 19. Stage 14 — Export and clean re-import validation

The production asset is exported as GLB and then loaded into a fresh Blender scene for independent verification.

For the completed V139 release, the export gate checked geometry before export and then re-imported the canonical GLB in a fresh Blender 5.2 factory-startup process. The final checks include:

- exactly one exported mesh object;
- **451,312 vertices / 902,838 triangles** after re-import;
- finite geometry and normals;
- `MAT_V4_V115_BrassStatsOnly` present;
- identity location, rotation, and scale;
- dimensions matching the authoring object;
- absence of source, rollback, camera, light, or debug objects in the exported GLB.

The V139 QA record is stored under:

- `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/previews/v139_final_surface_qa/`;
- `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/previews/v139_final_brass_qa/`.

The older deterministic V6 render-comparison folders remain in `reconstruction/v4/previews/final_lod0_v1/` and `reconstruction/v4/previews/final_glb_reimport_v1/` as historical evidence; their pixel metrics are not presented as V139 measurements.

## 20. Reproducibility and evidence management

The project keeps the information required to trace major outputs back to their inputs. Important artifacts are accompanied by manifests, reports, hashes, or validation records.

The repository retains:

- immutable source photographs;
- capture and reconstruction manifests;
- selected sparse lineage evidence;
- selected dense reconstruction reports;
- selected Poisson mesh and evidence;
- representative visual previews;
- reconstruction and verification scripts;
- test coverage for important data and geometry contracts;
- editable Blender and portable GLB outputs.

Large temporary stereo workspaces, caches, repeated diagnostics, and other intermediate outputs are not treated as project deliverables and are excluded once the selected lineage has been preserved.

## 21. Project directory map

```text
CSX4213_Project_V4_Images/
    688 authoritative V4 photographs (Git LFS)

capture_v4/manifests/
    Capture roles, sequence metadata, exclusions, and media audit

capture_v4/derived/
    372 MVS images, 372 vessel masks, and 372 feature masks (Git LFS)

docs/
    Current methodology, project documentation, and this report

reconstruction/v4/sparse/
    Sparse reconstruction artifacts

reconstruction/v4/dense/
    Selected fused dense point cloud

reconstruction/v4/mesh/
    Selected Poisson surface

reconstruction/v4/previews/
    Representative sparse, dense, mesh, and final-asset evidence

reconstruction/v4/blender/
    Blender authoring, canonical asset, GLB output, and current refinement workspace

scripts/
    Reconstruction, auditing, meshing, Blender, and verification utilities

tests/
    Regression and pipeline validation tests
```

## 22. Main technologies used

### Image and data processing

- Python
- PyTorch
- OpenCV
- FFmpeg
- ExifTool

### Segmentation

- Grounding DINO-T
- SAM 2.1

### Feature extraction and matching

- ALIKED-N16Rot
- LightGlue

### 3D reconstruction

- pyCOLMAP
- COLMAP 4.2
- CUDA PatchMatch Stereo
- COLMAP Stereo Fusion
- COLMAP Poisson Mesher

### 3D asset preparation

- Blender 5.2
- PBR material workflow
- UV atlas generation
- ambient-occlusion baking
- tangent-space normal/detail baking
- glTF/GLB export

## 23. What the project has accomplished

The project now provides a complete, evidence-backed image-to-3D workflow for a difficult reflective cultural object. The implemented system has:

- established a purpose-built multi-view capture protocol for a reflective brass vessel;
- separated geometry and appearance acquisition into appropriate image groups;
- produced full-resolution object masks for the V4 geometry sequence;
- integrated ALIKED-N16Rot and LightGlue into the reconstruction workflow;
- performed geometric verification and sparse camera reconstruction with COLMAP;
- registered 372 views in the selected sparse reconstruction;
- produced a CUDA PatchMatch dense reconstruction and a fused cloud of about 1.93 million points;
- generated a Poisson surface containing about 4.95 million vertices before production reduction;
- prepared a practical LOD0 mesh while preserving scan-derived geometry;
- derived brass appearance from uncoated project photographs;
- produced editable Blender and portable GLB deliverables;
- verified GLB export through a clean re-import and deterministic multi-view comparison;
- preserved compact visual and machine-readable evidence for the major reconstruction stages.

## 24. Conclusion

This project demonstrates a complete computer vision approach to reconstructing a real reflective object from photographs. The final workflow combines controlled acquisition, segmentation, learned feature matching, sparse geometry, dense multi-view stereo, surface reconstruction, and reproducible 3D asset preparation.

The central design principle is that the vessel shape remains image-derived. The 3D preparation stage improves usability, UVs, baked detail, material presentation, and export quality without replacing the reconstructed object with manually authored geometry. This makes the project both a computer vision reconstruction study and a practical pipeline for converting real-image evidence into a reusable 3D asset.
