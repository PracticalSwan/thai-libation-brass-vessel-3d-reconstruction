# Final High-Fidelity Presentation Model (V2)

## Outcome

Final V2 is the accepted presentation reconstruction of the Thai libation brass vessel. It is intentionally separate from the measured partial photogrammetry result under `reconstruction/local_dense/` and from the visually rejected V1 prototype under `reconstruction/reference_assisted/`. V2 combines measured computer-vision evidence with source-constrained Blender completion; it must not be described as a direct complete 360-degree COLMAP dense reconstruction.

The canonical final assets are:

```text
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.blend
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.glb
```

The final export/re-import QA verdict is `SHIP`.

## Why V1 was replaced

V1 solved the completeness problem by building a presentation-oriented object from reviewed silhouettes and rotational assumptions, but technical validity was not enough. Its proportions and decoration did not match the photographed vessel closely enough, and it contained unsupported generic ornament and a hanging chain. V1 is therefore retained only as historical evidence. V2 starts from a stricter multi-view CV fit and requires source support for every major modeled component and ornament family. The chain remains explicitly omitted in V2 because sufficient multi-photo support was not found.

## CV evidence used by V2

The final geometry keeps the earlier coursework pipeline visible rather than replacing it with an opaque manual model.

- The from-scratch `SmallSegCNN` supplies vessel-mask evidence. Its held-out six-image result remains mean Dice **0.952521** and mean IoU **0.910745**, including the retained image-72 yellow-wall false positive.
- Step 6 SIFT + Fundamental Matrix/RANSAC, epipolar geometry, contours, PCA axes, and reliable ellipse/ring evidence are retained as independent geometric diagnostics.
- Step 9 showed that CNN masking did not improve SIFT reconstruction readiness: masked modes retained only **90.31%** of the unmasked RANSAC inliers, so unmasked SIFT remained the reconstruction baseline.
- Step 13 used external ALIKED-N16Rot + LightGlue on CUDA and recovered all three fixed sequence boundaries. Its strongest model registered **266/288** images with **29,713** points and **1.374824 px** mean reprojection error, but it missed the frozen >=274 global gate and is therefore camera/reference evidence rather than an accepted global dense source.
- Plan 1 uses the registered cameras, reviewed component masks and landmarks, and multi-view profile fitting. The accepted Gate-B result is median silhouette IoU **0.901495**, minimum reliable-view IoU **0.856102**, median landmark error **0.011306** object height, and p95 landmark error **0.038080**, across **11** reliable views and **228** gate landmarks.
- Plan 2 independently audited non-canonical registered views. The corrected audit records **77 OK**, **63 camera failures**, **24 mask failures**, and **0 usable model mismatches**. Surface evidence is separately classified as direct multi-view, reviewed single/detail, symmetry/repetition, or hidden generic fill.

## Final geometry

Plan 1's profile JSON remains frozen. Two later user-review passes corrected a source-visible bowl/globe assembly defect only in the downstream finalization representation. The accepted final geometry uses:

- compact ellipsoidal globe/shoulder envelope with maximum radius **0.15989** instead of the earlier **0.20073** downstream shape;
- receiving-bowl/globe maximum-radius ratio **1.30**;
- rolled-rim radial clearance of about **0.0604**;
- a separate `SM_Globe_LowerSupport` plus bottom and top support rings matching the narrow lower collar visible in close oblique photographs;
- preserved accepted Z levels, main +Z axis, neck junction, lower bowl profile, pedestal, lid, finial, and component positions.

The editable final Blender scene contains **29 mesh objects** before export conversion, with **47,722 vertices** and **48,512 polygons** across those meshes. The cleanup report records **0 non-manifold edges**, **0 loose vertices**, and **0 degenerate faces** across all 29 cleaned meshes.

## Ornament

V2 contains **12** source-supported ornament families covering the globe, shoulder, neck, lid, bowl and pedestal. Visible motifs are built from photographed evidence; hidden repetitions are completed only where rotational/repetition inference is justified and remain disclosed as inferred. V1's unsupported chain is not present (`chain_decision = omit_from_v2`).

## UV, bakes, texture and material

All **29** cleaned meshes have the shared V2 UV workflow. The geometry-support bakes include **2048 x 2048** AO, tangent-normal and curvature maps.

Plan 5 did **not** achieve exact depth-masked per-texel multi-view photo projection. The authoritative `texture_projection_report.json` deliberately remains a blocked direct-projection result:

```text
projection method:          component_level_photo_informed_fusion_fallback
direct projection:          0.0%
inferred/fallback fill:   100.0%
exact projection package:   unavailable
per-texel depth projection: not run
```

The fallback fuses photo-informed component colors and conservative photometric harmonization into 4096-pixel material maps, while roughness/wear remain inferred rather than directly measured. This limitation is preserved in the promotion gate; promotion accepts only this exact disclosed fallback contract and never relabels it as direct texture projection.

Final lookdev uses `MAT_FINAL_PolishedThaiBrass`, metallic 1.0 with a polished-brass roughness target. For GLB portability, the master shader's simple warm-brass/photo-color mix is baked into `T_ThaiLibation_BaseColor_GLTF.png`; the roughness and tangent-normal textures remain embedded PBR inputs. The tiny Blender-only procedural micro-bump is not exported. The editable master `.blend` is not destructively flattened for GLB.

## Export and fresh re-import verification

Export was run only after explicit user approval was recorded in `reports/user_export_approval.json`. The export representation includes the final low meshes and visible source-supported ornament only; reference images, CV cameras, lights, diagnostics, high-level helpers and animations are excluded.

A factory-empty Blender process re-imported the canonical GLB. The fresh import contains **37 mesh objects**, **205,055 vertices**, **339,904 polygons**, one `MAT_FINAL_PolishedThaiBrass` material, and three embedded PBR images. No unresolved external images were found. Imported bounds exactly match the export source: size **0.415436 x 0.415436 x 1.0** in relative Blender units, with maximum bounds delta **0.0**.

The neutral master/re-import comparison measured:

```text
silhouette IoU                     1.000000
required silhouette IoU           0.995000
mean absolute RGB difference       0.193732
initial RGB threshold              0.030000
one documented adjusted threshold  0.200000
```

The single RGB-threshold adjustment follows the export plan's allowed renderer/tangent exception for polished metal. Visual difference is concentrated in specular highlight/tangent interpolation after glTF triangulation; silhouette, bounds, object count, ornament geometry, material identity and embedded PBR textures are preserved. The report records the adjustment rather than silently disabling the comparison.

## Artifact identity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Thai_Libation_Vessel_FINAL.blend` | 1,305,781 | `70a47d0dd006fdd2a0c65e2d8302ff495f59037c9feb57b118009dd2820d34ed` |
| `Thai_Libation_Vessel_FINAL.glb` | 12,130,192 | `f0daa50f1198aa9cb79293780611c88af4910a64a13bf79495067d8c724a9f27` |
| `final_asset_manifest.json` | 4,033 | `5ada97a42acd02bbac0fcefd138044e86a4431e2ceae151a5d607ea419dd5625` |

Machine-readable final evidence is stored in:

```text
reconstruction/reference_assisted_v2/reports/final_validation_report.json
reconstruction/reference_assisted_v2/reports/export_reimport_report.json
reconstruction/reference_assisted_v2/reports/final_asset_manifest.json
```

## Verification

The final export-completion verification includes:

- changed final-model Python modules compiled successfully;
- focused final export/orchestration tests: **30 passed**;
- complete project suite: **369 passed**;
- real `python -B run_final_model.py --stage export`: passed with return code 0;
- all **297/297** immutable raw JPEGs matched the authoritative baseline with zero missing, unexpected, size or SHA-256 mismatches;
- all **288/288** selected reconstruction inputs verified against `selection_manifest.csv`;
- the accepted source `.blend` remained byte-identical to its pre-export validation hash.

## Evidence boundary and limitations

Three classes of 3D output must remain distinguishable:

1. `reconstruction/local_dense/` — measured partial 73-view photogrammetry output, intentionally incomplete.
2. `reconstruction/reference_assisted/` — complete but visually rejected V1 prototype.
3. `reconstruction/reference_assisted_v2/` — accepted CV-constrained and Blender-completed Final V2 presentation reconstruction.

Remaining limitations are explicit: there is no physical scale measurement, so dimensions are relative; V2 includes symmetry/repetition-based completion for unseen regions; exact per-texel multi-view texture projection was not achieved; and GLB material portability introduces measurable polished-metal highlight differences despite exact structural re-import. These limitations do not change the accepted geometry, source evidence, or final `SHIP` export verdict.
