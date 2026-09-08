# Reference-Assisted Presentation Model

## Outcome

A separate reference-assisted V1 presentation model was generated after the direct photogrammetry result proved too fragmented for a useful complete asset. The original Steps 1-17 evidence remains unchanged. V1 is not presented as COLMAP-reconstructed geometry. Although it passed its own technical pre-clean gate, the user later rejected it because it does not look sufficiently like the photographed object; it is now preserved as a failed visual-fidelity prototype rather than the final presentation asset.

The V1 pre-clean asset is:

```text
reconstruction/reference_assisted/Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.blend
reconstruction/reference_assisted/Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.glb
```

`manual_cleanup_started=false`. No manual Blender sculpting, hole filling, mesh deletion, remeshing, or hand cleanup was performed in this phase.

## Why a separate rebuild was needed

The accepted Step 17 photo-textured mesh remains valid reconstruction evidence, but it contains 21 connected components and its largest component contains only 36.49% of faces. Large vessel regions are absent because dense reconstruction used the frozen 73-view local sparse component. Repairing that surface into a complete object would require inventing large amounts of geometry while still calling the result photogrammetry.

The presentation rebuild therefore uses a different, explicitly disclosed method: reviewed image silhouettes constrain the main proportions, rotational symmetry completes the round manufactured forms, photographs constrain the brass appearance, and repeated decoration is estimated from visible motifs.

## Reference evidence

Eight reviewed `normal_side` segmentation masks were used as the primary shape references:

```text
selected indices: 3, 10, 19, 28, 39, 50, 62, 72
```

Every mask hash was checked against `ml_dataset/manifest.csv` before measurement. Corresponding raw photographs were read from the immutable `IMG20260826122949/` source directory without modification.

Measured external width-to-height ratios across the eight masks ranged from 0.4025 to 0.4439. The aggregate silhouette measured:

```text
median external width / height = 0.4123424274
```

The source-photo warm-metal sample produced the median sRGB reference:

```text
R = 0.478431
G = 0.337255
B = 0.092157
```

No defensible physical measurement exists in the project, so Blender units remain relative rather than claimed centimeters or millimeters.

## Geometry method

`reference_assisted_model.py` measures reviewed masks and creates a deterministic reference report. `build_reference_model_blender.py` then generates new geometry in Blender from radial profiles rather than importing the broken Poisson mesh.

The main manufactured forms are rotationally modeled as intentional physical components:

- receiving bowl and pedestal;
- hollow-neck water vessel body;
- lid;
- finial;
- bowl cavity surface;
- stepped decorative rings;
- repeated relief ornaments;
- linked hanging chain.

The bowl, vessel, lid, finial, ornament pieces, and chain links are intentionally separate parts. This is different from the accidental disconnected fragments in the photogrammetry mesh. Each generated mesh object is closed/manifold under the automated edge check.

The final pre-clean build contains:

| Measurement | Value |
|---|---:|
| Mesh objects | 136 |
| Vertices | 65,882 |
| Polygons | 67,082 |
| Mesh objects with non-manifold edges | 0 |
| Manual cleanup started | No |

The GLB export applies Blender modifiers and triangulates/export-converts geometry, so its imported vertex/triangle counts are expected to differ from the editable `.blend` source counts.

## Material and inferred decoration

The brass material uses the measured photograph color as its starting point with metallic response, roughness `0.24`, controlled color variation, and fine procedural bump. Direct photographic highlights are not baked into base color because the original polished brass contains strong view-dependent reflections.

Visible construction features from the photographs informed stepped neck/lid rings, bowl bands, globe relief, neck relief, and other V1 details. V1 also included an inferred hanging chain plus repeated oval/lotus and diamond relief. Subsequent source-photo review did not find sufficient support for the large chain or the generic oval/diamond treatment, so V2 must omit those features unless independent image evidence proves them. These V1 elements must not be described as directly recovered texture or measured relief geometry.

No third-party texture was required for the accepted build. This avoids external licensing ambiguity and keeps the final appearance reproducible from project evidence plus deterministic Blender nodes.

## Automated validation

The final integrated command completed successfully:

```powershell
python -B run_reference_assisted_model.py --stage all
```

The validator renders a transparent orthographic silhouette of only the primary measured forms, excluding inferred ornament and chain, and compares it with the aggregate of the eight reviewed side masks after common height normalization.

Final result:

```text
silhouette IoU = 0.8060086279
required IoU   = 0.7000000000
accepted       = true
```

The four final beauty renders were opened and visually inspected:

```text
reconstruction/reference_assisted/previews/reference_front.png
reconstruction/reference_assisted/previews/reference_quarter.png
reconstruction/reference_assisted/previews/reference_side.png
reconstruction/reference_assisted/previews/reference_top_oblique.png
```

They show a complete bowl, globe, neck, stepped lid, finial, decoration, and linked chain rather than disconnected reconstruction ribbons.

Final verification also passed:

```text
5 / 5 focused reference-assisted tests
252 / 252 complete project tests
Python compilation for the new regular-Python modules/tests
```

## Artifact identity

Final pre-clean files:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.blend` | 764,914 | `9d04321e0d7e34aaf5dd52f8a2c935763d293fdeb724c46620b7f169c448a8c6` |
| `Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.glb` | 2,981,036 | `14fffd7afe294727055052534d6daee52671481ca3485febe196c1a79227ad77` |

Machine-readable evidence is stored in:

```text
reconstruction/reference_assisted/reports/reference_report.json
reconstruction/reference_assisted/build_report.json
reconstruction/reference_assisted/reports/validation_report.json
```

## Evidence boundary

Two different 3D outputs now exist and must remain distinguishable:

1. `reconstruction/local_dense/` is the measured Steps 14-17 photogrammetry result. Its incompleteness remains visible and documented.
2. `reconstruction/reference_assisted/` is a complete but visually rejected V1 presentation-oriented reconstruction constrained by reviewed photographs, segmentation silhouettes, rotational symmetry, and explicitly inferred decoration.

The second asset is useful as historical/prototype evidence but is not accepted as the final presentation model and must not be reported as a direct 360-degree dense reconstruction.

## Next phase

V1's requested pre-clean stop boundary was reached without manual Blender cleaning/sculpting. The user subsequently rejected V1's visual identity. The next phase is the separately designed V2 high-fidelity CV-constrained rebuild defined by `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md` and the `docs/superpowers/plans/2026-09-06-final-*.md` plan set. Those plans are complete; V2 implementation has not started. V2 may perform Blender cleaning/sculpting only after its CV-derived base geometry and source-derived ornament gates pass.
