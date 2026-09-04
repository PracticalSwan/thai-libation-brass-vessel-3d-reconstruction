# Reconstruction Input v1

This directory records the Step 9C recommended reconstruction subset. It intentionally does **not** duplicate the 288 PREPROCESSED JPEGs. `manifest.csv` references the verified files already stored under `../pycolmap_input/images/`.

Measured readiness decision:

- feature mode: `unmasked`
- adjacent edges: 287
- strong adjacent edges: 273
- weak adjacent edges: 14
- included images: 288
- excluded weak-but-bridged images: 0

An image is excluded only when neither incident adjacent edge is strong and its immediate predecessor/successor have a strong skip bridge. Weak images needed to preserve sequence coverage remain included.

This is readiness metadata only. Step 9 did not run pyCOLMAP or any reconstruction.
