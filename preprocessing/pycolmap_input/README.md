# Final pyCOLMAP Input Set

Use every image in `images/` as the input to the next pyCOLMAP stage.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames

The variant was selected from the representative SIFT matching evidence in
`../reports/sift_matching.json`. `WARN` images remain included because a
warning is a review signal, not an automatic rejection. Do not run pyCOLMAP
until the preprocessing milestone and raw-hash verification are accepted.
