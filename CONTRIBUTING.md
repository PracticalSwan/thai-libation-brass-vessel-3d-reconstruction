# Contributing

This is a collaborative CSX4213 coursework repository. Keep changes small, explainable, and reproducible.

## Before changing code

1. Read `AGENTS.md` and `CLAUDE.md`.
2. Inspect `git status` and preserve unrelated contributor work.
3. Keep the versioned files in `IMG20260826122949/` immutable. Never edit them in place.
4. Do not over-engineer, over-complicate, or over-test.

## Development rules

- Prefer focused changes that directly support preprocessing, pyCOLMAP reconstruction, meshing/texturing, documentation, or verification.
- Preserve image geometry during preprocessing.
- Add dependencies only when the existing stack cannot reasonably perform the task.
- Use clear Python and expose important computer-vision parameters so the workflow can be explained in class.
- Do not commit secrets or disposable temporary/test residue. Raw capture images and intentional image-processing evidence are expected repository content; cleanup obsolete experiments before committing.

## Verification

Run the narrow checks appropriate to the change. Python changes should at minimum compile, and behavior changes should exercise the real affected path. Do not add exhaustive tests for low-risk glue code simply to increase test count.

Before committing, inspect the exact staged diff and remove task-created temporary/test residue according to `AGENTS.md`.

## Commits

Use concise descriptive commits, for example:

- `feat(preprocessing): add image quality analysis`
- `fix(preprocessing): preserve accepted image coverage`
- `docs: document pycolmap workflow`

Do not force-push or rewrite shared history without explicit team agreement.
