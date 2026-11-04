# Repository Structure

`Pixel-Artifacts/` is the only main repository. It sits inside the single workspace root `D:\ARYAN\Projects\Major Project\Implementation`. Workspace-level organisation (datasets, legacy code, backups) is described in `WORKSPACE_ORGANIZATION.md`.

## Layout

```
Pixel-Artifacts/
├── src/
│   ├── paths.py              # resolves configs/paths.toml against the workspace root
│   ├── data/                 # sample sources (folder walk, in-place split-archive reader)
│   ├── models/               # head, loss, bundle save/load            (Phase 3)
│   ├── features/             # fp-v3 extractors, moved verbatim, hash-guarded (Phase 3)
│   ├── training/             # fp-v4 trainer                          (Phase 3)
│   ├── evaluation/           # clean / per-generator / robustness / masking (Phase 6–8)
│   └── explainability/       # Grad-CAM, pillar evidence              (Phase 9)
├── backend/  frontend/       # imported from five-pillar-detector at Phase 10
├── configs/
│   └── paths.toml            # every dataset / model / output location, workspace-relative
├── scripts/                  # check_workspace.py, build_dataset_manifests.py, check_leakage.py
├── tests/
├── docs/{architecture,datasets,experiments,evaluation,research}/
├── models/{fp-v3.0,fp-v4.0}/ # weights gitignored; manifests + checksums committed
├── results/{training,validation,testing,robustness,ablation,explainability,figures}/
├── CHANGELOG.md
└── README.md
```

`src/data/` is an addition to the requested tree. Dataset access is its own concern and is shared by training and evaluation.

## Rules

- **Paths.** No script hardcodes a drive path. Each one imports `src.paths` and reads `configs/paths.toml`. A reorganisation changes only that file.
- **Dataset manifests** live with the data (`DataSet/Our Dataset/CNN/manifests/`), not in the repo, because they describe the dataset rather than an experiment. Experiments record the manifest SHA-256 they used.
- **Gitignored:** weights, feature caches, datasets and large result tables. Only manifests and SHA-256 checksums are committed.
- **fp-v3.0** is copied in read-only and verified against `CHECKSUMS.json` (bundle `9cd68987…2430`). The extractor code moves without edits, and a test asserts the feature-code hash.
- Legacy folders remain available read-only until migration is confirmed.
