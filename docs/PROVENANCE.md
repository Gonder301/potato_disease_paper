# Provenance and limits

The release was assembled from the audited corrected v5 delivery and the final Kaggle result archive. The original archive, training engine and runtime are preserved without scientific changes. The new CPU reanalysis, training launch, preparation and figure-rendering adapters provide paths that work outside the original workspace. Their output is checked against the archived tables; they do not replace or manufacture experimental predictions.

## Original records

- Experiment signature: `773af3d89fa82d512df8da024e151be6df88ed9a06bb4ff40797612038c6824c`.
- Original final results SHA256: `501c4214e73583c628e799659b7e3a69b7aeaab38af28706e374c85ac3d71874`.
- `analysis/reuse_audit.csv` identifies the unchanged 60 laboratory random runs reused from v4.2.
- `analysis/auditoria.json` records the original audit, including comparison with the earlier 200 completed predictions. Its historical comparisons required earlier session files and are not newly claimed by a reanalysis of the final archive alone.
- The final archive retains `complete.json`, `history.csv` and `predictions.csv` per run, the original environments, the frozen state, all 60 partitions and the execution log. Paths referring to `/kaggle/` are historical; image lookup checks SHA256 when using a new input directory.
- `provenance/original_scripts/` contains selected preparation and checking scripts from the original workspace. Their project-relative paths require adaptation. The new documented commands use the published frozen records instead.

## Interpretation

The plan is 4 architectures × 2 datasets × 2 strategies × 15 seeds = 240 runs. The seeds index paired experimental settings; test images are not matched across strategies and repeated holdouts overlap. The laboratory task and field task also have different source categories. Uncertainty is conditional on these datasets, not independent farm-level replication.

Laboratory identities are externally published metadata. Field groups are visual clusters without a human near-duplicate reference, and the threshold was not independently calibrated. Neither zero SHA256 exposure nor visual grouping establishes the absence of all leakage. The observed gaps do not causally isolate duplicate effects. No exploratory ViT-versus-mean-CNN contrast survives its separate correction, so the study does not establish a general CNN/transformer robustness advantage.

The revised experiment was designed after reviewing earlier results; it was not preregistered. Manual field validation remains unavailable. The absence of final weights and immutable pretrained revisions limits future training replication, but does not prevent reanalysis of the saved predictions.

## Scope of public material

Only scientific assets were selected for this deposit. Private reviewer correspondence, proof files and author-administration documents are not included in the corrected package. The accepted article is linked at Frontiers; this deposit does not purport to replace the final typeset publication or its author metadata.
