# Reproducing the corrected study

The recorded v5 experiment is defined by its frozen image manifest, 60 split files and scientific specification. Its signature is `773af3d89fa82d512df8da024e151be6df88ed9a06bb4ff40797612038c6824c`. Replacing the split solver run is not equivalent to using these recorded partitions.

## Recalculate results without a GPU

Use Python 3.12 or newer in a virtual environment. From the repository root:

```bash
python -m pip install -r requirements-reanalysis.txt
python code/reanalyze.py --output reanalysis_output
```

Choose a new output directory for each execution. No Kaggle account, complete image collection, model weights, PyTorch or GPU is needed. The command checks the release file hashes and original result archive, extracts the archive inside the new output directory, validates the frozen protocol, test IDs, prediction hashes, labels, class probabilities, all 240 epoch histories and all 60 partitions, and recomputes the primary and exploratory statistics. It also checks the exposure and allocation tables, laboratory identity diagnostics, and the eight illustrative sample hashes. The numeric comparison tolerance is `atol=rtol=1e-12`.

Expected outcome: `verification.json` has `status: PASS`, 240 completed runs, 62,160 prediction rows, 24 primary comparisons with 17 q-values below 0.05, and six exploratory interactions with zero q-values below 0.05. Newly calculated tables are saved beside that report. The extracted inputs are retained for inspection. These are checks on recorded predictions, not a rerun of the image classifiers.

`reanalysis.ipynb` runs the same CPU procedure. The original reanalysis adapter, which imports the complete training dependencies, is also preserved as `code/reanalyze_v5.py`.

## Reconstruct classifier training on the frozen partitions

The original study was reconstructed after inspection of v4.2 results. Sixty laboratory random runs were reused without changing their predictions, and 180 further runs completed v5. The original continuation notebook is retained at `notebooks/Continuar_v5_desde_200_corridas.ipynb`; it was used after 200 runs were available and is not a standalone raw-data-to-240-runs notebook.

The repository-root `experimentos.ipynb` is a new launch wrapper around the preserved engine. It prepares an experiment from the published archive and uses the frozen assignments. Its default configuration reuses the same 60 laboratory random runs and schedules the remaining 180. Turning reuse off schedules 240 new runs, which are a new reconstruction rather than the original recorded results.

Attach these image distributions to Kaggle and enable Internet and a GPU:

- https://www.kaggle.com/datasets/arjuntejaswi/plant-village
- https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset

Do not add the supplemental Sankalana Healthy images. Image discovery verifies every image against the SHA256 in the frozen manifest, rather than trusting a directory name. A changed Kaggle distribution may fail that check and must not silently replace the recorded image bytes.

The recorded Linux environment used Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, timm 1.0.26, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, SciPy 1.16.3 and ImageHash 4.3.2. See `analysis/environment.json` and `requirements-training.txt`. PyTorch and torchvision must be installed from the appropriate official CUDA index for that environment; they are intentionally not replaced by an unqualified pip command in the launch notebook. The runtime checks the recorded package versions before reusing runs and checks CUDA before training.

Equivalent commands after preparing the recorded training environment:

```bash
python code/prepare_training.py --output frozen_state --reuse-lab-random
python code/run_training.py --state frozen_state --images /kaggle/input --output /kaggle/working/RESULTS_closure --session-hours 8
```

For an entirely new reconstruction, omit `--reuse-lab-random` and choose a different output directory. The preparation step copies only the frozen specification, assignments, reference metadata and environment, plus the 60 original runs when requested. It never edits the published archive. It does not recompute embeddings, relabel images or solve fresh partitions. `runtime_v5.py` resolves and checks image bytes before any incomplete run is trained.

The engine logs image checks, model checks, batches, epochs, validation, checkpoint saves, completed runs and an approximate remaining time. A heartbeat means the process responds; it does not certify training progress. With an eight-hour budget, the default fifteen-minute reserve targets a controlled pause after 7 h 45 min. Several sessions may be required. Re-run the command with the same output to resume a local session, or attach an exported `results_closure_v5.zip` and set `--state` to its extracted `RESULTS_closure` directory in the next session. Keep only one intended source state attached or specify it explicitly.

Checkpointing is at completed epochs. An interrupted partial epoch repeats. Completed-run predictions are checked and skipped. Final export omits checkpoints of completed runs. No final trained model weights are included in the published results. Immutable pretrained-weight revisions and hashes were not recorded, so future training is not guaranteed to be bitwise identical.

## Group definitions and identity validation

Laboratory training and evaluation use the published PlantVillage leaf IDs in `reference/leaf-map-pinned.json`. The source revision is `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, and its SHA256 is recorded in `analysis/reference_provenance.json`. `reference/plantvillage_referencia_hojas.csv` maps all 2,152 retained laboratory images to 538 identities.

Field groups use average-linkage cosine clustering of frozen DINOv2 embeddings at threshold 0.10, recomputed on the retained single-source field set. That threshold is a declared heuristic. It is not an independently calibrated or manually validated near-duplicate detector. Group membership and allocation used by every classifier are recorded in the frozen files.

`reference/v42_historico/` preserves the earlier manifest, embedding caches, visual clusters and partitions needed to trace the diagnostic comparison. Those earlier mixed-source field and laboratory visual-group splits are not the corrected classifier test partitions. Some inherited engine helper defaults also describe that earlier preparation workflow; use `runtime_v5.py` with the frozen v5 state, not the engine's historical scanner defaults.

`provenance/original_scripts/` preserves scripts used in the original workspace. They contain historical project-relative paths and are provided for provenance. They are not the portable entry points documented above. The new launch wrappers do not alter `engine_v5.py` or `runtime_v5.py`.

## Figures

`figures/` includes all four figures, both t-SNE coordinate tables, projection parameters and sample provenance. `samples/` contains only the eight illustrated source images, verified byte-for-byte against the experimental hashes. They illustrate published laboratory identities, not human annotations of field duplicates. Full image datasets and final model weights are not redistributed.

Regenerate figures from their published numeric coordinates, source samples and statistics with:

```bash
python -m pip install -r requirements-figures.txt
python code/render_figures.py --output regenerated_figures
```

The drawing adapter preserves the published coordinates and does not refit t-SNE or extract new embeddings. Historical scripts and embedding caches are retained for the provenance of the original projections. Rendering can differ across fonts and plotting versions; the original figure files remain available.
