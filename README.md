# Metabolic Twin Benchmark

Code accompanying *"Trajectory fit alone does not identify causal edges: a white-box masked ODE benchmark for trans-omics digital twins of drug action mechanisms"*.

This repository contains the pipeline that builds the candidate-edge prior masks, generates the semi-synthetic ground-truth universe, trains the masked ODE, and runs the evaluation and audit instruments reported in the paper.

The code here is the exact version used to produce the results in the paper. Running the steps below reproduces the data artifacts (final analysis-input files verified unchanged across two builds) and the training and evaluation workflow reported in the paper (commands and seeds in `docs/ARM_LIST.md`).

## What is (and is not) distributed here

**Distributed:** all code and documentation. TRRUST v2 (mouse) is distributed under CC BY-SA 4.0 (see LICENSES.md).

**Not distributed:** the candidate-edge masks and all data derived from them — including the semi-synthetic ground-truth networks and tensors. To avoid redistributing KEGG content, raw KEGG downloads and the candidate-edge masks and data derived from them are not included; the pipeline provides scripts to obtain the required inputs and regenerate these artifacts locally. The working directory `GSE166336_小鼠组学数据/` is not shipped; the pipeline creates it where you run it.

**Version verification.** The final analysis-input files were verified to be unchanged between two independent builds (30–31 August 2026 and 13 November 2026; record in docs/KEGG_version_record.md). KEGG is continuously updated; later downloads may yield different derived files (e.g., 4 additional edges in the full 193-compound metabolite-metabolite mask, outside the 33-metabolite working system); updates between those dates changed none of the results reported in the paper.

**Scope.** The exploratory rat-cohort re-analysis (paper section 2.10) is not part of the released pipeline. Trained parameters and per-arm training logs are not part of the release. The feature-wise z-statistics mentioned in the paper's Methods are not distributed; they are regenerated from the public GSE166336 supplement by the pipeline (step 2).

## Environments

Two Python environments are used (full pins in `docs/pip_freeze_venv.txt` and `docs/pip_freeze_dyn.txt`):

- mask construction, semi-synthetic generation, training, evaluation: the `venv` environment (numpy, pandas, requests, torch);
- real-data tensor preparation from the Excel supplement: the `dyn` environment (adds openpyxl).

## Pipeline order

    0. Inputs you must obtain yourself (fields per file: URL, download date,
       local landing path — see docs/DATA_SOURCES.md):
       - TRRUST v2 mouse (also shipped in data/)
       - GSE166336 supplement mmc3.xlsx (GEO)
       - KEGG: nothing to download manually; the pipeline fetches it via the
         free KEGG API (docs/kegg_kgml is created by
         code/build_masks/step2_download_kgml.py)

    1. Build the candidate masks (code/build_masks/):
       python step3_build_node_index.py            # node index
       python step2_download_kgml.py               # 13 mouse pathways -&gt; kegg_kgml/
       python ECrel_mask_build_corrected.py        # enzyme-metabolite mask (ECrel main)
       python ECrel_reaction_supplement.py         # +590 reaction-level edges
       python mask_MM_three_layer.py               # metabolite-metabolite mask
       python feedback_mask_build.py               # feedback mask
       python GErel_mask_build.py                  # gene-gene mask (KGML + TRRUST)
       python beta_translation_check.py            # QA

       First run requires internet (KEGG API). Responses are cached in
       GSE166336_小鼠组学数据/kegg_api_cache.json; reruns are offline.
       PYTHONHASHSEED=0; the 590-edge supplement step uses 621
       (see Known determinism caveats).

    2. Prepare real-data tensors (code/prepare_real/, dyn environment):
       python prepare_tensors_from_mmc3.py
       # writes train/validation tensors into the working directory

    3. Generate the semi-synthetic universe (code/generate/, PYTHONHASHSEED=0):
       python semisynth_simulation.py              # ground_truth.npz + train_tensors.npz
       (Hill and delayed families: hill_wrapper_v1.0.py, delay_wrapper_v1.2_1005.py)

    4. Train (code/train/, PYTHONHASHSEED=0; the edge-supplementation step uses 621):
       python train_twin.py --scheme 1 --tensor_file .../train_tensors.npz --tag _sim
       per-arm commands and seeds: docs/ARM_LIST.md

    5. Evaluate and audit (code/evaluate/):
       edge_recovery_eval.py  static_LASSO_baseline.py  KO_chain (v3-v9, triage)
       saturation_audit.py  independent_validation.py

## Known determinism caveats (read before re-running)

The archived pipeline has three latent hash-seed dependencies plus one insertion-order variant; all are handled by the utilities in `code/utils/`:

1. `ECrel` main script batches gene queries via `list(set(...))`; cache keys therefore depend on `PYTHONHASHSEED`. Reproduction uses `PYTHONHASHSEED=0` plus the shipped cache-warming utility (`code/utils/kegg_cache_preheater.py`), or a pre-warmed cache.
2. Edge directions from the ECrel API step are first-wins over a seed-dependent iteration order. The deterministic replayer (`code/utils/direction_finalizer.py`) regenerates them from the cache in file order (verified order-robust; reproduces the paper's 34/27/61 assignments).
3. The reaction-supplementation step stores ECs/reactions in `set()`; its traversal order depends on the hash seed. The build archived in the paper is reproduced with `PYTHONHASHSEED=621`.
4. The supplement step's equation-side annotation in the edge-list CSV can flip with cache insertion order (display-level only; direction values in the `.npy` files are unaffected).
5. In rare cases a downloaded KEGG response can be empty or malformed; if a mask build looks incomplete, delete `kegg_api_cache.json` and rerun with internet access.

If you regenerate masks from a *current* KEGG rather than the verified snapshot, expect small deviations; the two verification builds above are byte-identical.

## Score your own method

1. Fit your method on the `train_tensors.npz` generated in step 3 (or your own semi-synthetic universe generated with `semisynth_simulation.py`).
2. Export candidate-edge scores as a CSV with the exact header `regulator,target,score` (one row per candidate edge; regulators and targets use the node IDs registered in `gene_index_map.json` and `metabolite_index_map.json`). Rows with unregistered IDs are dropped; scores may be any real number, NaN rows are dropped.
3. Run `python code/evaluate/edge_recovery_eval.py --scores your_scores.csv --universe &lt;your_output_dir&gt;`; it reports edge-class AUROC / AUPRC / Top-k against the ground-truth network in `ground_truth.npz` (see `--help` for the exact interface).

## License

- Code: MIT.
- TRRUST v2 (mouse) and its derived edge table: CC BY-SA 4.0, (c) TRRUST authors; cite Han et al., Nucleic Acids Res. 2018; shipped in data/.
- KEGG raw content and KEGG-derived artifacts are not distributed (see above); KEGG (c) Kanehisa Laboratories, used under its academic terms — cite KEGG.
- Full matrix: LICENSES.md.

## Provenance

Build logs for every mask are in `docs/provenance/`; the KEGG version-verification record is `docs/KEGG_version_record.md`; per-file checksums are in `docs/指纹对照表.md` and MANIFEST. Per-arm training commands and seeds are in `docs/ARM_LIST.md`; the environment and configuration snapshot is in `docs/CONFIG_SNAPSHOT.md`.