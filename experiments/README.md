# Research scripts

These are the team's original scripts from the final version of the project (branch `Lander`, May 2025), plus the LLM concept-selection experiment from a side branch. They produced the numbers and figures in the report. They are kept as they were, so they still contain commented-out variants and hard-coded settings. For a clean, runnable version of the method, use the `dgam/` package in the repository root.

The scripts expect the datasets under `./data/` relative to the folder you run them from, with the layout described in the main README. They need CLIP and a few more packages (`pip install -r experiments/requirements.txt`). Several scripts select a GPU backend at the top (`"mps"` for Apple silicon, or `torch_directml` for AMD on Windows); change this to `"cuda"` or `"cpu"` for your machine.

## Labelling method (DGAM)

| Script | What it does |
| --- | --- |
| `Mapping.py`, `Mapping_preprocessing.py`, `Mapping_all.py` | Step 1: CLIP embeddings and image-concept cosine similarities, for the raw attribute names, for the attributes rewritten as sentences, and for all variants at once. |
| `Histogram.py` | Similarity histograms per concept, used to design the fit. |
| `Gaussian.py` | Steps 2–4: double Gaussian fit, correction step, concept score and ranking, per-concept thresholds, and the evaluation and plots for the report. |
| `rankingofzo.py` | Compares the ranking by distance between the two Gaussian means with a second ranking of the CUB concepts. |

## Baseline (Stochastic CBM)

| Script | What it does |
| --- | --- |
| `negative_similarity.py` | Labels a concept as present when the image is closer to the positive phrase ("has blue wings") than to the negative one ("does not have blue wings"), and reports its balanced accuracy. |
| `Mapping_negative.py`, `Mapping_preprocessing_negative.py`, `Histogram_negative.py` | Similarities and histograms for the negative phrases. |

## Supervised extension (report appendix A)

| Script | What it does |
| --- | --- |
| `main.py` | Entry point: builds the image-concept datasets and trains the networks (1)–(4) from the appendix. |
| `datasets.py`, `sampling.py` | Dataset classes for CUB, VOC and CelebA, and over- and undersampling for the class imbalance. |
| `model.py`, `transformer_model.py` | Classifier networks. |
| `train.py`, `metrics.py`, `visualization.py` | Training loop with weighted loss, metrics and training curves. |
| `siamees.py` | The Siamese network (5). |
| `CBM.py` | A supervised concept predictor: one network from the CLIP image embedding to all 312 CUB concepts. |
| `dataset_tester.py`, `CLIP_tester.py`, `Output_plotter.py` | Small checks and plotting helpers. |

## LLM concept selection (report appendix B)

`llm_concept_selection/` holds the prompts given to GPT-4o (`prompt*.txt`), the concepts it kept per prompt and run (`*_results/*.csv`), the per-concept balanced accuracies they are compared against, and `AUC_plot.py`, which draws the ROC curves (`*_results/ROC_*.svg`).

## Stored outputs

`outputs/` contains the small text results of the final runs: concept scores and rankings (`gaussian_rankings/`), thresholds and balanced accuracy per concept for DGAM and the baseline (`optimal_thresholds_*`), the comparison with the ground-truth ranking (`*_rank_comparison/`), and the training curves and final test metrics of the supervised network (`processed_concepts_full_run/`). The CLIP similarities and embeddings themselves (about 250 MB) are not included.
