# SEM Microstructure → Heat Treatment Prediction

A multimodal machine learning pipeline that predicts steel heat treatment parameters
(intercritical annealing temperature and holding time) from SEM microstructure images
combined with chemical composition data.

---

## Problem

Given a scanning electron microscope image of a dual-phase (ferrite/martensite) steel
and its alloy composition, predict the heat treatment cycle that produced it —
specifically `Cycle1_HoldingTemp (°C)` and `Cycle1_HoldingTime (min)`.

---

## Pipeline overview

```
Raw SEM images  ──► Image cleaning         (remove annotations, scale bars)
                         │
                         ├──► CNN feature extraction   (14 pretrained backbones)
                         └──► Morphological extraction (33 interpretable features)

Metadata CSV    ──► Column sanitisation ──► Imputation ──► Scaling/encoding
                         (14 composition cols)

[CNN features | Morphological features | Tabular features]
                         │
                    Ensemble regressors
                  (RF · GBR · AdaBoost)
                         │
                  Bayesian hyperparameter tuning
```

---

## Results (latest run)

Best model: **Gradient Boosted Regressor** — test R² = **0.90**

| Target | R² | MAE |
|---|---|---|
| HoldingTemp (°C) | 0.84 | 31.4 °C |
| HoldingTime (min) | 0.97 | 3.1 min |

Dataset: 111 usable samples (79 train / 15 val / 17 test), 818 total features
(768 DINOv2-ViT-B/14 image + 33 morphological + 17 tabular).

---

## Quickstart

```bash
pip install -r requirements.txt

# Full pipeline (downloads images from Google Drive, extracts features, trains)
python main.py

# CI/CD training only (env-var configured, writes timestamped runs/ dir)
python run_training.py

# Pre-extract features without training
python prepare_features.py
```

Copy `config_example.json` → `config.json` and set your Google Drive credentials
before running.

---

## Feature streams

### Image features
Frozen pretrained CNN/ViT backbones extract embeddings per image.
14 backbones are registered; configure which to use in `config.json`:

```json
"extraction": { "backbones": ["dinov2_vitb14", "resnet50"] }
```

Available: `resnet18/50/101`, `vgg16/19`, `densenet121`, `efficientnet_b0/b4`,
`convnext_tiny`, `mobilenet_v3`, `dinov2_vits14/vitb14/vitl14`.

### Morphological features
33 interpretable descriptors extracted from the segmented microstructure:
phase fractions, ferrite grain geometry, martensite connectivity, boundary
network density, banding index, GLCM texture, LBP texture, and local contrast.
See [`docs/morphological_feature_extraction.md`](docs/morphological_feature_extraction.md).

### Tabular features
14 chemical composition columns (C, Mn, Si, Cr, Mo, Ni, Al, …).
Imputation is split by missingness mechanism — MICE for correlated elements
(Cr, Mo, S, Ni, Al), zero-fill + presence indicator for structural absences
(Ti, Nb, V), median for the rest.
See [`docs/imputation_design.md`](docs/imputation_design.md).

---

## Image cleaning

Raw SEM images contain annotation overlays (yellow F/M phase labels, red circles,
white scale bar text, panel letters) that corrupt segmentation and CNN features.
A colour-mask + inpainting pipeline removes them before any feature extraction.
See [`docs/image_cleaning_design.md`](docs/image_cleaning_design.md).

---

## Hyperparameter tuning

Bayesian optimisation (Optuna) searches over `n_estimators`, `learning_rate`,
`max_depth`, and preprocessing parameters as part of [`pipeline_benchmark.ipynb`](notebooks/pipeline_benchmark.ipynb).
Results are saved to `runs/hyperparams.json` and loaded automatically by subsequent runs.
See [`docs/bayesian_optimisation_design.md`](docs/bayesian_optimisation_design.md).

---

## Notebooks

### Recommended running order

```
1. image_cleaning_demo.ipynb       — verify annotation removal on raw images
2. imputation_validation.ipynb     — confirm MICE strategy before training
3. prepare_features.ipynb          — extract + cache CNN and morphological features
4. pipeline_benchmark.ipynb        — sweep preprocessing × regressors × backbones,
                                     runs Bayesian tuning and writes hyperparams.json
5. microstructure_demo.ipynb       — end-to-end training run using best config
6. metrics_history.ipynb           — plot R² improvement across runs
```

`microstructure_generator.ipynb` and `imbalance_exploration.ipynb` are standalone
and can be run at any time independently.

### All notebooks

| Notebook | Purpose |
|---|---|
| [`image_cleaning_demo.ipynb`](notebooks/image_cleaning_demo.ipynb) | Before/after annotation removal validation |
| [`imputation_validation.ipynb`](notebooks/imputation_validation.ipynb) | MICE vs median imputation comparison |
| [`prepare_features.ipynb`](notebooks/prepare_features.ipynb) | CNN + morphological feature cache builder |
| [`pipeline_benchmark.ipynb`](notebooks/pipeline_benchmark.ipynb) | Preprocessing × regressor × backbone sweep + Bayesian tuning |
| [`microstructure_demo.ipynb`](notebooks/microstructure_demo.ipynb) | End-to-end pipeline walkthrough |
| [`metrics_history.ipynb`](notebooks/metrics_history.ipynb) | R² improvement over runs |
| [`microstructure_generator.ipynb`](notebooks/microstructure_generator.ipynb) | Synthetic SEM image generation (fractal + fuzzy logic) |
| [`imbalance_exploration.ipynb`](notebooks/imbalance_exploration.ipynb) | Dataset balance analysis |

---

## Repository structure

```
src/
  extraction/          — CNN backbones, morphological extractor
  preprocessing/       — imputation, scaling, encoding, image cleaner
  model_trainer.py     — ensemble training + evaluation
  features.py          — FeaturePipeline (download → extract → load)
datasets/
  metadata_latest.csv  — alloy composition + heat treatment labels
docs/                  — design documents for each pipeline component
runs/                  — timestamped training artefacts (metrics, plots, models)
notebooks/             — exploratory and validation notebooks
```

---

## Dependencies

```
torch · torchvision · scikit-learn · scikit-image
scipy · numpy · pandas · optuna · Pillow · tqdm
google-auth · google-api-python-client
```

Python ≥ 3.10 required.
