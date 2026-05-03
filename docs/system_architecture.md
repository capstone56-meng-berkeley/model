# 4. System Architecture

## 4.1 Overview

The system predicts two heat-treatment parameters — **HoldingTemp (°C)** and **HoldingTime (min)** — from scanning electron micrographs (SEM) of dual-phase steel microstructures paired with bulk chemical composition. The dataset is small by deep-learning standards (~111 unique alloy/treatment recipes, ~500 images after augmentation), so the architectural philosophy is **frozen, pretrained large feature extractors driving a lightweight learned regressor**: roughly 86 M frozen parameters supply representations to ~12 K trainable parameters, a ratio of ~7,000 : 1. This decoupling lets us re-train the regressor in seconds while caching the expensive image embeddings on disk.

The end-to-end pipeline is:

```
SEM image folder + metadata.csv
        │
        ▼
┌──────────────────────────────────────────────┐
│  Data layer: download → align → split        │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Feature extraction (three streams)          │
│   A) Tabular  : 14 chemistry cols → ~150 ft │
│   B) Morph    : 9-stage CV pipeline → 33 ft │
│   C) Image    : DINOv2 ViT-B/14 → 768 ft    │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Fusion (early concat by default)            │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Regressor: MultiOutputRegressor(GBR / XGB / │
│   KNN), Bayesian-tuned via skopt             │
└──────────────────────────────────────────────┘
        │
        ▼
   {HoldingTemp, HoldingTime} predictions
        +  runs/<timestamp>/ artefacts
```

**Repository layout:**

| Path | Role |
|---|---|
| [src/](../src/) | Core library: config, data loading, preprocessing, extraction, training |
| [src/extraction/](../src/extraction/) | CNN backbone registry + morphology pipeline |
| [src/preprocessing/](../src/preprocessing/) | FeaturePreprocessor (MICE, scaling, encoding) |
| [notebooks/](../notebooks/) | Demo, ablation, Bayesian tuning, fusion experiments |
| [datasets/](../datasets/) | Metadata CSVs (latest, grouped, debug variants) |
| [features/](../features/) | Persisted morphology cache (`morph_features_c1.npz`) |
| [runs/](../runs/) | Timestamped run artefacts + `metrics_log.csv` history |
| [main.py](../main.py) | Interactive entry point with Drive download |
| [run_training.py](../run_training.py) | CI/CD entry point (env-var configured) |

---

## 4.2 Data Layer

**Source.** Images are pulled from a Google Drive folder (one image per `(alloy × cycle1)` recipe, plus augmentations); tabular metadata lives in a single CSV — currently [datasets/metadata_latest.csv](../datasets/metadata_latest.csv) (528 rows, ~60 columns including alloy ID, image filename, chemistry, and heat-treatment parameters).

**Sample selection.** Each row corresponds to one SEM image; multiple images per physical recipe are averaged at the feature level via the row-ID convention (§4.3.5). The unique-recipe count is ~111; total rows after augmentation reach ~528.

**Targets.**

| Target | Column | Units | Notes |
|---|---|---|---|
| HoldingTemp | `Cycle1_HoldingTemp (C)` | °C | Right-skewed, ~600–900 °C dominant range |
| HoldingTime | `Cycle1_HoldingTime (min)` | minutes | Long tail; log-friendly distribution |

**Split strategy.** A nested `train_test_split` produces a 70 / 15 / 15 train / val / test partition with `random_state=42` ([run_training.py:205](../run_training.py#L205)). The same seed is reused across the entire pipeline so that any two runs with identical configuration are bit-for-bit reproducible.

**Leakage controls.**

1. **Chemistry allowlist.** Only the 14 elements in `CHEMICAL_COLUMNS` ([run_training.py:108](../run_training.py#L108)) are passed to the model — recipe-derived columns (alloy name, image filename, augmentation flag) are explicitly excluded.
2. **Recipe-column exclusion.** Heat-treatment parameters other than the two targets (e.g. cycle-2 settings, cooling rates) are dropped at load time so the model cannot learn shortcuts from correlated process variables.
3. **Train-only preprocessor fitting.** `FeaturePreprocessor.fit()` is called on the training split only; `transform()` is then applied to val/test. MICE imputation, scalers, and one-hot encoders never see held-out data during fitting.
4. **Grouped CV (in progress).** Standard k-fold leaks because multiple augmented images of the same physical sample land in different folds. [notebooks/grouped_cv_demo.ipynb](../notebooks/grouped_cv_demo.ipynb) and [datasets/metadata_grouped.csv](../datasets/metadata_grouped.csv) (with `group_key` / `group_size` columns) implement `GroupKFold` over recipe — this is what shows the realistic ~0.36 R² drop versus the leaky ~0.78 (§4.13.3).

---

## 4.3 Three-Stream Feature Architecture

**Motivation.** No single representation captures everything that matters: bulk chemistry sets the alloy's *capacity* to form phases; morphology (grain size, phase fraction, banding) is the direct *outcome* of the heat treatment we want to invert; and CNN embeddings capture *latent texture* that hand-engineered descriptors miss. Fusing all three lets each stream cover the others' blind spots.

**The three streams:**

| Stream | Source | Dim | Trainable | Role |
|---|---|---|---|---|
| **A — Tabular** | Chemistry CSV (14 elements) | ~100–160 (post-encoding) | Preprocessor stats | "What can this alloy form?" |
| **B — Morphology** | SEM image, OpenCV pipeline | 33 | None (deterministic) | "What did it actually form?" — interpretable |
| **C — Image** | SEM image, frozen ViT | 768 (DINOv2 ViT-B/14) | None | "What did it actually form?" — latent texture |

**Stream alignment.** Every stream emits an `(N, d)` matrix indexed by the metadata row ID. Multi-image rows (augmented variants) are averaged within stream before concatenation. Rows with missing images get a column-mean fill so the model never sees `NaN`. The alignment is enforced by `_align_cache_to_ids()` in [src/features.py:71](../src/features.py#L71), which strips `_F_<n>` augmentation suffixes from filenames and groups by recipe ID.

---

## 4.4 Tabular Stream — Chemistry Features

**The CHEMICAL_COLUMNS allowlist.** Fourteen elements, fixed at the module level so any addition or removal is a code change visible in code review:

```
C, Mn, Si, Cr, P, S, Mo, Cu, Ni, Al, Nb, V, Ti, Fe
```

**Preprocessing pipeline** ([src/preprocessing/pipeline.py](../src/preprocessing/pipeline.py)):

1. **Binary indicators** for `Ti, Nb, V` — these microalloying elements are *structurally* missing (often legitimately zero), so we encode "present / absent" as a separate signal rather than imputing a number.
2. **MICE imputation** (`IterativeImputer`, `max_iter=10`) for `Cr, Mo, S, Ni, Al` — these are correlated with each other and with Mn/Si, so chained regression imputation recovers more signal than median fill.
3. **Median fill** for the remaining numeric columns.
4. **StandardScaler** on all continuous features (configurable to `minmax` / `robust` / `none`).
5. **One-hot encoding** for any residual categorical columns.

**The FeaturePreprocessor class.** Implements the standard scikit-learn `fit(X_train) → transform(X)` interface so it composes cleanly with `Pipeline` and `cross_val_score`. State (imputer means, scaler stats, OHE categories) is serialised with the run artefact for inference reproducibility.

---

## 4.5 Morphological Stream — Physically Interpretable Features

**Why hand-engineered features alongside CNN embeddings.** Two reasons. First, **interpretability**: a metallurgist can read "ferrite grain area CV = 0.42, banding index = 0.18" and form a hypothesis; "DINOv2 dim 312 = 0.7" tells them nothing. Second, **small-data robustness**: 33 physically grounded features regularise better than 768 latent ones when N ≈ 100. In practice the morph stream contributes ~0.05–0.10 R² on top of image embeddings alone (see ablation plots in [runs/morph_ablation_*.png](../runs/)).

**Nine-stage extraction pipeline** ([src/extraction/morphology.py](../src/extraction/morphology.py)):

| Stage | Operation | Outputs |
|---|---|---|
| 1 | Grayscale load + scale-bar mask (bottom 12% / 20% crop) | 512×512 array |
| 2 | Phase segmentation: Otsu primary, Gaussian Mixture Model fallback | martensite/ferrite fractions, phase entropy |
| 3 | Ferrite grain geometry: connected components on ferrite mask | count, area mean/std/CV/skew/kurtosis, aspect ratio, solidity, equiv. diameter |
| 4 | Ferrite spatial distribution: nearest-neighbour distances | NND mean, NND std |
| 5 | Martensite island geometry + topology | island count, area mean, aspect ratio, connectivity, spacing |
| 6 | Grain boundary network: Canny + skeletonisation | boundary density, mean width, banding index |
| 7 | GLCM (Haralick) texture, 4 angles averaged | contrast, energy, homogeneity, correlation, dissimilarity |
| 8 | Local Binary Pattern (P=24, R=3) | LBP entropy, LBP uniformity |
| 9 | Intensity / local contrast (7×7 window) | local contrast mean/std, intensity mean/std |

**The 33 output features.** Names are fixed in `_FEATURE_NAMES` at [src/extraction/morphology.py:35](../src/extraction/morphology.py#L35) so cache files are self-describing.

| # | Feature | Stage | Metallurgical meaning |
|---|---|---|---|
| 1–3 | `morph_martensite_fraction`, `morph_ferrite_fraction`, `morph_phase_entropy` | 2 | Volume fraction of each phase; entropy ≈ phase-mixing disorder |
| 4–12 | `morph_ferrite_*` (count, area mean/std/cv/skew/kurt, aspect, solidity, equiv. diam) | 3 | Ferrite grain size distribution shape |
| 13–14 | `morph_ferrite_nnd_mean/std` | 4 | Ferrite spatial uniformity (clustered vs. dispersed) |
| 15–19 | `morph_martensite_island_*` | 5 | Martensite island morphology + connectivity |
| 20–22 | `morph_boundary_density/width`, `morph_banding_index` | 6 | Grain-boundary network density + ferrite-band alignment |
| 23–27 | `morph_glcm_*` (5 Haralick) | 7 | Texture coarseness, regularity, directional correlation |
| 28–29 | `morph_lbp_entropy`, `morph_lbp_uniformity` | 8 | Local micro-pattern complexity |
| 30–33 | `morph_local_contrast_mean/std`, `morph_intensity_mean/std` | 9 | Global tone + local contrast |

**Failure handling.** A degenerate segmentation (e.g. blank micrograph, extreme bimodality breaks Otsu) returns an all-`NaN` row of length 33. Downstream the imputer fills these with column means, so a single bad image cannot crash a run.

---

## 4.6 Image Stream — Deep CNN/ViT Embeddings

The image stream uses a **frozen pretrained backbone** as a fixed feature extractor: forward-pass only, `requires_grad=False`, no gradient flow ([src/extraction/backbones.py:35](../src/extraction/backbones.py#L35)). With ~100 training samples this avoids the catastrophic over-fitting that fine-tuning would invite, while still benefiting from the rich representations these models learnt on ImageNet / LVD-142M. A **backbone registry** ([src/extraction/backbones.py:11](../src/extraction/backbones.py#L11)) wraps each model behind a common `BaseBackbone` interface (`load()`, `freeze()`, `embed(batch) → tensor`), so swapping backbones is a one-line config change and adding a new one is a single registry entry — an extensible plugin architecture rather than an `if/elif` chain.

**The 11 supported backbones:**

| Family | Variants | Embedding dim |
|---|---|---|
| **DINOv2** (ViT, self-supervised on LVD-142M) | S/14, **B/14 (default)**, L/14 | 384, 768, 1024 |
| **ResNet** (CNN, ImageNet supervised) | 18, 50, 101 | 512, 2048, 2048 |
| **VGG** (CNN, ImageNet supervised) | 16, 19 | 512, 512 |
| **EfficientNet** (CNN, ImageNet supervised) | B0, B4 | 1280, 1792 |
| **ConvNeXt** | Tiny | 768 |
| **DenseNet** | 121 | 1024 |
| **MobileNetV3** | Large | 960 |

**Default choice — data-driven, not arbitrary.** Backbone selection was decided by the ablation in [notebooks/backbone_ablation.ipynb](../notebooks/backbone_ablation.ipynb) (artefacts: [runs/ablation_cv_r2.png](../runs/ablation_cv_r2.png), [runs/ablation_test_r2.png](../runs/ablation_test_r2.png)). DINOv2 ViT-B/14 won on test-R² *and* CV-R² across both targets while costing only a moderate 768-dim embedding — L/14 gave marginal lifts at 33% more dimensions; ResNet/VGG/EfficientNet families consistently lagged DINOv2. Every entry in [runs/metrics_log.csv](../runs/metrics_log.csv) since 2026-04-14 uses `dinov2_vitb14`.

**Parameter budget.** The frozen backbone is ~86 M parameters; the trainable regressor (a `MultiOutputRegressor(GBR)` with default depth/n_estimators) materialises ~12 K split-decision parameters across both targets — a ~7,000 : 1 frozen-to-learned ratio. This asymmetry is the *whole point* of the architecture: heavy representation learning from ImageNet/LVD-142M, light task adaptation from our 100-sample SEM dataset.

---

## 4.7 Compute Architecture — Apple Silicon Optimisations

**Device auto-detection.** The extractor walks a CUDA → MPS → CPU fallback chain at construction ([src/extraction/extractor.py:75](../src/extraction/extractor.py#L75)) and is trivially extensible to other PyTorch backends.

**Hybrid parallelism — the right primitive per workload:**

| Workload | Primitive | Why |
|---|---|---|
| CNN extraction | MPS GPU + `DataLoader(num_workers=2)` | Tensor math is batch-parallel and GPU-bound; workers overlap I/O |
| Morphology | `ProcessPoolExecutor(max_workers=8)` | NumPy/scikit-image are CPU-bound and GIL-blocked; processes side-step the GIL ([src/extraction/morphology.py:139](../src/extraction/morphology.py#L139)) |
| Multi-backbone sweep | `ThreadPoolExecutor` | I/O overlap between disk cache writes; backbones run sequentially on the GPU |

Neither approach generalises across pipelines — using `ProcessPoolExecutor` for GPU work would serialise tensors across process boundaries; using a `DataLoader` for OpenCV would block on the GIL. Picking the right primitive per stage is what produces the measured ~9× end-to-end speedup over a naive single-thread CPU baseline.

**MPS-specific notes.** `pin_memory` is enabled only for CUDA ([extractor.py:128](../src/extraction/extractor.py#L128)) — on Apple Silicon's unified memory there is no host→device copy to pin against, so pinning would add overhead with no benefit.

---

## 4.8 Caching Layer

**Why cache.** Feature extraction dominates wall-clock time: a full DINOv2 pass on ~500 images takes ~30 s on MPS; the morphology pipeline takes ~90 s with 8 worker processes. The downstream regressor trains and cross-validates in seconds. Experiments are read-heavy — we re-tune hyperparameters dozens of times against the same features — so caching extraction output is the single highest-leverage performance decision.

**Cache formats:**

| Cache | Path | Contents | Scope |
|---|---|---|---|
| Image embeddings | `data/image_cache_<backbone>.npz` | `X` (N, d) + `filenames` (N,) | One per backbone |
| Morphology | `features/morph_features_c1.npz` | `X` (N, 33) + `filenames` (N,) | Single, shared across backbones |

The morph cache is shared because morphology is independent of which CNN backbone we use; the image cache is per-backbone because embeddings are not.

**Idempotency.** `extract_cnn()` and `extract_morph()` skip work when the cache file exists *and* covers the requested filenames. The `force=True` flag rebuilds from scratch — used after preprocessing changes or when debugging suspected stale caches (this is what the `microstructure_debug` notebook does, see commit `0aa40a2`).

**Cache verification — `FeaturePipeline.verify()`.** Reports per-backbone cache existence, row count, dimensionality, and alignment to the current metadata IDs. This is the first cell of every demo notebook so a stale cache surfaces immediately rather than silently mis-aligning rows.

---

## 4.9 Fusion Strategies (in progress — [notebooks/stream_fusion.ipynb](../notebooks/stream_fusion.ipynb))

| Strategy | How | When it wins |
|---|---|---|
| **Early fusion (concat)** — *current default* | `[X_chem ‖ X_morph ‖ X_img]` → single regressor | Simple, strong baseline; works when total dim ≪ N·something |
| **PCA-compressed early fusion** | Image 768 → ~20 components, morph 33 → 10, then concat | Mitigates dimensionality dilution when image dim swamps the other streams |
| **Late fusion (stacking)** | Per-stream base learner → Ridge meta-learner | Best when streams disagree systematically; gives meta-coefficients showing per-stream contribution ([runs/stream_fusion_meta_coef.png](../runs/stream_fusion_meta_coef.png)) |

**Dimensionality dilution caveat.** With early concat, the 768-dim image stream numerically dominates the 33-dim morphology stream and ~150-dim chemistry stream, so a tree-based regressor often ignores the smaller streams entirely. PCA compression and late fusion are the two mitigations under evaluation.

---

## 4.10 Regression Layer ([notebooks/bayes_tuning.ipynb](../notebooks/bayes_tuning.ipynb), [notebooks/pipeline_benchmark.ipynb](../notebooks/pipeline_benchmark.ipynb))

**Model suite.** A deliberately wide initial sweep — Bagging (ExtraTrees, RandomForest), Stacking (Ridge meta-learner), Boosting (GBR, AdaBoost, XGBoost), and KNN — was narrowed by **quick-regression pre-screening** (5-fold CV on default hyperparameters) to the three that actually move the needle. **Final candidates: `XGB`, `KNN`, `GBR`.** The screen confirmed that linear models lack capacity for this nonlinear relationship and that AdaBoost overfits at this sample size.

**MultiOutputRegressor wrapper.** `HoldingTemp` and `HoldingTime` are predicted by **independent** regressors composed inside `MultiOutputRegressor`. This trades the chance of learning target-target correlations for two clean, separately tunable models — empirically the right call here because the two targets have very different units, scales, and distribution shapes.

**Hyperparameter defaults** (pre-tuning starting point):

| Model | Key defaults |
|---|---|
| GBR | `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8` |
| XGB | `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `reg_alpha=0.1` |
| KNN | `n_neighbors=5`, `weights='distance'`, `metric='minkowski'` |

**Bayesian hyperparameter optimisation (skopt).** `gp_minimize` over a per-model search space, with the objective being negative mean RepeatedKFold R². Search spaces are defined in [src/hyperparams.py](../src/hyperparams.py); the Bayesian loop converges in 30–50 iterations (see [notebooks/bayes_impact.ipynb](../notebooks/bayes_impact.ipynb), [notebooks/bayes_untuned_vs_tuned.png](../notebooks/bayes_untuned_vs_tuned.png)). Best parameters are persisted to [runs/hyperparams.json](../runs/hyperparams.json) and replayed on subsequent runs.

**Total learnable parameter count.** ~12 K split-decision parameters across both targets for the GBR-MultiOutput configuration — versus 86 M frozen in the backbone (§4.6).

---

## 4.11 Evaluation Architecture

**Cross-validation.** `RepeatedKFold(n_splits=5, n_repeats=10)` — 50 folds total — for stable R² estimates. With N ≈ 100, single-fold variance is high enough that a single 5-fold CV result is unreliable; repeating ten times tightens the confidence interval to a useful range.

**Held-out test set.** A 15% partition is set aside *before* CV runs and never touched until the final report — this gives an unbiased final metric and a check against accidentally tuning on the validation set.

**Per-target metrics.** R², MAE, RMSE are reported for each target individually, plus an averaged aggregate. The two targets sit on different scales (°C vs. minutes) so per-target metrics matter more than the aggregate.

**Latest results** ([runs/metrics_log.csv](../runs/metrics_log.csv)):

| Date | Notebook | N | Backbone | Best model | Test R² avg | HoldingTemp R² | HoldingTime R² | CV R² |
|---|---|---|---|---|---|---|---|---|
| 2026-04-18 | demo | 111 | dinov2_vitb14 | GBR | **0.901** | 0.84 | 0.97 | 0.51 |
| 2026-04-20 | demo | 111 | dinov2_vitb14 | ABR | 0.743 | 0.52 | 0.99 | 0.85 |
| 2026-05-02 | demo | 527 | dinov2_vitb14 | GBR | 0.514 | 0.55 | 0.52 | 0.61 |
| 2026-05-02 | debug | 444 | dinov2_vitb14 | RF | 0.522 | 0.81 | 0.35 | 0.83 |

The drop from 0.90 (N=111) to 0.51 (N=527) reflects the move from leaky standard CV to grouped CV — see §4.13.3 below and [runs/grouped_cv_results.csv](../runs/grouped_cv_results.csv).

**MetricsLogger.** Each run writes one row to [runs/metrics_log.csv](../runs/metrics_log.csv) tagged with timestamp, git commit, notebook, sample count, feature dimensions, best model, per-target metrics, and a free-text notes column. This file is the ground-truth experiment ledger.

**Run artefact convention.** Each run materialises [runs/&lt;timestamp&gt;_&lt;short-hash&gt;/](../runs/) containing `model_comparison.png`, `predictions_<target>.png`, `residuals_<target>.png`, `best_model.joblib`, and `manifest.json`. The timestamp + hash combination makes runs sortable and globally unique without coordination.

---

## 4.12 Reproducibility & Operational Architecture

**Two entry points:**

| Entry | Audience | Behaviour |
|---|---|---|
| [main.py](../main.py) | Interactive / first-time setup | CLI flags, Drive download, prompts on missing config |
| [run_training.py](../run_training.py) | CI/CD, automation | Pure env-var configuration, no prompts, exits non-zero on failure |

**Configuration.** Dataclass-based — `Config`, `PreprocessingConfig`, `MissingDataConfig`, `ScalingConfig`, `EncodingConfig`, `ExtractionConfig`, `ImageCleaningConfig` ([src/config.py](../src/config.py)). Loaded from `config.json` with environment-variable overrides for CI. Dataclasses (rather than dicts) give us schema validation and IDE autocomplete and prevent typos in config keys.

**Run artefact convention.** See §4.11 — timestamped directories, hyperparam JSON, plots, model joblib, manifest.

**Deterministic seeding.** `random_state=42` is threaded through every randomised component: `train_test_split`, all model constructors, `RepeatedKFold`, `IterativeImputer`, and the Bayesian optimiser's initial points ([src/config.py:147](../src/config.py#L147), [run_training.py:76](../run_training.py#L76), [src/model_trainer.py:24](../src/model_trainer.py#L24)).

---

## 4.13 Architectural Summary

### 4.13.1 Component inventory

| Module | Role | Trainable / Frozen / Deterministic |
|---|---|---|
| `data_loader.py` | Drive + local CSV loader | Deterministic |
| `FeaturePreprocessor` | MICE, scaling, OHE | Trainable (statistics fit on train) |
| `MorphologicalExtractor` | 9-stage CV pipeline → 33 features | Deterministic |
| `BackboneRegistry` (DINOv2 ViT-B/14) | Image embedding | **Frozen** (86 M params, `requires_grad=False`) |
| `FeaturePipeline` | Cache management + alignment | Deterministic |
| `MultiOutputRegressor(GBR / XGB / KNN)` | Two-target regression | **Trainable** (~12 K params) |
| `gp_minimize` (skopt) | Hyperparameter search | Trainable, seeded |
| `MetricsLogger` | Append run results to CSV | Deterministic |

### 4.13.2 Total parameter accounting

- **Frozen pretrained:** ~86 M (DINOv2 ViT-B/14)
- **Learned:** ~12 K (regressor split decisions, both targets combined)
- **Ratio:** ≈ 7,000 : 1 frozen-to-learned

### 4.13.3 Design decisions and tradeoffs

- **Frozen backbone over fine-tuning.** With ~100 unique recipes, fine-tuning would over-fit; frozen embeddings + a small regressor under-fits more gracefully and is recoverable. *Cost*: we cannot specialise the visual representation to SEM-specific texture. *Benefit*: the regressor trains in seconds, enabling Bayesian search over hundreds of configurations.
- **Three streams over one.** Adds engineering overhead (three caches, three alignments) but removes single-stream blind spots. Morphology gives interpretability the CNN cannot; chemistry encodes information the image physically cannot contain.
- **Early concat fusion as default.** Simplest baseline; loses some morph/chem signal to dimensionality dilution. Late-fusion stacking is the planned next step (§4.9).
- **MultiOutputRegressor over a joint model.** Sacrifices the chance to learn target-target correlations in exchange for clean per-target diagnostics and independent tuning. The two targets have different units and very different difficulty (Time was easier than Temp in early runs); independent models let us tune each appropriately.
- **Standard k-fold → grouped CV.** The single most impactful diagnostic of the project. With augmentation, standard k-fold leaks because copies of the same physical sample land in different folds — inflating R² from a realistic ~0.36 to a leaky ~0.78 ([runs/grouped_cv_results.csv](../runs/grouped_cv_results.csv)). Grouped CV by recipe (`metadata_grouped.csv`) gives the honest number; this is the baseline future work has to beat.
- **Caching as a first-class concern.** Roughly 80% of project iteration time would be re-extraction without caches. The per-backbone `.npz` convention also makes the image-feature space inspectable from a notebook in two lines (`np.load(...)`).
- **Apple Silicon as primary target.** The MPS → CPU fallback was a deliberate choice to keep the project laptop-runnable end-to-end; the architecture extends cleanly to CUDA but does not require it.
