# Morphological Feature Extraction Design

## Overview

The CNN backbone (VGG/ResNet) extracts a global texture representation of each
SEM image but collapses spatial structure into a single vector. It cannot answer
physically meaningful questions: *What fraction of this image is martensite? How
coarse are the ferrite grains? Is the martensite phase forming a percolating
network or isolated islands?* These are precisely the quantities that change
systematically with heat treatment parameters (intercritical annealing
temperature, holding time, quench rate).

Morphological feature extraction runs as a parallel stage before the CNN,
producing a small (~32 feature) interpretable vector per image grounded in
quantitative metallography. This supplements the CNN's learnt representation
with explicitly computed microstructural descriptors that have been validated
against processing parameter prediction in the DP steel ML literature.

---

## Input images

All morphological extraction runs on the **AI-cleaned augmented images**
(`augumented_data` column / folder). These are cleaned versions of the raw SEM
images with annotation overlays (panel letters, yellow F/M phase labels)
removed. The scale bar region (bottom-right corner) may still be present and is
masked before any analysis.

**Why not raw images:** The raw images contain yellow text annotations (F, M
phase labels), panel letters (b, c, d, h), and scale bar graphics. These
introduce bright non-phase pixels that corrupt both phase segmentation (white
text is classified as martensite) and grain geometry measurements (scale bar
rectangle distorts boundary statistics).

---

## Pipeline position

```
augmented image paths
        │
        ├──► MorphologicalExtractor.extract(image_paths)
        │         → X_morph  shape (N, 32)  float64
        │           NaN rows for failed images
        │
        └──► FeatureExtractor.extract_features(image_paths)    (existing)
                  → X_images  shape (N, backbone_dim)

X_morph → FeaturePreprocessor (median imputation + standard scaling)
        → X_morph_scaled  shape (N, 32)

Final:  [X_images | X_morph_scaled | X_tabular_chem_process]
```

`X_morph` joins the **tabular stream**, not the image stream. It goes through
`FeaturePreprocessor` with median imputation (same Group 3 path as composition
columns) and standard scaling. It is not concatenated raw with the CNN features.

---

## New files

```
src/extraction/
    morphology.py          — MorphologicalExtractor, per-image computation stages
    morphology_config.py   — MorphologyConfig dataclass
```

`FeatureExtractor` and all existing backbone code are unchanged.

---

## `MorphologyConfig`

```python
@dataclass
class MorphologyConfig:
    img_size: int = 512              # resize longest axis before analysis
    scale_bar_mask: tuple = (0.88, 0.80)  # mask rows > 88%, cols > 80%
    otsu_fallback_gmm: bool = True   # GMM fallback if fraction out of [0.05, 0.90]
    min_grain_area: int = 30         # px² — discard sub-pixel noise regions
    boundary_sigma: float = 1.5      # Gaussian sigma for Canny pre-smoothing
    canny_low: float = 0.05          # Canny low threshold (fraction of dtype range)
    canny_high: float = 0.15         # Canny high threshold
    local_patch_size: int = 7        # window size for local contrast features
    lbp_radius: int = 3              # LBP neighbourhood radius
    lbp_n_points: int = 24           # LBP sampling points (8 * radius)
    glcm_distances: tuple = (1, 3)   # pixel distances for GLCM computation
    glcm_angles: tuple = (0, 45, 90, 135)  # angles in degrees, averaged
    cache_path: str = "features/morph_features.npz"
```

---

## `MorphologicalExtractor` — public API

```python
class MorphologicalExtractor:
    def __init__(self, config: MorphologyConfig): ...

    def extract(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Returns (N, n_features) array and filenames.
           NaN rows for failed images."""

    def extract_single(self, image_path: str) -> np.ndarray:
        """Returns (n_features,) array. NaN-filled on any failure."""

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of 'morph_*' feature name strings."""
```

---

## Per-image computation stages

### Stage 1 — Load and prepare

- Load as **grayscale** (SEM images are inherently single-channel; RGB is
  redundant and triples memory)
- Resize to `img_size` on the longest axis, pad to square with the image's
  median pixel value to avoid introducing bright/dark border artefacts
- Apply scale bar mask: zero out the bottom-right corner defined by the
  `scale_bar_mask` fraction tuple before any downstream computation

### Stage 2 — Phase segmentation

- Compute Otsu global threshold on the masked grayscale array
- Binary map: `martensite = img > threshold`
  (martensite is brighter in SEM back-scattered / secondary electron contrast)
- **Validation**: if `martensite_fraction` ∉ [0.05, 0.90] and
  `otsu_fallback_gmm=True`, fit a 2-component Gaussian Mixture Model on the
  intensity histogram and use the decision boundary between the two Gaussians
- If fraction is still outside range after GMM, mark the image as failed →
  return NaN row

**Why Otsu with GMM fallback:** Otsu assumes a bimodal histogram. For images
where one phase strongly dominates (e.g. >85% martensite after a high-temperature
cycle), the histogram is unimodal and Otsu places the threshold erroneously.
The 2-component GMM fits the underlying distribution regardless of peak symmetry,
at the cost of slightly more computation. Otsu runs first because it is cheaper
and correct for the majority of images.

### Stage 3 — Ferrite grain geometry

- `label_map = skimage.measure.label(~martensite_binary)` — label ferrite regions
- `props = skimage.measure.regionprops(label_map)`
- Filter out regions smaller than `min_grain_area` pixels
- Per-region measurements: area, equivalent circular diameter, aspect ratio
  (minor\_axis\_length / major\_axis\_length), solidity (area / convex hull area)
- **Aggregated statistics**: mean, std, coefficient of variation (std/mean),
  skewness, kurtosis across all ferrite grains; plus grain count

### Stage 4 — Ferrite spatial distribution

- Compute centroids of all ferrite regions from regionprops
- For each grain, find the nearest-neighbour grain centroid using a KD-tree
- Aggregate: `nnd_mean`, `nnd_std` — captures whether grains are uniformly
  distributed or clustered

### Stage 5 — Martensite island geometry and topology

- Label `martensite_binary` with `skimage.measure.label`
- Per-island: area, aspect ratio (same definition as ferrite)
- **Connectivity**: fraction of martensite pixels belonging to the single largest
  connected component. Approaches 1.0 as the martensite network percolates
  (associated with higher intercritical annealing temperatures)
- **Island spacing**: mean nearest-neighbour distance between martensite island
  centroids — low spacing = finely dispersed hard phase

### Stage 6 — Grain boundary network

- `edges = skimage.feature.canny(img_float, sigma=boundary_sigma, ...)`
- `skeleton = skimage.morphology.skeletonize(edges)` — 1-pixel-wide boundary
- `boundary_density = skeleton.sum() / (img_h * img_w)` — scale-invariant
- **Banding index**: ratio of horizontal edge density to vertical edge density
  (from two directional Canny passes at 0° and 90°). >1.0 = banded martensite
  morphology from rolling texture; ~1.0 = equiaxed / random distribution
- Mean boundary width: (dilated edge pixels − skeleton pixels) / skeleton pixels

### Stage 7 — GLCM texture features

- Compute Gray Level Co-occurrence Matrix at all `glcm_distances` ×
  `glcm_angles` combinations using `skimage.feature.graycomatrix`
- Extract five Haralick-derived properties via `skimage.feature.graycoprops`:
  contrast, energy, homogeneity, correlation, dissimilarity
- Average across angles for rotation invariance; keep distance-indexed values
  separate (different scales capture different structural levels)
- Applied to the **full masked image** (not per-region) — robust at N=88

### Stage 8 — LBP texture features

- Compute Local Binary Pattern histogram with `skimage.feature.local_binary_pattern`
  using `lbp_radius` and `lbp_n_points`
- From the normalised histogram extract two summary statistics:
  - `lbp_entropy`: `scipy.stats.entropy(hist)` — high for complex/irregular
    local texture (lath martensite); low for smooth homogeneous regions
  - `lbp_uniformity`: fraction of codes with ≤2 bitwise 0→1 transitions —
    high in smooth equiaxed ferrite, low in lath/distorted martensite
- Full LBP histogram (256+ bins) is intentionally not used — far too many
  features for N=88

### Stage 9 — Intensity and local contrast

- `intensity_mean`, `intensity_std` on the masked grayscale — proxy for
  average phase composition and heterogeneity
- Slide a `local_patch_size × local_patch_size` window across the image;
  compute standard deviation per patch:
  - `local_contrast_mean` — mean patch std; tracks texture roughness
  - `local_contrast_std` — variation in texture roughness across image

---

## Failure handling

Any exception in `extract_single` — corrupt file, degenerate segmentation (all
one phase), zero grains found after area filtering, division by zero in any
ratio — is caught, logged with the filename, and returns
`np.full(n_features, np.nan)`. The NaN row is handled by `FeaturePreprocessor`'s
median imputer downstream. No special-case logic is needed in the training loop.

---

## Complete feature vector (32 features)

| Name | Stage | Description |
|------|-------|-------------|
| `morph_martensite_fraction` | 2 | Fraction of unmasked pixels classified as martensite |
| `morph_ferrite_fraction` | 2 | 1 − martensite_fraction (explicit for feature importance) |
| `morph_phase_entropy` | 2 | Entropy of binary phase map |
| `morph_ferrite_grain_count` | 3 | Number of ferrite regions > min_grain_area |
| `morph_ferrite_area_mean` | 3 | Mean ferrite grain area (px²) |
| `morph_ferrite_area_std` | 3 | Std of ferrite grain areas |
| `morph_ferrite_area_cv` | 3 | Coefficient of variation of grain areas |
| `morph_ferrite_area_skewness` | 3 | Skewness — right-skewed = few large grains dominating |
| `morph_ferrite_area_kurtosis` | 3 | Kurtosis — high = bimodal grain size distribution |
| `morph_ferrite_aspect_ratio_mean` | 3 | Mean minor/major axis ratio; <1 = elongated/lath |
| `morph_ferrite_solidity_mean` | 3 | Mean convex hull fill; low = irregular/dendritic |
| `morph_ferrite_equiv_diam_mean` | 3 | Mean equivalent circular diameter |
| `morph_ferrite_nnd_mean` | 4 | Mean nearest-neighbour distance between grain centroids |
| `morph_ferrite_nnd_std` | 4 | Std of nearest-neighbour distances |
| `morph_martensite_island_count` | 5 | Number of discrete martensite regions |
| `morph_martensite_island_area_mean` | 5 | Mean martensite island area (px²) |
| `morph_martensite_island_aspect_ratio_mean` | 5 | Mean aspect ratio; low = lath morphology |
| `morph_martensite_connectivity` | 5 | Fraction of M pixels in largest connected component |
| `morph_martensite_island_spacing_mean` | 5 | Mean nearest-neighbour distance between M centroids |
| `morph_boundary_density` | 6 | Skeleton edge pixels / total image pixels |
| `morph_boundary_mean_width` | 6 | Mean width of boundary regions (px) |
| `morph_banding_index` | 6 | Horizontal / vertical edge density ratio |
| `morph_glcm_contrast` | 7 | GLCM contrast — sharpness of phase boundaries |
| `morph_glcm_energy` | 7 | GLCM energy — texture uniformity |
| `morph_glcm_homogeneity` | 7 | GLCM homogeneity — closeness to diagonal |
| `morph_glcm_correlation` | 7 | GLCM correlation — spatial intensity dependency |
| `morph_glcm_dissimilarity` | 7 | GLCM dissimilarity — texture irregularity |
| `morph_lbp_entropy` | 8 | LBP histogram entropy — complex texture = high |
| `morph_lbp_uniformity` | 8 | Fraction of uniform LBP codes |
| `morph_local_contrast_mean` | 9 | Mean local patch std — texture roughness |
| `morph_local_contrast_std` | 9 | Variation of texture roughness across image |
| `morph_intensity_mean` | 9 | Mean grayscale intensity |
| `morph_intensity_std` | 9 | Std of grayscale intensity |

`morph_ferrite_fraction` and `morph_martensite_fraction` are exactly collinear.
Both are retained for interpretability in feature importance plots; regularised
models (Random Forest, XGBoost) are not materially harmed by one redundant
binary complement.

---

## Features intentionally excluded

| Feature | Reason excluded |
|---------|----------------|
| Full LBP histogram (256+ bins) | ~3× the feature count of the entire vector at N=88; summarised to entropy + uniformity |
| Per-phase GLCM (separate for F and M) | Requires per-region pixel crop; marginal gain over whole-image GLCM at small N |
| Topological genus / handles | Requires 3D reconstruction or EBSD — not available from 2D SEM |
| Kernel Average Misorientation (KAM) | EBSD-only |
| `boundary_length_total` (absolute) | Redundant with `boundary_density`; absolute count scales with image size |

---

## Dependencies

```
scikit-image >= 0.21    # measure, feature.canny, morphology.skeletonize,
                        # feature.graycomatrix, feature.local_binary_pattern,
                        # filters.threshold_otsu
scipy >= 1.11           # stats.entropy, spatial.KDTree, mixture.GaussianMixture
```

Both are part of the standard scientific Python stack. No new heavy dependencies.

---

## Caching

`extract()` checks for `cache_path` before running. On cache hit, loads and
returns the stored `(X_morph, filenames)` array directly. Cache is stored as
`.npz` (same pattern as the CNN image feature cache). The `--no-cache` flag in
`run_training.py` and `main.py` clears both the CNN cache and the morphology
cache.

---

## Literature basis

The feature groups and their relevance to DP steel processing parameter
prediction are grounded in the following references:

### Phase fraction and grain geometry

- **Hybrid Data-Driven Deep Learning Framework for Material Mechanical Properties
  Prediction with the Focus on Dual-Phase Steel Microstructures** — establishes
  martensite volume fraction and ferrite grain size as primary microstructural
  descriptors for mechanical property prediction in DP steels.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9822330/

- **The Prediction of the Mechanical Properties for Dual-Phase High Strength
  Steel Grades Based on Microstructure Characteristics** — quantifies phase
  fraction, grain size distribution, and shape descriptors (aspect ratio,
  circularity, solidity) as the core feature set for property prediction.
  https://www.mdpi.com/2075-4901/8/4/242

- **The effects of heat-treatment parameters on the mechanical properties and
  microstructures of a low-carbon dual-phase steel** — demonstrates systematic
  changes in ferrite grain size and martensite fraction with intercritical
  annealing temperature and cooling rate.
  https://www.sciencedirect.com/science/article/abs/pii/S092150932301225X

- **Microstructure Distribution Parameters for Ferrite-Martensite Dual-Phase
  Steel** — defines spatial distribution parameters including nearest-neighbour
  distance and phase spacing for F-M DP steels specifically.
  https://www.researchgate.net/publication/351127245_Microstructure_Distribution_Parameters_for_Ferrite-Martensite_Dual-Phase_Steel

### Martensite connectivity and topology

- **Topological Analysis of Martensite Morphology in Dual-Phase Steels** —
  introduces topological descriptors for martensite including connectivity,
  genus, and independent body count. Connectivity is identified as the most
  accessible 2D measure correlating with mechanical behaviour.
  https://www.scientific.net/AMR.409.725

- **The role of connectivity of martensite on the tensile properties of a low
  alloy steel** — shows that martensite interconnectivity (fraction of the hard
  phase forming a continuous network) is a stronger predictor of strength and
  ductility than volume fraction alone.
  https://www.sciencedirect.com/science/article/abs/pii/S0261306906001610

- **Inverse design of dual-phase steel microstructures using generative machine
  learning model and Bayesian optimization** — includes martensite island spacing
  and connectivity as inputs to an inverse design framework, confirming their
  sensitivity to processing conditions.
  https://www.sciencedirect.com/science/article/abs/pii/S0749641923002607

### GLCM texture features

- **A methodology of steel microstructure recognition using SEM images by machine
  learning based on textural analysis** — benchmarks GLCM features (contrast,
  correlation, energy, homogeneity, dissimilarity, ASM) across five steel
  microstructure classes. Contrast and dissimilarity discriminate most strongly
  between irregular bainitic/martensitic and smooth ferritic regions.
  https://www.sciencedirect.com/science/article/abs/pii/S2352492820325253

- **Advanced Steel Microstructural Classification by Deep Learning Methods** —
  uses GLCM alongside CNN features; GLCM features provide the strongest
  variation signal between steel classes in the classical feature set.
  https://www.nature.com/articles/s41598-018-20037-5

- **Overview: Machine Learning for Segmentation and Classification of Complex
  Steel Microstructures** — surveys GLCM, LBP, and Gabor texture methods across
  recent steel microstructure ML literature; identifies GLCM as the most widely
  validated texture descriptor for SEM images.
  https://www.mdpi.com/2075-4701/14/5/553

### LBP texture features

- **Hybrid machine learning and regression framework for automated phase
  classification and quantification in SEM images of commercial steels** —
  combines LBP histogram features with morphological descriptors; LBP entropy
  and uniformity are identified as effective summary statistics for distinguishing
  ferrite-pearlite from martensite-austenite microstructure types.
  https://link.springer.com/article/10.1007/s43939-025-00323-6

### Banding and spatial anisotropy

- **Identification of Martensite Bands in Dual-Phase Steels: A Deep Learning
  Object Detection Approach** — characterises banding in DP steels as a
  distinct spatial morphology arising from rolling; shows banded vs. equiaxed
  martensite distributions produce different property outcomes for the same
  volume fraction.
  https://onlinelibrary.wiley.com/doi/10.1002/srin.202200836

### Segmentation methodology

- **Segmentation of dual phase steel micrograph: An automated approach** —
  validates Otsu thresholding for ferrite/martensite segmentation in SEM images
  of DP steel; discusses failure modes for strongly asymmetric phase fractions
  and recommends fallback strategies.
  https://www.sciencedirect.com/science/article/abs/pii/S0263224113001681

- **Generic dual-phase classification models through deep learning semantic
  segmentation method and image gray-level optimization** — benchmarks automatic
  segmentation methods on DP steel SEM; gray-level optimisation (equivalent to
  adaptive thresholding / GMM) outperforms fixed Otsu for minority-phase-dominant
  images.
  https://www.sciencedirect.com/science/article/abs/pii/S1359646223006693

### Composition–microstructure–property linkage

- **Building a quantitative composition-microstructure-property relationship of
  dual-phase steels via multimodal data mining** — combines chemical composition,
  processing parameters, and morphological descriptors in a unified ML framework;
  morphological features provide complementary signal to composition features
  even when composition is already included as input.
  https://www.sciencedirect.com/science/article/abs/pii/S1359645423002859

- **Prediction of mechanical properties and microstructure of dual-phase steel
  based on deep learning method** — demonstrates that explicit morphological
  features (grain size, phase fraction, aspect ratio) retain predictive value
  alongside deep CNN features on small DP steel datasets.
  https://link.springer.com/article/10.1007/s10853-025-11185-x

- **Enhancing Prediction Accuracy of Machine Learning Models for Materials
  Informatics Problems in Alloy Design: A Case Study on Dual-Phase Steel** —
  confirms that feature-engineered morphological descriptors combined with
  composition features outperform composition-only models for processing
  parameter prediction in DP steel.
  https://link.springer.com/article/10.1007/s11665-025-12401-0
