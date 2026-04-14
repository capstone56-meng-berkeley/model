"""
Morphological feature extraction from SEM microstructure images.

Extracts 32 physically interpretable features per image covering:
  - Phase fractions (martensite / ferrite)
  - Ferrite grain geometry and spatial distribution
  - Martensite island geometry and network topology
  - Grain boundary network properties
  - GLCM texture (5 Haralick properties)
  - LBP texture summary statistics
  - Intensity and local contrast statistics

All features are prefixed ``morph_``.  Any per-image failure returns a
NaN-filled vector so downstream imputation handles it transparently.

See docs/morphological_feature_extraction.md for full design rationale
and literature references.
"""

import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np

from .morphology_config import MorphologyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name definition (order must match _compute_features return order)
# ---------------------------------------------------------------------------

_FEATURE_NAMES: List[str] = [
    # Phase fractions (stage 2)
    "morph_martensite_fraction",
    "morph_ferrite_fraction",
    "morph_phase_entropy",
    # Ferrite grain geometry (stage 3)
    "morph_ferrite_grain_count",
    "morph_ferrite_area_mean",
    "morph_ferrite_area_std",
    "morph_ferrite_area_cv",
    "morph_ferrite_area_skewness",
    "morph_ferrite_area_kurtosis",
    "morph_ferrite_aspect_ratio_mean",
    "morph_ferrite_solidity_mean",
    "morph_ferrite_equiv_diam_mean",
    # Ferrite spatial distribution (stage 4)
    "morph_ferrite_nnd_mean",
    "morph_ferrite_nnd_std",
    # Martensite island geometry + topology (stage 5)
    "morph_martensite_island_count",
    "morph_martensite_island_area_mean",
    "morph_martensite_island_aspect_ratio_mean",
    "morph_martensite_connectivity",
    "morph_martensite_island_spacing_mean",
    # Grain boundary network (stage 6)
    "morph_boundary_density",
    "morph_boundary_mean_width",
    "morph_banding_index",
    # GLCM texture (stage 7)
    "morph_glcm_contrast",
    "morph_glcm_energy",
    "morph_glcm_homogeneity",
    "morph_glcm_correlation",
    "morph_glcm_dissimilarity",
    # LBP texture (stage 8)
    "morph_lbp_entropy",
    "morph_lbp_uniformity",
    # Intensity / local contrast (stage 9)
    "morph_local_contrast_mean",
    "morph_local_contrast_std",
    "morph_intensity_mean",
    "morph_intensity_std",
]

_N_FEATURES = len(_FEATURE_NAMES)  # 33


# ---------------------------------------------------------------------------
# Public extractor class
# ---------------------------------------------------------------------------

class MorphologicalExtractor:
    """
    Extract morphological features from SEM microstructure images.

    Usage::

        config = MorphologyConfig()
        extractor = MorphologicalExtractor(config)
        X_morph, filenames = extractor.extract(image_paths)
        # X_morph: (N, 33) float64, NaN rows for failed images

    The extractor is stateless — no fitting required.  Results are cached
    to ``config.cache_path`` as an .npz file on first run.
    """

    def __init__(self, config: Optional[MorphologyConfig] = None):
        self.config = config or MorphologyConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image_paths: List[str],
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract morphological features for a list of images.

        Args:
            image_paths: Paths to image files (augmented / cleaned images).
            use_cache:   If True, load from / save to ``config.cache_path``.

        Returns:
            Tuple of:
              - X: float64 array of shape (N, n_features).  NaN rows indicate
                   failed images.
              - filenames: list of basenames in the same order as X.
        """
        cache = self.config.cache_path
        filenames = [os.path.basename(p) for p in image_paths]

        if use_cache and cache and os.path.exists(cache):
            logger.info("Loading morphological features from cache: %s", cache)
            data = np.load(cache, allow_pickle=True)
            return data["X"].astype(np.float64), list(data["filenames"])

        logger.info("Extracting morphological features from %d images...", len(image_paths))
        rows = []
        for i, path in enumerate(image_paths):
            feat = self.extract_single(path)
            rows.append(feat)
            if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                logger.info("  %d / %d", i + 1, len(image_paths))

        X = np.vstack(rows)

        if use_cache and cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.savez(cache, X=X, filenames=np.array(filenames))
            logger.info("Cached morphological features to %s", cache)

        return X, filenames

    def extract_single(self, image_path: str) -> np.ndarray:
        """
        Extract morphological features for one image.

        Returns a (n_features,) float64 array.  Returns NaN-filled vector
        on any failure (corrupt file, degenerate segmentation, etc.).
        """
        try:
            return self._compute_features(image_path)
        except Exception as exc:
            logger.warning("Morphology extraction failed for %s: %s", image_path, exc)
            return np.full(_N_FEATURES, np.nan, dtype=np.float64)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return ordered list of feature names."""
        return list(_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _compute_features(self, image_path: str) -> np.ndarray:
        """Run all 9 stages and return the (n_features,) vector."""
        cfg = self.config

        # Stage 1 — load and prepare
        img = _load_grayscale(image_path, cfg.img_size, cfg.scale_bar_mask)

        # Stage 2 — phase segmentation
        martensite_mask, m_fraction = _segment_phases(
            img,
            cfg.otsu_fallback_gmm,
            cfg.fraction_min,
            cfg.fraction_max,
        )
        f_fraction = 1.0 - m_fraction
        phase_entropy = _binary_entropy(m_fraction)

        # Stage 3 — ferrite grain geometry
        ferrite_geo = _grain_geometry(~martensite_mask, cfg.min_grain_area)

        # Stage 4 — ferrite spatial distribution
        ferrite_nnd_mean, ferrite_nnd_std = _nearest_neighbour_distances(
            ferrite_geo["centroids"]
        )

        # Stage 5 — martensite island geometry + topology
        mart_geo = _grain_geometry(martensite_mask, cfg.min_grain_area)
        m_connectivity = _phase_connectivity(martensite_mask, cfg.min_grain_area)
        m_spacing_mean = _nearest_neighbour_distances(mart_geo["centroids"])[0]

        # Stage 6 — grain boundary network
        boundary_density, boundary_mean_width, banding_index = _boundary_features(
            img,
            cfg.boundary_sigma,
            cfg.canny_low,
            cfg.canny_high,
        )

        # Stage 7 — GLCM texture
        glcm_feats = _glcm_features(
            img,
            cfg.glcm_distances,
            cfg.glcm_angles_deg,
            cfg.glcm_levels,
        )

        # Stage 8 — LBP texture
        lbp_entropy, lbp_uniformity = _lbp_features(
            img, cfg.lbp_radius, cfg.lbp_n_points
        )

        # Stage 9 — intensity / local contrast
        lc_mean, lc_std = _local_contrast(img, cfg.local_patch_size)
        i_mean = float(np.mean(img))
        i_std = float(np.std(img))

        return np.array([
            m_fraction,
            f_fraction,
            phase_entropy,
            ferrite_geo["count"],
            ferrite_geo["area_mean"],
            ferrite_geo["area_std"],
            ferrite_geo["area_cv"],
            ferrite_geo["area_skewness"],
            ferrite_geo["area_kurtosis"],
            ferrite_geo["aspect_ratio_mean"],
            ferrite_geo["solidity_mean"],
            ferrite_geo["equiv_diam_mean"],
            ferrite_nnd_mean,
            ferrite_nnd_std,
            mart_geo["count"],
            mart_geo["area_mean"],
            mart_geo["aspect_ratio_mean"],
            m_connectivity,
            m_spacing_mean,
            boundary_density,
            boundary_mean_width,
            banding_index,
            glcm_feats["contrast"],
            glcm_feats["energy"],
            glcm_feats["homogeneity"],
            glcm_feats["correlation"],
            glcm_feats["dissimilarity"],
            lbp_entropy,
            lbp_uniformity,
            lc_mean,
            lc_std,
            i_mean,
            i_std,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _load_grayscale(
    path: str,
    img_size: int,
    scale_bar_mask: Tuple[float, float],
) -> np.ndarray:
    """
    Load image as float64 grayscale in [0, 1], resize, and mask scale bar.

    Resize preserves aspect ratio (longest axis → img_size) and pads to
    square with the image median value.
    """
    from PIL import Image as PILImage

    with PILImage.open(path) as pil_img:
        grey = pil_img.convert("L")

    w, h = grey.size
    scale = img_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    grey = grey.resize((new_w, new_h), PILImage.LANCZOS)
    arr = np.asarray(grey, dtype=np.float64) / 255.0

    # Pad to square with median value
    if arr.shape != (img_size, img_size):
        pad_val = float(np.median(arr))
        canvas = np.full((img_size, img_size), pad_val, dtype=np.float64)
        canvas[:new_h, :new_w] = arr
        arr = canvas

    # Apply scale-bar mask
    row_frac, col_frac = scale_bar_mask
    if row_frac < 1.0 or col_frac < 1.0:
        row_cut = int(math.floor(row_frac * img_size))
        col_cut = int(math.floor(col_frac * img_size))
        arr[row_cut:, col_cut:] = 0.0

    return arr


def _segment_phases(
    img: np.ndarray,
    otsu_fallback_gmm: bool,
    fraction_min: float,
    fraction_max: float,
) -> Tuple[np.ndarray, float]:
    """
    Segment martensite (bright) from ferrite (dark) using Otsu thresholding.

    Falls back to a 2-component GMM when the Otsu fraction falls outside
    [fraction_min, fraction_max].  Raises ValueError if segmentation is still
    degenerate after fallback.

    Returns:
        martensite_mask: bool array, True = martensite pixel
        m_fraction:      fraction of unmasked pixels classified as martensite
    """
    from skimage.filters import threshold_otsu

    # Only operate on non-zero pixels (zero = masked scale-bar region)
    valid_mask = img > 0.0
    valid_pixels = img[valid_mask]

    if valid_pixels.size == 0:
        raise ValueError("Image has no valid (unmasked) pixels.")

    thresh = threshold_otsu(valid_pixels)
    martensite_mask = img > thresh
    # Exclude masked area from fraction calculation
    m_fraction = float(martensite_mask[valid_mask].sum()) / float(valid_mask.sum())

    if otsu_fallback_gmm and not (fraction_min <= m_fraction <= fraction_max):
        logger.debug(
            "Otsu fraction %.3f outside [%.2f, %.2f]; trying GMM fallback.",
            m_fraction, fraction_min, fraction_max,
        )
        thresh_gmm = _gmm_threshold(valid_pixels)
        martensite_mask = img > thresh_gmm
        m_fraction = float(martensite_mask[valid_mask].sum()) / float(valid_mask.sum())

    if not (fraction_min <= m_fraction <= fraction_max):
        raise ValueError(
            f"Martensite fraction {m_fraction:.3f} still outside "
            f"[{fraction_min}, {fraction_max}] after GMM fallback."
        )

    return martensite_mask, m_fraction


def _gmm_threshold(pixels: np.ndarray) -> float:
    """Fit a 2-component GMM and return the decision boundary."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(pixels.reshape(-1, 1))

    means = gmm.means_.flatten()
    low_idx, high_idx = np.argsort(means)
    mu_low, mu_high = means[low_idx], means[high_idx]
    # Decision boundary: midpoint weighted by inverse std
    std_low = math.sqrt(gmm.covariances_.flatten()[low_idx])
    std_high = math.sqrt(gmm.covariances_.flatten()[high_idx])
    # Weighted midpoint between the two means
    thresh = (mu_low / std_low + mu_high / std_high) / (1.0 / std_low + 1.0 / std_high)
    return float(np.clip(thresh, 0.0, 1.0))


def _binary_entropy(p: float) -> float:
    """Shannon entropy of a binary distribution with probability p."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return float(-(p * math.log2(p) + q * math.log2(q)))


def _grain_geometry(binary_mask: np.ndarray, min_area: int) -> dict:
    """
    Compute per-region geometry statistics from a binary mask.

    Returns a dict with keys:
        count, area_mean, area_std, area_cv, area_skewness, area_kurtosis,
        aspect_ratio_mean, solidity_mean, equiv_diam_mean, centroids.
    """
    from skimage.measure import label, regionprops

    lbl = label(binary_mask)
    props = regionprops(lbl)
    props = [p for p in props if p.area >= min_area]

    if len(props) == 0:
        return {
            "count": 0.0,
            "area_mean": np.nan,
            "area_std": np.nan,
            "area_cv": np.nan,
            "area_skewness": np.nan,
            "area_kurtosis": np.nan,
            "aspect_ratio_mean": np.nan,
            "solidity_mean": np.nan,
            "equiv_diam_mean": np.nan,
            "centroids": np.empty((0, 2)),
        }

    areas = np.array([p.area for p in props], dtype=np.float64)
    aspect_ratios = np.array(
        [
            p.axis_minor_length / p.axis_major_length
            if p.axis_major_length > 0
            else 1.0
            for p in props
        ],
        dtype=np.float64,
    )
    solidities = np.array([p.solidity for p in props], dtype=np.float64)
    equiv_diams = np.array([p.equivalent_diameter_area for p in props], dtype=np.float64)
    centroids = np.array([p.centroid for p in props], dtype=np.float64)  # (N, 2)

    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas))

    return {
        "count": float(len(props)),
        "area_mean": area_mean,
        "area_std": area_std,
        "area_cv": area_std / area_mean if area_mean > 0 else np.nan,
        "area_skewness": float(_skewness(areas)),
        "area_kurtosis": float(_kurtosis(areas)),
        "aspect_ratio_mean": float(np.mean(aspect_ratios)),
        "solidity_mean": float(np.mean(solidities)),
        "equiv_diam_mean": float(np.mean(equiv_diams)),
        "centroids": centroids,
    }


def _skewness(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)  # excess kurtosis


def _nearest_neighbour_distances(centroids: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and std of nearest-neighbour distances between centroids.

    Returns (np.nan, np.nan) if fewer than 2 centroids.
    """
    if centroids.shape[0] < 2:
        return np.nan, np.nan

    from scipy.spatial import KDTree

    tree = KDTree(centroids)
    dists, _ = tree.query(centroids, k=2)  # k=2: self + nearest neighbour
    nn_dists = dists[:, 1]  # exclude self (distance = 0)
    return float(np.mean(nn_dists)), float(np.std(nn_dists))


def _phase_connectivity(binary_mask: np.ndarray, min_area: int) -> float:
    """
    Fraction of phase pixels belonging to the single largest connected component.

    Returns 0.0 if no regions exist; 1.0 if all pixels are in one component.
    """
    from skimage.measure import label, regionprops

    lbl = label(binary_mask)
    props = regionprops(lbl)
    props = [p for p in props if p.area >= min_area]

    if len(props) == 0:
        return 0.0

    total = sum(p.area for p in props)
    largest = max(p.area for p in props)
    return float(largest / total) if total > 0 else 0.0


def _boundary_features(
    img: np.ndarray,
    sigma: float,
    low: float,
    high: float,
) -> Tuple[float, float, float]:
    """
    Compute boundary density, mean boundary width, and banding index.

    Returns:
        boundary_density:   skeleton pixels / total pixels
        boundary_mean_width: mean width of pre-skeleton edge regions (pixels)
        banding_index:      horizontal edge density / vertical edge density
    """
    from skimage.feature import canny
    from skimage.morphology import skeletonize
    import numpy as np

    # Full-image edges
    edges = canny(img, sigma=sigma, low_threshold=low, high_threshold=high)
    skeleton = skeletonize(edges)

    n_pixels = img.size
    skel_sum = int(skeleton.sum())
    boundary_density = skel_sum / n_pixels if n_pixels > 0 else 0.0

    # Mean boundary width: (edge - skeleton) / skeleton
    edge_sum = int(edges.sum())
    if skel_sum > 0:
        boundary_mean_width = float((edge_sum - skel_sum) / skel_sum)
    else:
        boundary_mean_width = 0.0

    # Banding index — directional Canny
    # Horizontal edges: gradient strongest in vertical direction (sobel_h)
    # Vertical edges: gradient strongest in horizontal direction (sobel_v)
    from skimage.filters import sobel_h, sobel_v
    from skimage.filters import gaussian

    smoothed = gaussian(img, sigma=sigma)
    horiz_mag = np.abs(sobel_h(smoothed))  # responds to horizontal boundaries
    vert_mag = np.abs(sobel_v(smoothed))   # responds to vertical boundaries

    horiz_density = float(np.mean(horiz_mag))
    vert_density = float(np.mean(vert_mag))

    if vert_density > 1e-8:
        banding_index = horiz_density / vert_density
    else:
        banding_index = 1.0

    return boundary_density, boundary_mean_width, banding_index


def _glcm_features(
    img: np.ndarray,
    distances: Tuple[int, ...],
    angles_deg: Tuple[float, ...],
    levels: int,
) -> dict:
    """
    Compute GLCM-derived Haralick texture properties.

    Returns dict with keys: contrast, energy, homogeneity, correlation,
    dissimilarity.  Values are averaged across all distance/angle combinations.
    """
    from skimage.feature import graycomatrix, graycoprops

    # Quantise to [0, levels-1]
    img_uint = (img * (levels - 1)).astype(np.uint8)

    angles_rad = [math.radians(a) for a in angles_deg]

    glcm = graycomatrix(
        img_uint,
        distances=list(distances),
        angles=angles_rad,
        levels=levels,
        symmetric=True,
        normed=True,
    )  # shape: (levels, levels, n_distances, n_angles)

    props = {}
    for prop in ("contrast", "energy", "homogeneity", "correlation", "dissimilarity"):
        vals = graycoprops(glcm, prop)  # shape: (n_distances, n_angles)
        props[prop] = float(np.mean(vals))

    return props


def _lbp_features(
    img: np.ndarray,
    radius: int,
    n_points: int,
) -> Tuple[float, float]:
    """
    Compute LBP histogram entropy and uniformity.

    Uniformity = fraction of LBP codes with ≤ 2 bitwise 0→1 transitions
    (the "uniform" patterns that dominate smooth / equiaxed regions).
    """
    from skimage.feature import local_binary_pattern
    from scipy.stats import entropy as scipy_entropy

    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")

    # Histogram over uniform LBP codes: 0..n_points (uniform) + n_points+1 (non-uniform)
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist_norm = hist.astype(np.float64)
    hist_norm /= hist_norm.sum() if hist_norm.sum() > 0 else 1.0

    lbp_entropy = float(scipy_entropy(hist_norm + 1e-12))  # avoid log(0)

    # Uniform codes are indices 0..n_points; non-uniform is index n_points+1
    uniform_fraction = float(hist_norm[:n_points + 1].sum())

    return lbp_entropy, uniform_fraction


def _local_contrast(img: np.ndarray, patch_size: int) -> Tuple[float, float]:
    """
    Compute mean and std of local standard deviation over sliding patches.

    Uses a fast approximation: local variance = E[x²] - E[x]² via uniform
    filter, avoiding an explicit sliding window loop.
    """
    from scipy.ndimage import uniform_filter

    p = patch_size
    mean_sq = uniform_filter(img ** 2, size=p)
    sq_mean = uniform_filter(img, size=p) ** 2
    local_var = np.maximum(mean_sq - sq_mean, 0.0)
    local_std = np.sqrt(local_var)

    return float(np.mean(local_std)), float(np.std(local_std))
