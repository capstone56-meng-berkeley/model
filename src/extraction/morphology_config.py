"""Configuration dataclass for morphological feature extraction."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MorphologyConfig:
    """
    Configuration for MorphologicalExtractor.

    All spatial parameters (img_size, min_grain_area, local_patch_size,
    lbp_radius, lbp_n_points) are defined in terms of the resized image
    coordinate space (img_size × img_size).
    """

    # ------------------------------------------------------------------ #
    # Image preparation
    # ------------------------------------------------------------------ #
    img_size: int = 512
    """Resize longest axis to this before analysis. Smaller = faster;
    512 retains sufficient detail for grain-level features."""

    scale_bar_mask: Tuple[float, float] = (0.88, 0.80)
    """(row_frac, col_frac): zero out pixels below row_frac AND to the right
    of col_frac.  Covers the scale-bar / annotation region in the bottom-right
    corner.  Set to (1.0, 1.0) to disable masking entirely."""

    # ------------------------------------------------------------------ #
    # Phase segmentation
    # ------------------------------------------------------------------ #
    otsu_fallback_gmm: bool = True
    """If Otsu produces a martensite fraction outside [fraction_min,
    fraction_max], fall back to a 2-component GMM threshold."""

    fraction_min: float = 0.05
    """Minimum plausible martensite fraction before flagging as failed."""

    fraction_max: float = 0.90
    """Maximum plausible martensite fraction before flagging as failed."""

    # ------------------------------------------------------------------ #
    # Region filtering
    # ------------------------------------------------------------------ #
    min_grain_area: int = 30
    """Minimum region area in pixels² to include in grain statistics.
    Removes sub-pixel noise artefacts after labelling."""

    # ------------------------------------------------------------------ #
    # Grain boundary / Canny
    # ------------------------------------------------------------------ #
    boundary_sigma: float = 1.5
    """Gaussian sigma for Canny pre-smoothing."""

    canny_low: float = 0.05
    """Canny low hysteresis threshold (fraction of dtype range)."""

    canny_high: float = 0.15
    """Canny high hysteresis threshold (fraction of dtype range)."""

    # ------------------------------------------------------------------ #
    # Local contrast
    # ------------------------------------------------------------------ #
    local_patch_size: int = 7
    """Side length (px) of the sliding window for local std computation."""

    # ------------------------------------------------------------------ #
    # LBP
    # ------------------------------------------------------------------ #
    lbp_radius: int = 3
    """Neighbourhood radius for Local Binary Pattern."""

    lbp_n_points: int = 24
    """Number of circularly-symmetric sampling points (typically 8*radius)."""

    # ------------------------------------------------------------------ #
    # GLCM
    # ------------------------------------------------------------------ #
    glcm_distances: Tuple[int, ...] = (1, 3)
    """Pixel distances at which the GLCM is computed.  Results are averaged
    across all distances before the five Haralick properties are extracted."""

    glcm_angles_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    """Angles (degrees) at which the GLCM is computed.  Results are averaged
    for rotation invariance."""

    glcm_levels: int = 64
    """Number of grey levels used when quantising the image for GLCM.
    64 is sufficient for SEM images and keeps the matrix manageable."""

    # ------------------------------------------------------------------ #
    # Caching
    # ------------------------------------------------------------------ #
    cache_path: str = "features/morph_features.npz"
    """Path to the .npz cache file.  Set to '' to disable caching."""
