"""
image_cleaner.py
----------------
Removes overlaid annotations from SEM microstructure images:
  - Yellow phase labels (F, M, Ferrite, Martensite, etc.)
  - Red/colored circles and arrows
  - White text (scale bar text, panel letters like (a), (b))
  - White scale bar line
  - Bottom data strip (SEM instrument metadata row)
  - Thin white diagonal border lines

Processing order:
  1. Inpaint colored overlays (yellow, red, bright-white text/arrows)
  2. Crop bottom fraction (removes scale bar + data strip)
  3. Optional resize to canonical output_size
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass
class CleanConfig:
    # --- bottom crop (scale bar lives here) ---
    crop_bottom_fraction: float = 0.10   # remove bottom 10% of image height
    crop_top_fraction: float = 0.0       # remove top N% (panel letters sometimes top-left)

    # --- yellow mask (phase labels: F, M, Ferrite …) ---
    inpaint_yellow: bool = True
    yellow_h_low: int = 15              # HSV hue range (0–180 in cv2)
    yellow_h_high: int = 45
    yellow_s_min: int = 80
    yellow_v_min: int = 100

    # --- red mask (annotation circles) ---
    inpaint_red: bool = True
    red_s_min: int = 20              # low threshold catches JPEG halos around red text
    red_v_min: int = 100

    # --- bright-white mask (scale text, panel letters, arrows) ---
    inpaint_white: bool = True
    white_threshold: int = 235          # all channels above this → white
    # Only look for white blobs in corner regions + scale bar row
    # Set to False to inpaint white anywhere in the image
    white_corners_only: bool = True
    corner_fraction: float = 0.20       # top/bottom 20%, left/right 20%

    # --- inpainting ---
    dilate_kernel: int = 3             # px — expands mask to catch anti-aliased edges
    inpaint_radius: int = 7            # cv2.inpaint neighborhood radius
    inpaint_method: int = cv2.INPAINT_TELEA

    # --- output ---
    output_size: Optional[Tuple[int, int]] = None  # (W, H), e.g. (224, 224)


def _make_yellow_mask(hsv: np.ndarray, cfg: CleanConfig) -> np.ndarray:
    lo = np.array([cfg.yellow_h_low, cfg.yellow_s_min, cfg.yellow_v_min], dtype=np.uint8)
    hi = np.array([cfg.yellow_h_high, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _make_red_mask(hsv: np.ndarray, cfg: CleanConfig) -> np.ndarray:
    # Red wraps around hue 0/180
    lo1 = np.array([0, cfg.red_s_min, cfg.red_v_min], dtype=np.uint8)
    hi1 = np.array([8, 255, 255], dtype=np.uint8)
    lo2 = np.array([165, cfg.red_s_min, cfg.red_v_min], dtype=np.uint8)
    hi2 = np.array([180, 255, 255], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))


def _make_white_mask(bgr: np.ndarray, cfg: CleanConfig) -> np.ndarray:
    t = cfg.white_threshold
    mask = np.all(bgr >= t, axis=2).astype(np.uint8) * 255

    if cfg.white_corners_only:
        h, w = bgr.shape[:2]
        cf = cfg.corner_fraction
        region = np.zeros((h, w), dtype=np.uint8)
        cy, cx = int(h * cf), int(w * cf)
        region[:cy, :] = 255          # top strip
        region[h - cy:, :] = 255      # bottom strip
        region[:, :cx] = 255          # left strip
        region[:, w - cx:] = 255      # right strip
        mask = cv2.bitwise_and(mask, region)

    return mask


def _dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, k)


def clean_image(img: np.ndarray, cfg: CleanConfig | None = None) -> np.ndarray:
    """
    Clean a single SEM image (H×W×3 BGR numpy array).
    Returns a cleaned BGR numpy array.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (as returned by cv2.imread).
    cfg : CleanConfig, optional
        Cleaning configuration. Defaults to CleanConfig().

    Returns
    -------
    np.ndarray
        Cleaned image in BGR format.
    """
    if cfg is None:
        cfg = CleanConfig()

    out = img.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

    # --- Build combined annotation mask ---
    combined_mask = np.zeros(out.shape[:2], dtype=np.uint8)

    if cfg.inpaint_yellow:
        combined_mask = cv2.bitwise_or(combined_mask, _make_yellow_mask(hsv, cfg))

    if cfg.inpaint_red:
        combined_mask = cv2.bitwise_or(combined_mask, _make_red_mask(hsv, cfg))

    if cfg.inpaint_white:
        combined_mask = cv2.bitwise_or(combined_mask, _make_white_mask(out, cfg))

    # Dilate to catch anti-aliased edges
    if combined_mask.any():
        combined_mask = _dilate(combined_mask, cfg.dilate_kernel)
        out = cv2.inpaint(out, combined_mask, cfg.inpaint_radius, cfg.inpaint_method)

    # --- Crop top/bottom ---
    h, w = out.shape[:2]
    top = int(h * cfg.crop_top_fraction)
    bottom = h - int(h * cfg.crop_bottom_fraction)
    out = out[top:bottom, :]

    # --- Resize ---
    if cfg.output_size is not None:
        out = cv2.resize(out, cfg.output_size, interpolation=cv2.INTER_AREA)

    return out


def clean_image_file(
    input_path: str,
    output_path: str | None = None,
    cfg: CleanConfig | None = None,
) -> np.ndarray:
    """
    Load an image from disk, clean it, optionally save it, and return the result.

    Parameters
    ----------
    input_path : str
        Path to the source image.
    output_path : str, optional
        If given, write the cleaned image here.
    cfg : CleanConfig, optional
        Cleaning configuration.

    Returns
    -------
    np.ndarray
        Cleaned image in BGR format.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    cleaned = clean_image(img, cfg)

    if output_path is not None:
        cv2.imwrite(output_path, cleaned)

    return cleaned
