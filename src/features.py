"""
Feature preparation pipeline — canonical library for all feature streams.

Separates the one-time *write* operations (download, CNN extraction, morph
extraction) from the per-run *read* operations (load + align + concatenate).

Typical CI/CD or script usage
------------------------------
    from src.features import FeaturePipeline

    fp = FeaturePipeline(
        data_dir="data",
        temp_dir="data/temp_images",
        features_dir="features",
    )

    # ── one-time setup (idempotent — skips existing caches) ──────────────
    image_paths = fp.download_images(df, folder_col="augumented_data")
    fp.extract_cnn(image_paths, backbones=["dinov2_vitb14", "resnet50"])
    fp.extract_morph(image_paths)

    # ── per-run (read-only) ───────────────────────────────────────────────
    X_img   = fp.load_image_features("dinov2_vitb14", df["id"])
    X_morph = fp.load_morph_features(df["id"])
    X_full, log = fp.build_feature_matrix(X_tab, "dinov2_vitb14", df["id"])

Public API
----------
FeaturePipeline(data_dir, temp_dir, features_dir, credentials_path, token_path)
  .download_images(df, folder_col, id_col)   → list[str]
  .extract_cnn(image_paths, backbones, ...)  → dict[str, Path]
  .extract_morph(image_paths, ...)           → Path
  .load_image_features(backbone, df_ids)     → np.ndarray | None
  .load_morph_features(df_ids)               → np.ndarray | None
  .build_feature_matrix(X_tab, backbone, df_ids) → tuple[np.ndarray, str]
  .verify()                                  → dict  (cache status report)
"""

from __future__ import annotations

import gc
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Filename suffix pattern shared across all alignment helpers.
# Strips e.g. "_F_0.jpg" so the base row-id can be matched.
_F_RE = re.compile(r"_F_\d+\.[a-z]+$", re.IGNORECASE)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_id(raw: str) -> str:
    """Lower-case, collapse hyphens/spaces to underscores."""
    return re.sub(r"[-\s]+", "_", str(raw).strip().lower())


def _align_cache_to_ids(
    cache_path: Union[str, Path],
    df_ids: pd.Series,
) -> tuple[np.ndarray, int]:
    """
    Load an .npz image cache and align rows to *df_ids* by row ID.

    The cache must contain arrays ``X`` (N_images, feat_dim) and
    ``filenames`` (N_images,).  Filenames are normalised by stripping the
    ``_F_<n>.<ext>`` suffix so they match the row IDs in *df_ids*.

    Multiple images that map to the same row ID are averaged.
    Rows with no matching image are filled with column means.

    Returns
    -------
    X_aligned : float32 ndarray of shape (len(df_ids), feat_dim)
    n_matched : number of rows that had at least one matching image
    """
    data = np.load(cache_path, allow_pickle=True)
    X_cache: np.ndarray = data["X"].astype(np.float32)
    filenames: list[str] = list(data["filenames"])

    id_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, fname in enumerate(filenames):
        rid = _normalise_id(_F_RE.sub("", os.path.basename(fname)))
        id_to_indices[rid].append(i)

    feat_dim = X_cache.shape[1]
    n_rows = len(df_ids)
    X_aligned = np.full((n_rows, feat_dim), np.nan, dtype=np.float32)
    n_matched = 0

    for r, row_id in enumerate(df_ids.astype(str).map(_normalise_id)):
        idxs = id_to_indices.get(row_id, [])
        if idxs:
            X_aligned[r] = X_cache[idxs].mean(axis=0)
            n_matched += 1

    # Fill rows with no images using column means of matched rows
    col_means = np.nanmean(X_aligned, axis=0)
    nan_rows = np.isnan(X_aligned).any(axis=1)
    if nan_rows.any():
        X_aligned[nan_rows] = col_means

    return X_aligned, n_matched


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Manages all three feature streams: CNN image embeddings, morphological
    features, and (externally built) tabular features.

    Parameters
    ----------
    data_dir :
        Root data directory.  CNN caches are written as
        ``<data_dir>/image_cache_<backbone>.npz``.
    temp_dir :
        Directory where raw images live (downloaded or placed manually).
        Defaults to ``<data_dir>/temp_images``.
    features_dir :
        Directory for non-image feature caches (morph .npz).
        Defaults to ``<data_dir>/../features``.
    credentials_path :
        Path to Google OAuth ``credentials.json`` (only needed for
        :meth:`download_images`).
    token_path :
        Path to cached Google OAuth token (only needed for
        :meth:`download_images`).
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        temp_dir: Union[str, Path] | None = None,
        features_dir: Union[str, Path] | None = None,
        credentials_path: Union[str, Path] = "credentials.json",
        token_path: Union[str, Path] = "token.json",
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.temp_dir = Path(temp_dir).resolve() if temp_dir else self.data_dir / "temp_images"
        self.features_dir = Path(features_dir).resolve() if features_dir else self.data_dir.parent / "features"
        self.credentials_path = Path(credentials_path).resolve()
        self.token_path = Path(token_path).resolve()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def cnn_cache_path(self, backbone: str) -> Path:
        return self.data_dir / f"image_cache_{backbone}.npz"

    @property
    def morph_cache_path(self) -> Path:
        return self.features_dir / "morph_features_c1.npz"

    # ------------------------------------------------------------------
    # Write operations (idempotent — skip if cache already exists)
    # ------------------------------------------------------------------

    def download_images(
        self,
        df: pd.DataFrame,
        folder_col: str = "augumented_data",
        id_col: str = "id",
        force: bool = False,
    ) -> list[str]:
        """
        Download images from Google Drive folders referenced in *df[folder_col]*.

        Skips entirely if images already exist in ``temp_dir`` (unless
        *force=True*).  Requires ``credentials.json``.

        Parameters
        ----------
        df : DataFrame with at least *id_col* and *folder_col* columns.
        folder_col : Column containing Google Drive folder share URLs.
        id_col : Column containing row identifiers used to name files.
        force : Re-download even if images are already present.

        Returns
        -------
        list of absolute image paths found in ``temp_dir`` after the call.
        """
        existing = self._list_images()
        if existing and not force:
            logger.info(
                "download_images: %d images already in %s — skipping. "
                "Pass force=True to re-download.",
                len(existing), self.temp_dir,
            )
            return existing

        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"credentials.json not found at {self.credentials_path}. "
                "Place images manually in: " + str(self.temp_dir)
            )

        from src.drive_client import GoogleDriveClient

        _folder_re = re.compile(
            r"drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]+)"
        )
        folder_entries: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            cell = row.get(folder_col, "")
            if not isinstance(cell, str):
                continue
            m = _folder_re.search(cell)
            if m:
                folder_entries.append((_normalise_id(str(row[id_col])), m.group(1)))

        if not folder_entries:
            logger.warning("download_images: no Drive folder links found in column %r.", folder_col)
            return self._list_images()

        logger.info("download_images: found %d folder links — starting download.", len(folder_entries))
        drive = GoogleDriveClient(
            credentials_path=str(self.credentials_path),
            token_path=str(self.token_path),
        )
        downloaded = 0
        for row_id, folder_id in folder_entries:
            files = drive.list_files_in_folder(
                folder_id, file_extensions=list(IMAGE_EXTENSIONS)
            )
            for idx, f in enumerate(files):
                ext = os.path.splitext(f["name"])[1].lower() or ".jpg"
                dest = self.temp_dir / f"{row_id}_F_{idx}{ext}"
                if not dest.exists():
                    drive.download_file(f["id"], str(dest))
                    downloaded += 1

        logger.info("download_images: downloaded %d new images.", downloaded)
        return self._list_images()

    def extract_cnn(
        self,
        image_paths: list[str] | None = None,
        backbones: list[str] | None = None,
        img_size: int = 224,
        batch_size: int = 16,
        num_workers: int = 2,
        force: bool = False,
    ) -> dict[str, Path]:
        """
        Extract CNN embeddings for each backbone and write per-backbone caches.

        Skips any backbone whose cache already exists (unless *force=True*).

        Parameters
        ----------
        image_paths : Paths to images.  Defaults to all images in ``temp_dir``.
        backbones : Backbone names to run.  Defaults to all registered backbones.
        img_size, batch_size, num_workers : Passed to :class:`ExtractionConfig`.
        force : Re-extract even if a cache already exists.

        Returns
        -------
        dict mapping backbone name → cache path for every backbone processed.
        """
        from src.extraction.backbones import BackboneRegistry
        from src.extraction.extractor import ExtractionConfig, FeatureExtractor

        if image_paths is None:
            image_paths = self._list_images()
        if not image_paths:
            raise RuntimeError(
                f"No images found in {self.temp_dir}. "
                "Run download_images() first or place images manually."
            )

        if backbones is None:
            backbones = sorted(BackboneRegistry.list_available())

        written: dict[str, Path] = {}
        for backbone_name in backbones:
            cache = self.cnn_cache_path(backbone_name)
            if cache.exists() and not force:
                logger.info("extract_cnn: [%s] cache exists — skipping.", backbone_name)
                written[backbone_name] = cache
                continue

            logger.info("extract_cnn: [%s] extracting from %d images...", backbone_name, len(image_paths))
            try:
                cfg = ExtractionConfig(
                    backbones=[backbone_name],
                    img_size=img_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pooling="avg",
                )
                extractor = FeatureExtractor(cfg)
                X, filenames = extractor.extract_features(image_paths)
                np.savez(str(cache), X=X, filenames=np.array(filenames))
                logger.info("extract_cnn: [%s] saved %s → %s", backbone_name, X.shape, cache)
                written[backbone_name] = cache
                del extractor
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            except Exception as exc:
                logger.error("extract_cnn: [%s] FAILED: %s", backbone_name, exc)

        return written

    def extract_morph(
        self,
        image_paths: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        """
        Extract morphological features and write the morph cache.

        Skips if cache already exists (unless *force=True*).

        Parameters
        ----------
        image_paths : Paths to images.  Defaults to all images in ``temp_dir``.
        force : Re-extract even if cache exists.

        Returns
        -------
        Path to the morph cache .npz file.
        """
        from src.extraction.morphology import MorphologicalExtractor
        from src.extraction.morphology_config import MorphologyConfig

        if image_paths is None:
            image_paths = self._list_images()
        if not image_paths:
            raise RuntimeError(
                f"No images found in {self.temp_dir}. "
                "Run download_images() first or place images manually."
            )

        cache = self.morph_cache_path
        if cache.exists() and not force:
            logger.info("extract_morph: cache exists at %s — skipping.", cache)
            return cache

        logger.info("extract_morph: extracting from %d images...", len(image_paths))
        cfg = MorphologyConfig(cache_path=str(cache))
        extractor = MorphologicalExtractor(cfg)
        X, filenames = extractor.extract(image_paths, use_cache=False)
        # MorphologicalExtractor writes the cache itself when cache_path is set,
        # but save explicitly here for consistency and to capture filenames.
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache), X=X, filenames=np.array(filenames))
        logger.info("extract_morph: saved %s → %s", X.shape, cache)
        return cache

    # ------------------------------------------------------------------
    # Read operations (no side effects — safe for any notebook/CI run)
    # ------------------------------------------------------------------

    def load_image_features(
        self,
        backbone: str,
        df_ids: pd.Series,
    ) -> np.ndarray | None:
        """
        Load CNN embeddings for *backbone* and align to *df_ids*.

        Returns ``None`` (with a warning) if the cache does not exist.

        Parameters
        ----------
        backbone : Backbone name, e.g. ``"dinov2_vitb14"``.
        df_ids : Series of row IDs (used as alignment keys).

        Returns
        -------
        float32 ndarray of shape ``(len(df_ids), feat_dim)``, or ``None``.
        """
        cache = self.cnn_cache_path(backbone)
        if not cache.exists():
            logger.warning(
                "load_image_features: no cache for backbone=%r at %s. "
                "Run prepare_features.ipynb or fp.extract_cnn() first.",
                backbone, cache,
            )
            return None

        X, n_matched = _align_cache_to_ids(cache, df_ids)
        logger.info(
            "load_image_features: backbone=%s  shape=%s  matched=%d/%d",
            backbone, X.shape, n_matched, len(df_ids),
        )
        return X

    def load_morph_features(
        self,
        df_ids: pd.Series,
    ) -> np.ndarray | None:
        """
        Load morphological features and align to *df_ids*.

        The morph cache stores features indexed by image filename; this method
        aligns them to the dataset row order via the shared row-ID convention.
        Preprocessing (standard scaling, median imputation) is applied here so
        the returned array is model-ready.

        Returns ``None`` (with a warning) if the cache does not exist.

        Parameters
        ----------
        df_ids : Series of row IDs (used as alignment keys).

        Returns
        -------
        float64 ndarray of shape ``(len(df_ids), n_morph_features)``, or ``None``.
        """
        cache = self.morph_cache_path
        if not cache.exists():
            logger.warning(
                "load_morph_features: no morph cache at %s. "
                "Run prepare_features.ipynb or fp.extract_morph() first.",
                cache,
            )
            return None

        from src.config import EncodingConfig, MissingDataConfig, PreprocessingConfig, ScalingConfig
        from src.extraction.morphology import MorphologicalExtractor
        from src.preprocessing import FeaturePreprocessor

        cd = np.load(str(cache), allow_pickle=True)
        X_raw: np.ndarray = cd["X"].astype(np.float64)

        # Align by filename if cache stores per-image rows; otherwise use positional
        # alignment assuming the cache was built on the same df ordering.
        if "filenames" in cd:
            # Build id→row mapping from filenames in the cache
            filenames = list(cd["filenames"])
            id_to_rows: dict[str, list[int]] = defaultdict(list)
            for i, fname in enumerate(filenames):
                rid = _normalise_id(_F_RE.sub("", os.path.basename(str(fname))))
                id_to_rows[rid].append(i)

            n_feats = X_raw.shape[1]
            X_aligned = np.full((len(df_ids), n_feats), np.nan, dtype=np.float64)
            n_matched = 0
            for r, row_id in enumerate(df_ids.astype(str).map(_normalise_id)):
                idxs = id_to_rows.get(row_id, [])
                if idxs:
                    X_aligned[r] = np.nanmean(X_raw[idxs], axis=0)
                    n_matched += 1
        else:
            # Legacy cache: rows correspond 1:1 with df rows
            n_rows = min(len(X_raw), len(df_ids))
            X_aligned = np.full((len(df_ids), X_raw.shape[1]), np.nan, dtype=np.float64)
            X_aligned[:n_rows] = X_raw[:n_rows]
            n_matched = int((~np.isnan(X_aligned).any(axis=1)).sum())

        morph_names = MorphologicalExtractor.get_feature_names()
        df_morph = pd.DataFrame(X_aligned, columns=morph_names[:X_aligned.shape[1]])

        prep_cfg = PreprocessingConfig(
            missing_data=MissingDataConfig(
                column_drop_threshold=0.95,
                row_fill_threshold=1.0,
                numeric_fill_strategy="median",
                categorical_fill_strategy="mode",
            ),
            scaling=ScalingConfig(method="standard", enabled=True),
            encoding=EncodingConfig(categorical="onehot", max_categories=50),
        )
        preprocessor = FeaturePreprocessor(prep_cfg)
        X_out = preprocessor.fit_transform(df_morph).astype(np.float64)

        logger.info(
            "load_morph_features: shape=%s  matched=%d/%d",
            X_out.shape, n_matched, len(df_ids),
        )
        return X_out

    def build_feature_matrix(
        self,
        X_tab: np.ndarray,
        backbone: str,
        df_ids: pd.Series,
    ) -> tuple[np.ndarray, str]:
        """
        Concatenate all available feature streams into a single matrix.

        Stream order: image (if available) → morph (if available) → tabular.

        Parameters
        ----------
        X_tab : Preprocessed tabular feature matrix, shape ``(n, d_tab)``.
        backbone : CNN backbone name to use for image features.
        df_ids : Series of row IDs for alignment.

        Returns
        -------
        X_full : Combined float64 ndarray of shape ``(n, d_total)``.
        stream_log : Human-readable summary, e.g.
            ``"512 (image) + 24 (morph) + 38 (tabular) = 574 total"``.
        """
        X_img = self.load_image_features(backbone, df_ids)
        X_morph = self.load_morph_features(df_ids)

        parts = []
        labels = []
        if X_img is not None:
            parts.append(X_img.astype(np.float64))
            labels.append(f"{X_img.shape[1]} (image)")
        if X_morph is not None:
            parts.append(X_morph.astype(np.float64))
            labels.append(f"{X_morph.shape[1]} (morph)")
        parts.append(X_tab.astype(np.float64))
        labels.append(f"{X_tab.shape[1]} (tabular)")

        X_full = np.concatenate(parts, axis=1)
        stream_log = " + ".join(labels) + f" = {X_full.shape[1]} total"
        return X_full, stream_log

    # ------------------------------------------------------------------
    # Verification / status
    # ------------------------------------------------------------------

    def verify(self) -> dict:
        """
        Return a status dict summarising which caches exist and their shapes.

        Suitable for printing in a notebook or asserting in CI::

            status = fp.verify()
            assert status["morph"]["exists"], "morph cache missing — run prepare_features"

        Returns
        -------
        dict with keys ``"images"``, ``"cnn"``, ``"morph"``::

            {
              "images":  {"count": int, "dir": str},
              "cnn":     {backbone: {"exists": bool, "shape": tuple|None, "path": str}},
              "morph":   {"exists": bool, "shape": tuple|None, "path": str},
            }
        """
        from src.extraction.backbones import BackboneRegistry

        status: dict = {}

        # Raw images
        images = self._list_images()
        status["images"] = {"count": len(images), "dir": str(self.temp_dir)}

        # CNN caches
        status["cnn"] = {}
        for backbone in sorted(BackboneRegistry.list_available()):
            cache = self.cnn_cache_path(backbone)
            if cache.exists():
                d = np.load(str(cache), allow_pickle=True)
                shape = tuple(d["X"].shape)
            else:
                shape = None
            status["cnn"][backbone] = {
                "exists": cache.exists(),
                "shape": shape,
                "path": str(cache),
            }

        # Morph cache
        cache = self.morph_cache_path
        if cache.exists():
            d = np.load(str(cache), allow_pickle=True)
            morph_shape = tuple(d["X"].shape)
        else:
            morph_shape = None
        status["morph"] = {
            "exists": cache.exists(),
            "shape": morph_shape,
            "path": str(cache),
        }

        return status

    def print_status(self) -> None:
        """Print a human-readable summary of cache status."""
        s = self.verify()
        print("=" * 60)
        print("FeaturePipeline status")
        print(f"  temp_dir     : {self.temp_dir}")
        print(f"  data_dir     : {self.data_dir}")
        print(f"  features_dir : {self.features_dir}")
        print("=" * 60)
        print(f"  Raw images   : {s['images']['count']} files")
        print()
        print("  CNN caches:")
        for name, info in s["cnn"].items():
            mark = "✓" if info["exists"] else "✗"
            detail = str(info["shape"]) if info["shape"] else "missing"
            print(f"    [{mark}] {name:<22} {detail}")
        print()
        mark = "✓" if s["morph"]["exists"] else "✗"
        detail = str(s["morph"]["shape"]) if s["morph"]["shape"] else "missing"
        print(f"  Morph cache  : [{mark}] {detail}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _list_images(self) -> list[str]:
        """Return sorted list of image paths currently in temp_dir."""
        if not self.temp_dir.exists():
            return []
        return sorted(
            str(self.temp_dir / f)
            for f in os.listdir(self.temp_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        )
