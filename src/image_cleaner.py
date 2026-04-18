"""
SEM microstructure image cleaning pipeline.

Two backends are supported and selected via ``ImageCleaningConfig.backend``:

``"classical"`` (default)
    Fully deterministic, no API calls.  Per image:
    1. Detect the scale-bar / annotation strip at the bottom via horizontal
       Sobel gradient magnitude — the strip boundary is the last row with a
       sustained drop in gradient energy.
    2. Crop above that boundary.
    3. Resize to ``output_size × output_size`` with INTER_AREA (alias-free
       downsampling).
    4. Apply CLAHE to boost local phase contrast.
    5. Denoise with a bilateral filter to suppress JPEG 8×8 artefacts while
       keeping grain boundaries sharp.
    6. Quality gate: reject images whose std falls below ``min_std`` or whose
       retained height is below ``min_height_px`` after cropping.

``"claude"``
    Sends each image to the Claude Vision API (``claude-haiku-4-5-20251001``)
    to obtain:
    - Precise scale-bar bounding box (handles non-standard positions)
    - Phase classification: ``dp_steel`` / ``single_phase`` / ``unclear``
    - Focus quality score (1–5)
    The bounding box is used for the crop step; remaining steps are the same
    as the classical backend.  Results are cached in
    ``annotations_path`` (JSON sidecar) so subsequent runs are free.

Usage::

    from src.image_cleaner import ImageCleaner
    from src.config import ImageCleaningConfig

    cfg = ImageCleaningConfig(backend="classical", output_size=512)
    cleaner = ImageCleaner(cfg)
    report = cleaner.clean_directory("data/temp_images", "data/images_cleaned")
    print(report.summary())
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ImageAnnotation:
    """Claude Vision annotation for one image."""
    filename: str
    scale_bar_bbox: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2
    phase_class: str = "unknown"    # dp_steel | single_phase | unclear
    focus_score: int = 0            # 1-5
    notes: str = ""


@dataclass
class CleanResult:
    filename: str
    status: str          # ok | skipped | error
    reason: str = ""
    crop_px: int = 0     # rows removed from bottom
    phase_class: str = "unknown"
    focus_score: int = 0


@dataclass
class CleanReport:
    results: list[CleanResult] = field(default_factory=list)

    def summary(self) -> str:
        ok      = sum(1 for r in self.results if r.status == "ok")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        errors  = sum(1 for r in self.results if r.status == "error")
        non_dp  = sum(1 for r in self.results if r.phase_class == "single_phase")
        avg_crop = (
            np.mean([r.crop_px for r in self.results if r.status == "ok"])
            if ok else 0.0
        )
        lines = [
            f"Cleaned : {ok}",
            f"Skipped : {skipped}",
            f"Errors  : {errors}",
            f"Non-DP  : {non_dp}  (single-phase, annotated but kept)",
            f"Avg crop: {avg_crop:.1f} px",
        ]
        return "\n".join(lines)

    def non_dp_images(self) -> list[str]:
        return [r.filename for r in self.results if r.phase_class == "single_phase"]

    def to_dict(self) -> dict:
        return {"results": [r.__dict__ for r in self.results]}


# ---------------------------------------------------------------------------
# Core cleaner
# ---------------------------------------------------------------------------

class ImageCleaner:
    """
    Clean SEM microstructure images using the configured backend.

    Parameters
    ----------
    config : ImageCleaningConfig
        Loaded from ``src.config`` or constructed directly.
    """

    def __init__(self, config):
        self.cfg = config
        self._annotations: dict[str, ImageAnnotation] = {}
        if config.annotations_path and Path(config.annotations_path).exists():
            self._load_annotations(config.annotations_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_directory(
        self,
        input_dir: str,
        output_dir: str,
        force: bool = False,
    ) -> CleanReport:
        """
        Clean all images in *input_dir*, write results to *output_dir*.

        Parameters
        ----------
        input_dir : str
        output_dir : str
        force : bool
            Re-process images that already exist in *output_dir*.

        Returns
        -------
        CleanReport
        """
        in_path  = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        logger.info(f"Found {len(image_files)} images in {input_dir}")

        # --- Claude annotation pass (batch before processing) ---
        if self.cfg.backend == "claude":
            unannotated = [
                p for p in image_files
                if p.name not in self._annotations
            ]
            if unannotated:
                logger.info(f"Annotating {len(unannotated)} images via Claude Vision...")
                self._annotate_batch(unannotated)
                if self.cfg.annotations_path:
                    self._save_annotations(self.cfg.annotations_path)

        report = CleanReport()
        for img_path in image_files:
            dest = out_path / img_path.name
            if dest.exists() and not force:
                report.results.append(CleanResult(
                    filename=img_path.name, status="skipped", reason="already exists"
                ))
                continue
            result = self._clean_one(img_path, dest)
            report.results.append(result)
            if result.status == "ok":
                logger.debug(f"  {img_path.name}: crop={result.crop_px}px  phase={result.phase_class}")
            elif result.status == "error":
                logger.warning(f"  {img_path.name}: FAILED — {result.reason}")

        return report

    def clean_one(self, input_path: str, output_path: str) -> CleanResult:
        """Clean a single image."""
        return self._clean_one(Path(input_path), Path(output_path))

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _clean_one(self, src: Path, dst: Path) -> CleanResult:
        try:
            img_bgr = cv2.imread(str(src))
            if img_bgr is None:
                return CleanResult(src.name, "error", "cv2 could not read file")

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # --- Determine crop boundary ---
            annotation = self._annotations.get(src.name)
            if annotation and annotation.scale_bar_bbox:
                # Claude gave us a precise bounding box: crop above y1
                crop_row = annotation.scale_bar_bbox[1]
            else:
                crop_row = self._detect_scale_bar_row(gray)

            crop_px = gray.shape[0] - crop_row
            cropped = img_bgr[:crop_row, :]

            # Sanity check: don't accept a crop that removes >40% of the image
            if crop_row < gray.shape[0] * 0.60:
                # Fall back: use bottom 12% fixed crop
                crop_row = int(gray.shape[0] * 0.88)
                crop_px  = gray.shape[0] - crop_row
                cropped  = img_bgr[:crop_row, :]

            if cropped.shape[0] < self.cfg.min_height_px:
                return CleanResult(src.name, "error",
                                   f"only {cropped.shape[0]}px after crop (min={self.cfg.min_height_px})")

            # --- Resize ---
            sz = self.cfg.output_size
            resized = cv2.resize(cropped, (sz, sz), interpolation=cv2.INTER_AREA)

            # --- CLAHE ---
            if self.cfg.clahe_enabled:
                lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=self.cfg.clahe_clip_limit,
                    tileGridSize=(self.cfg.clahe_tile_size, self.cfg.clahe_tile_size)
                )
                l_chan = clahe.apply(l_chan)
                lab = cv2.merge([l_chan, a_chan, b_chan])
                resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # --- Bilateral denoise ---
            if self.cfg.denoise_enabled:
                resized = cv2.bilateralFilter(
                    resized,
                    d=self.cfg.bilateral_d,
                    sigmaColor=self.cfg.bilateral_sigma_color,
                    sigmaSpace=self.cfg.bilateral_sigma_space,
                )

            # --- Quality gate ---
            gray_out = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            std = float(gray_out.std())
            if std < self.cfg.min_std:
                return CleanResult(src.name, "error",
                                   f"output std={std:.1f} below min_std={self.cfg.min_std}")

            # --- Save ---
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            phase  = annotation.phase_class  if annotation else "unknown"
            focus  = annotation.focus_score  if annotation else 0

            return CleanResult(
                src.name, "ok",
                crop_px=crop_px,
                phase_class=phase,
                focus_score=focus,
            )

        except Exception as exc:
            return CleanResult(src.name, "error", str(exc))

    # ------------------------------------------------------------------
    # Scale-bar detection (classical)
    # ------------------------------------------------------------------

    def _detect_scale_bar_row(self, gray: np.ndarray) -> int:
        """
        Return the row index above which the microstructure lives.

        Strategy: compute per-row mean of the horizontal Sobel magnitude.
        The annotation strip creates a sharp horizontal edge that produces
        a spike in this signal.  We scan upward from the bottom and find
        the last row where the signal exceeds ``gradient_threshold`` ×
        the image's median row-gradient.  We then add a small buffer.

        Falls back to a fixed 12% bottom crop if no clear edge is found.
        """
        h, w = gray.shape

        # Horizontal Sobel → captures horizontal lines (scale bar top edge)
        sobel_h = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        row_energy = np.abs(sobel_h).mean(axis=1)   # (H,)

        median_energy = float(np.median(row_energy))
        threshold     = median_energy * self.cfg.gradient_threshold_factor

        # Only look in the bottom 30% of the image
        search_start = int(h * 0.70)
        search_rows  = row_energy[search_start:]

        # Find the topmost spike in the bottom 30%
        spike_indices = np.where(search_rows > threshold)[0]
        if len(spike_indices) == 0:
            # No clear edge detected → use fixed bottom crop
            return int(h * (1.0 - self.cfg.fallback_crop_fraction))

        # The scale bar top edge is the lowest spike row (highest in the strip)
        strip_top = search_start + int(spike_indices.min())

        # Add a small buffer to avoid clipping the last row of microstructure
        strip_top = max(0, strip_top - self.cfg.edge_buffer_px)
        return strip_top

    # ------------------------------------------------------------------
    # Claude Vision annotation
    # ------------------------------------------------------------------

    def _annotate_batch(self, paths: list[Path]) -> None:
        """
        Send images to Claude Vision and store results in self._annotations.
        Processes in batches of cfg.claude_batch_size to respect rate limits.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for claude backend: pip install anthropic"
            ) from exc

        api_key = self.cfg.claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Required for backend='claude'."
            )

        client = anthropic.Anthropic(api_key=api_key)
        batch_size = self.cfg.claude_batch_size

        for i in range(0, len(paths), batch_size):
            batch = paths[i: i + batch_size]
            for img_path in batch:
                try:
                    annotation = self._annotate_one(client, img_path)
                    self._annotations[img_path.name] = annotation
                    logger.debug(
                        f"  {img_path.name}: bbox={annotation.scale_bar_bbox}  "
                        f"phase={annotation.phase_class}  focus={annotation.focus_score}"
                    )
                except Exception as exc:
                    logger.warning(f"  {img_path.name}: annotation failed — {exc}")
                    self._annotations[img_path.name] = ImageAnnotation(
                        filename=img_path.name, notes=str(exc)
                    )

    def _annotate_one(self, client, img_path: Path) -> ImageAnnotation:
        """Send one image to Claude and parse the structured response."""
        # Encode image as base64
        with open(img_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        suffix = img_path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        prompt = (
            "You are analysing an SEM (scanning electron microscope) image of a steel "
            "microstructure.\n\n"
            "Return ONLY a JSON object with these fields (no markdown, no prose):\n"
            "{\n"
            '  "has_scale_bar": true|false,\n'
            '  "scale_bar_bbox": [x1, y1, x2, y2] or null,\n'
            '  "phase_class": "dp_steel"|"single_phase"|"unclear",\n'
            '  "focus_score": 1-5,\n'
            '  "notes": "brief note if anything unusual"\n'
            "}\n\n"
            "Rules:\n"
            "- scale_bar_bbox: pixel coordinates (top-left origin) of the ENTIRE "
            "annotation strip (scale bar + label row). Use the full image width, "
            "e.g. [0, 280, 470, 314] if the strip occupies the bottom 34 rows of a "
            "470×314 image.\n"
            "- phase_class: 'dp_steel' = two-phase ferrite+martensite microstructure "
            "(dark and bright regions); 'single_phase' = uniformly bright/dark "
            "(martensitic or ferritic only); 'unclear' = focus/quality too poor to judge.\n"
            "- focus_score: 1=very blurry, 5=sharp grain boundaries visible.\n"
            "Output JSON only."
        )

        response = client.messages.create(
            model=self.cfg.claude_model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        text = response.content[0].text.strip()
        # Strip any accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)

        bbox = parsed.get("scale_bar_bbox")
        if bbox and len(bbox) == 4:
            bbox = tuple(int(v) for v in bbox)
        else:
            bbox = None

        return ImageAnnotation(
            filename=img_path.name,
            scale_bar_bbox=bbox,
            phase_class=parsed.get("phase_class", "unknown"),
            focus_score=int(parsed.get("focus_score", 0)),
            notes=parsed.get("notes", ""),
        )

    # ------------------------------------------------------------------
    # Annotation persistence
    # ------------------------------------------------------------------

    def _load_annotations(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for item in data.get("annotations", []):
            bbox = item.get("scale_bar_bbox")
            if bbox:
                bbox = tuple(bbox)
            self._annotations[item["filename"]] = ImageAnnotation(
                filename=item["filename"],
                scale_bar_bbox=bbox,
                phase_class=item.get("phase_class", "unknown"),
                focus_score=item.get("focus_score", 0),
                notes=item.get("notes", ""),
            )
        logger.info(f"Loaded {len(self._annotations)} cached annotations from {path}")

    def _save_annotations(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "annotations": [
                {
                    "filename":       a.filename,
                    "scale_bar_bbox": list(a.scale_bar_bbox) if a.scale_bar_bbox else None,
                    "phase_class":    a.phase_class,
                    "focus_score":    a.focus_score,
                    "notes":          a.notes,
                }
                for a in self._annotations.values()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self._annotations)} annotations to {path}")
