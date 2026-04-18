#!/usr/bin/env python3
"""
Standalone SEM image cleaning script.

Cleans all images in an input directory and writes results to an output
directory.  Use this to pre-process images once before training, or to
regenerate the cleaned image set after downloading new images.

Usage
-----
# Classical backend (default):
python clean_images.py

# Explicit paths:
python clean_images.py --input data/temp_images --output data/images_cleaned

# Claude Vision backend (requires ANTHROPIC_API_KEY):
python clean_images.py --backend claude

# Force re-process already-cleaned images:
python clean_images.py --force

# Dry-run: audit without writing output:
python clean_images.py --dry-run

# Save a JSON report:
python clean_images.py --report runs/clean_report.json

Environment variables
---------------------
ANTHROPIC_API_KEY   Required when --backend=claude
CLEANING_BACKEND    Overrides --backend default ("classical")
"""

import argparse
import json
import logging
import os
import sys

# Allow running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config import ImageCleaningConfig
from src.image_cleaner import ImageCleaner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean SEM microstructure images (scale-bar removal, CLAHE, denoise)"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.getenv("IMAGES_DIR", "data/temp_images"),
        help="Input directory of raw images (default: data/temp_images)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.getenv("CLEANING_OUTPUT_DIR", "data/images_cleaned"),
        help="Output directory for cleaned images (default: data/images_cleaned)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["classical", "claude"],
        default=os.getenv("CLEANING_BACKEND", "classical"),
        help="Cleaning backend: 'classical' (default) or 'claude' (Vision API)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=512,
        help="Output image size in pixels (square, default: 512)",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE contrast enhancement",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable bilateral denoising",
    )
    parser.add_argument(
        "--gradient-threshold",
        type=float,
        default=3.0,
        help="Scale-bar edge detection threshold factor (default: 3.0, higher = less aggressive)",
    )
    parser.add_argument(
        "--annotations",
        default="data/scale_bar_annotations.json",
        help="Path to Claude annotation cache JSON (default: data/scale_bar_annotations.json)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process images that already exist in output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit input images without writing any output",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Path to save a JSON cleaning report (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-image results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    log = logging.getLogger(__name__)

    print("=" * 60)
    print("SEM Image Cleaner")
    print("=" * 60)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Backend  : {args.backend}")
    print(f"  Size     : {args.output_size}×{args.output_size}")
    print(f"  CLAHE    : {'off' if args.no_clahe else 'on'}")
    print(f"  Denoise  : {'off' if args.no_denoise else 'on'}")
    if args.dry_run:
        print("  Mode     : DRY RUN (no files written)")

    if not os.path.isdir(args.input):
        print(f"\nERROR: Input directory not found: {args.input}")
        sys.exit(1)

    if args.backend == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable is required for --backend=claude")
        sys.exit(1)

    cfg = ImageCleaningConfig(
        backend=args.backend,
        output_size=args.output_size,
        gradient_threshold_factor=args.gradient_threshold,
        clahe_enabled=not args.no_clahe,
        denoise_enabled=not args.no_denoise,
        annotations_path=args.annotations,
        output_dir=args.output,
    )

    cleaner = ImageCleaner(cfg)

    if args.dry_run:
        # Audit only: open each image, check it's readable, print stats
        from PIL import Image
        import numpy as np
        import glob

        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        files = sorted(
            p for p in (
                glob.glob(os.path.join(args.input, f"*{e}")) for e in exts
                for _ in [None]
            )
        )
        # flatten
        files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in set(exts)
        )
        ok, bad = 0, []
        for p in files:
            try:
                img = Image.open(p)
                arr = np.array(img.convert("L"))
                crop_row = cleaner._detect_scale_bar_row(arr) if cfg.backend == "classical" else None
                crop_px = arr.shape[0] - crop_row if crop_row else None
                if args.verbose:
                    print(f"  {os.path.basename(p):50s}  {img.size}  crop≈{crop_px}px")
                ok += 1
            except Exception as e:
                bad.append((os.path.basename(p), str(e)))
        print(f"\nAudit complete: {ok} readable, {len(bad)} errors")
        for fname, err in bad:
            print(f"  ERROR: {fname} — {err}")
        return

    print()
    report = cleaner.clean_directory(args.input, args.output, force=args.force)

    print()
    print(report.summary())

    non_dp = report.non_dp_images()
    if non_dp:
        print(f"\nNon-DP (single-phase) images — will be median-imputed in pipeline:")
        for fname in non_dp:
            print(f"  {fname}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.report}")

    errors = [r for r in report.results if r.status == "error"]
    if errors:
        print(f"\n{len(errors)} images failed to clean:")
        for r in errors:
            print(f"  {r.filename}: {r.reason}")


if __name__ == "__main__":
    main()
