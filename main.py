#!/usr/bin/env python3
"""
Main entry point for the SEM Microstructure Heat Treatment Prediction Model.

Usage:
    python main.py                  # Run full pipeline
    python main.py --download-only  # Only download images
    python main.py --train-only     # Only train (assumes images exist)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.config import load_config, ensure_dir, PreprocessingConfig, MissingDataConfig, ScalingConfig, EncodingConfig
from src.data_loader import DataLoader
from src.extraction import FeatureExtractor, MorphologicalExtractor, MorphologyConfig
from src.extraction.extractor import ExtractionConfig
from src.preprocessing import FeaturePreprocessor
from src.model_trainer import ModelTrainer, plot_predictions
from src.column_sanitizer import sanitize_dataframe
from src.image_cleaner import ImageCleaner

# MICE imputation: correlated alloy elements with random missingness
MICE_COLUMNS = ["cr", "mo", "s", "ni", "al"]

# Binary presence indicators: structural missingness (element not in alloy)
INDICATOR_COLUMNS = ["ti", "nb", "v"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SEM Microstructure Heat Treatment Prediction Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file (default: config.json)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download images, don't train"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train model (assumes images already downloaded)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached features, re-extract"
    )

    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()

    print("=" * 60)
    print("SEM Microstructure Heat Treatment Prediction Model")
    print("=" * 60)

    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config(args.config, args.env)
    print(f"  Data source: {config.data_source}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Backbones: {config.extraction.backbones}")

    if not config.features.label_columns:
        print("\n  Warning: No label columns configured in config.json")
        print("  Configure features.label_columns to enable training")

    # Initialize data loader
    data_loader = DataLoader(config)

    # Step 2: Load/download data
    if not args.train_only:
        print("\n[2/6] Loading data...")
        image_paths, labels_df = data_loader.load_data()
        if not labels_df.empty:
            labels_df = sanitize_dataframe(labels_df)
        print(f"  Images: {len(image_paths)}")
        print(f"  Labels shape: {labels_df.shape if not labels_df.empty else 'N/A'}")
    else:
        print("\n[2/6] Skipping download (--train-only mode)")
        images_dir = config.images_dir
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found: {images_dir}")
            sys.exit(1)

        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_paths = [
            os.path.join(images_dir, f)
            for f in sorted(os.listdir(images_dir))
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        labels_df = None
        print(f"  Found {len(image_paths)} images in {images_dir}")

    if args.download_only:
        print("\n  Download complete (--download-only mode)")
        return

    # Step 3: Clean images
    print("\n[3/7] Cleaning images...")
    raw_images_dir = config.images_dir
    cleaning_cfg = config.image_cleaning
    cleaned_images_dir = cleaning_cfg.output_dir

    if not os.path.isdir(raw_images_dir) or not any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for f in os.listdir(raw_images_dir)
    ):
        print(f"  No images found in {raw_images_dir} — skipping cleaning")
        cleaned_images_dir = raw_images_dir
    else:
        cleaner = ImageCleaner(cleaning_cfg)
        clean_report = cleaner.clean_directory(raw_images_dir, cleaned_images_dir, force=False)
        print(clean_report.summary())
        non_dp = clean_report.non_dp_images()
        if non_dp:
            print(f"  Non-DP images flagged (will be median-imputed): {len(non_dp)}")
        # Replace image_paths with cleaned versions
        image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_paths = [
            os.path.join(cleaned_images_dir, os.path.basename(p))
            for p in image_paths
            if os.path.exists(os.path.join(cleaned_images_dir, os.path.basename(p)))
        ]
        print(f"  Cleaned images available: {len(image_paths)}")

    # Step 4: Extract image features
    print("\n[4/7] Extracting image features...")

    # Build extraction config
    extraction_config = ExtractionConfig(
        backbones=config.extraction.backbones,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pooling=config.extraction.pooling
    )

    if args.no_cache and os.path.exists(config.feature_cache):
        os.remove(config.feature_cache)
        print("  Removed cached features")

    # Check cache
    image_cache_path = config.feature_cache.replace('.npz', '_images.npz')
    if os.path.exists(image_cache_path) and not args.no_cache:
        print(f"  Loading cached image features from {image_cache_path}")
        data = np.load(image_cache_path, allow_pickle=True)
        X_images = data['X']
        image_filenames = list(data['filenames'])
    else:
        feature_extractor = FeatureExtractor(extraction_config)
        X_images, image_filenames = feature_extractor.extract_features(image_paths)

        # Cache image features
        ensure_dir(os.path.dirname(image_cache_path) or '.')
        np.savez(image_cache_path, X=X_images, filenames=image_filenames)
        print(f"  Cached image features to {image_cache_path}")

    print(f"  Image features shape: {X_images.shape}")

    # Step 4: Preprocess tabular features
    print("\n[5/7] Preprocessing tabular features...")

    X_tabular = None
    if labels_df is not None and not labels_df.empty and config.features.feature_columns:
        # Build preprocessing config from main config
        preproc_config = PreprocessingConfig(
            missing_data=config.features.preprocessing.missing_data,
            scaling=config.features.preprocessing.scaling,
            encoding=config.features.preprocessing.encoding
        )

        feature_cols = config.features.feature_columns
        mice_cols = [c for c in MICE_COLUMNS if c in feature_cols]
        indicator_cols = [c for c in INDICATOR_COLUMNS if c in feature_cols]

        preprocessor = FeaturePreprocessor(
            preproc_config,
            column_types=config.features.column_types,
            mice_columns=mice_cols,
            indicator_columns=indicator_cols,
        )
        X_tabular = preprocessor.fit_transform(labels_df[config.features.feature_columns])
        print(f"  Tabular features shape: {X_tabular.shape}")
        print(f"  Tabular feature names: {preprocessor.get_feature_names()[:5]}...")
    else:
        print("  No tabular feature columns configured, skipping")

    # Step 5: Morphological features
    print("\n[6/7] Extracting morphological features...")

    X_morphological = None

    import re as _re_m
    import glob as _glob_m
    from collections import defaultdict as _dd_m

    _F_RE_M2 = _re_m.compile(r'_F_\d+\.[a-z]+$', _re_m.IGNORECASE)

    def _img_key2(path):
        return _F_RE_M2.sub('', os.path.basename(path)).lower()

    def _id_key2(row_id):
        return str(row_id).strip().lower().replace('-', '_').replace(' ', '_')

    images_dir_m = config.images_dir  # data/temp_images in drive mode
    all_imgs_m2 = sorted(
        _glob_m.glob(os.path.join(images_dir_m, '*.jpg')) +
        _glob_m.glob(os.path.join(images_dir_m, '*.png'))
    )

    if not all_imgs_m2:
        print(f"  No images found in {images_dir_m} — skipping morphological features")
    elif labels_df is None or labels_df.empty or 'id' not in labels_df.columns:
        print("  No labels/id column — skipping morphological features")
    else:
        key_to_paths_m2 = _dd_m(list)
        for p in all_imgs_m2:
            key_to_paths_m2[_img_key2(p)].append(p)

        morph_cfg2 = MorphologyConfig(cache_path=config.morph_cache)
        morph_extractor2 = MorphologicalExtractor(morph_cfg2)
        morph_feat_names = morph_extractor2.get_feature_names()
        n_feats_m2 = len(morph_feat_names)

        if config.morph_cache and os.path.exists(config.morph_cache):
            print(f"  Loading morphological features from cache: {config.morph_cache}")
            _cd2 = np.load(config.morph_cache, allow_pickle=True)
            X_morph_raw = _cd2["X"].astype(np.float64)
        else:
            X_morph_raw = np.full((len(labels_df), n_feats_m2), np.nan, dtype=np.float64)
            for row_idx, row_id in enumerate(labels_df['id']):
                paths = key_to_paths_m2.get(_id_key2(row_id), [])
                if not paths:
                    continue
                feats = np.vstack([morph_extractor2.extract_single(p) for p in paths])
                X_morph_raw[row_idx] = np.nanmean(feats, axis=0)
            if config.morph_cache:
                ensure_dir(os.path.dirname(config.morph_cache) or ".")
                np.savez(config.morph_cache, X=X_morph_raw)
                print(f"  Cached morphological features to {config.morph_cache}")

        df_morph2 = pd.DataFrame(X_morph_raw, columns=morph_feat_names)
        morph_preproc2 = FeaturePreprocessor(
            PreprocessingConfig(
                missing_data=MissingDataConfig(
                    column_drop_threshold=0.95,
                    row_fill_threshold=1.0,
                    numeric_fill_strategy="median",
                    categorical_fill_strategy="mode",
                ),
                scaling=ScalingConfig(method="standard", enabled=True),
                encoding=EncodingConfig(categorical="onehot", max_categories=50),
            )
        )
        X_morphological = morph_preproc2.fit_transform(df_morph2)
        n_matched_m2 = int((~np.isnan(X_morph_raw).any(axis=1)).sum())
        print(f"  Rows matched to images: {n_matched_m2}/{len(labels_df)}")
        print(f"  Morphological features: {X_morphological.shape[1]}")

    # Step 6: Concatenate features
    print("\n[7/7] Concatenating features...")

    parts = [X_images]
    dim_log = [f"{X_images.shape[1]} (CNN image)"]

    if X_morphological is not None and X_morphological.shape[1] > 0:
        if X_morphological.shape[0] == X_images.shape[0]:
            parts.append(X_morphological)
            dim_log.append(f"{X_morphological.shape[1]} (morphological)")
        else:
            print(f"  Warning: morphological samples ({X_morphological.shape[0]}) != "
                  f"image samples ({X_images.shape[0]}) — skipping morphological")

    if X_tabular is not None and X_tabular.shape[1] > 0:
        if X_tabular.shape[0] == X_images.shape[0]:
            parts.append(X_tabular)
            dim_log.append(f"{X_tabular.shape[1]} (tabular)")
        else:
            print(f"  Warning: tabular samples ({X_tabular.shape[0]}) != "
                  f"image samples ({X_images.shape[0]}) — skipping tabular")

    X_combined = np.concatenate(parts, axis=1)
    print(f"  Combined: {' + '.join(dim_log)} = {X_combined.shape[1]}")

    # Extract labels
    label_columns = config.features.label_columns
    Y = np.zeros((X_combined.shape[0], len(label_columns)), dtype=np.float32)

    if labels_df is not None and not labels_df.empty and label_columns:
        for i, col in enumerate(label_columns):
            if col in labels_df.columns:
                Y[:, i] = labels_df[col].values[:X_combined.shape[0]].astype(np.float32)

    print(f"  Labels shape: {Y.shape}")

    # Check if we have labels to train
    if Y.shape[1] == 0 or not label_columns:
        print("\n  No labels configured. Cannot train model.")
        print("  Configure features.label_columns in config.json")
        print("\n  Feature extraction complete")
        return

    # Step 7: Train model
    print("\n[7/7] Training models...")
    trainer = ModelTrainer(config)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = trainer.split_data(X_combined, Y)

    best_model_name = trainer.train_and_evaluate(
        X_train, Y_train, X_val, Y_val, label_columns
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics, Y_pred = trainer.evaluate_on_test(X_test, Y_test, label_columns)

    # Save model
    model_path = trainer.save_model()

    # Plot predictions
    plot_path = os.path.join(config.model_dir, "predictions.png")
    plot_predictions(Y_test, Y_pred, label_columns, save_path=plot_path)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Best model: {best_model_name}")
    print(f"  Test R2: {test_metrics['R2']:.4f}")
    print(f"  Model saved: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
