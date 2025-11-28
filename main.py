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

from src.config import load_config, ensure_dir
from src.data_loader import DataLoader
from src.extraction import FeatureExtractor
from src.extraction.extractor import ExtractionConfig
from src.preprocessing import FeaturePreprocessor
from src.preprocessing.pipeline import PreprocessingConfig as PipelinePreprocessingConfig
from src.model_trainer import ModelTrainer, plot_predictions


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

    # Step 3: Extract image features
    print("\n[3/6] Extracting image features...")

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
    print("\n[4/6] Preprocessing tabular features...")

    X_tabular = None
    if labels_df is not None and not labels_df.empty and config.features.feature_columns:
        # Build preprocessing config from main config
        preproc_config = PipelinePreprocessingConfig(
            column_types=config.features.column_types,
            missing_data=config.features.preprocessing.missing_data,
            scaling=config.features.preprocessing.scaling,
            encoding=config.features.preprocessing.encoding
        )

        preprocessor = FeaturePreprocessor(preproc_config)
        X_tabular = preprocessor.fit_transform(labels_df, config.features.feature_columns)
        print(f"  Tabular features shape: {X_tabular.shape}")
        print(f"  Tabular feature names: {preprocessor.get_feature_names()[:5]}...")
    else:
        print("  No tabular feature columns configured, skipping")

    # Step 5: Concatenate features
    print("\n[5/6] Concatenating features...")

    if X_tabular is not None and X_tabular.shape[1] > 0:
        # Ensure same number of samples
        if X_images.shape[0] != X_tabular.shape[0]:
            print(f"  Warning: Image samples ({X_images.shape[0]}) != "
                  f"Tabular samples ({X_tabular.shape[0]})")
            min_samples = min(X_images.shape[0], X_tabular.shape[0])
            X_images = X_images[:min_samples]
            X_tabular = X_tabular[:min_samples]

        X_combined = np.concatenate([X_images, X_tabular], axis=1)
        print(f"  Combined features: {X_images.shape[1]} (image) + "
              f"{X_tabular.shape[1]} (tabular) = {X_combined.shape[1]}")
    else:
        X_combined = X_images
        print(f"  Using image features only: {X_combined.shape}")

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

    # Step 6: Train model
    print("\n[6/6] Training models...")
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
