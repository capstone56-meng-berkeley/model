#!/usr/bin/env python3
"""
Main entry point for the SEM Microstructure Heat Treatment Prediction Model.

Usage:
    python main.py                  # Run full pipeline
    python main.py --download-only  # Only download images
    python main.py --train-only     # Only train (assumes images exist)
"""

import argparse
import sys

from src.config import load_config
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
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
    print("\n[1/5] Loading configuration...")
    config = load_config(args.config, args.env)
    print(f"  Data source: {config.data_source}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")

    if not config.features.label_columns:
        print("\n⚠ Warning: No label columns configured in config.json")
        print("  Configure features.label_columns to enable training")

    # Initialize data loader
    data_loader = DataLoader(config)

    # Step 2: Load/download data
    if not args.train_only:
        print("\n[2/5] Loading data...")
        image_paths, labels_df = data_loader.load_data()
        print(f"  Images: {len(image_paths)}")
        print(f"  Labels shape: {labels_df.shape if not labels_df.empty else 'N/A'}")
    else:
        print("\n[2/5] Skipping download (--train-only mode)")
        # Load from existing temp dir or local
        import os
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
        print("\n✓ Download complete (--download-only mode)")
        return

    # Step 3: Extract features
    print("\n[3/5] Extracting features...")
    feature_extractor = FeatureExtractor(config)

    label_columns = config.features.label_columns

    if args.no_cache:
        import os
        if os.path.exists(config.feature_cache):
            os.remove(config.feature_cache)
            print("  Removed cached features")

    X, Y, filenames = feature_extractor.load_or_extract_features(
        image_paths=image_paths,
        labels_df=labels_df,
        label_columns=label_columns
    )
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {Y.shape}")

    # Check if we have labels to train
    if Y.shape[1] == 0 or not label_columns:
        print("\n⚠ No labels configured. Cannot train model.")
        print("  Configure features.label_columns in config.json")
        print("\n✓ Feature extraction complete")
        return

    # Step 4: Train model
    print("\n[4/5] Training models...")
    trainer = ModelTrainer(config)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = trainer.split_data(X, Y)

    best_model_name = trainer.train_and_evaluate(
        X_train, Y_train, X_val, Y_val, label_columns
    )

    # Step 5: Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    test_metrics, Y_pred = trainer.evaluate_on_test(X_test, Y_test, label_columns)

    # Save model
    model_path = trainer.save_model()

    # Plot predictions
    import os
    plot_path = os.path.join(config.model_dir, "predictions.png")
    plot_predictions(Y_test, Y_pred, label_columns, save_path=plot_path)

    print("\n" + "=" * 60)
    print("✓ Pipeline complete!")
    print(f"  Best model: {best_model_name}")
    print(f"  Test R2: {test_metrics['R2']:.4f}")
    print(f"  Model saved: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
