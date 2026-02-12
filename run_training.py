#!/usr/bin/env python3
"""
CI/CD training script for SEM Microstructure Heat Treatment Prediction.

All inputs are configured via environment variables (see README or plan).
Outputs a timestamped run directory under runs/ with metrics JSON,
plots, and the saved best model.

Usage:
    # Local CSV:
    DATASET_PATH=datasets/metadata_latest.csv python run_training.py

    # Google Sheets:
    GOOGLE_SHEET_ID=... python run_training.py

    # Custom targets + model selection:
    TARGET_COLUMNS="Cycle1_HoldingTemp (C)" REGRESSION_MODELS=RF,GBR python run_training.py
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # headless rendering for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config, ensure_dir, load_config
from src.model_trainer import (
    ModelTrainer,
    evaluate_model,
    plot_learning_curves,
    plot_model_comparison,
    plot_predictions,
)
from src.preprocessing import FeaturePreprocessor
from src.preprocessing.pipeline import (
    EncodingConfig,
    MissingDataConfig,
    PreprocessingConfig,
    ScalingConfig,
)


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------

def parse_env():
    """Read all configuration from environment variables with defaults."""
    target_str = os.getenv(
        "TARGET_COLUMNS",
        "Cycle1_HoldingTemp (C),Cycle1_HoldingTime (min)"
    )
    backbone_str = os.getenv("BACKBONES", "resnet50,vgg16")
    model_str = os.getenv("REGRESSION_MODELS", "RF,GBR,ABR")

    return {
        "worksheet_name": os.getenv("WORKSHEET_NAME", "Sheet1"),
        "dataset_path": os.getenv("DATASET_PATH", ""),
        "target_columns": [c.strip() for c in target_str.split(",") if c.strip()],
        "backbones": [b.strip() for b in backbone_str.split(",") if b.strip()],
        "regression_models": [m.strip() for m in model_str.split(",") if m.strip()],
        "n_estimators": int(os.getenv("N_ESTIMATORS", "100")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "0.1")),
        "test_size": float(os.getenv("TEST_SIZE", "0.15")),
        "val_size": float(os.getenv("VAL_SIZE", "0.15")),
        "random_seed": int(os.getenv("RANDOM_SEED", "42")),
        "scaling_method": os.getenv("SCALING_METHOD", "standard"),
        "imputer_strategy": os.getenv("IMPUTER_STRATEGY", "median"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_column(df, pattern):
    """Find a DataFrame column matching *pattern*, normalising whitespace."""
    pattern_norm = re.sub(r"\s+", " ", pattern).strip()
    for col in df.columns:
        if re.sub(r"\s+", " ", col).strip() == pattern_norm:
            return col
    return None


CHEMICAL_COLUMNS = [
    "C (wt.%)", "Mn (wt.%)", "Si", "Cr (wt.%)", "P", "S",
    "Mo", "Cu", "Ni", "Al", "Nb", "V", "Ti", "Fe",
]


def plot_residuals(Y_true, Y_pred, target_columns, save_path):
    """Histogram of residuals per target."""
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    n = len(target_columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, target_columns)):
        residuals = Y_true[:, i] - Y_pred[:, i]
        ax.hist(residuals, bins=15, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{col}")

    plt.suptitle("Residual Distribution (Test Set)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(env):
    """Load dataset from local CSV or Google Sheets."""
    if env["dataset_path"]:
        path = env["dataset_path"]
        print(f"  Loading local CSV: {path}")
        df_raw = pd.read_csv(path, header=1)
    else:
        print(f"  Loading from Google Sheets (worksheet: {env['worksheet_name']})...")
        config = load_config()
        config.worksheet_name = env["worksheet_name"]
        from src.data_loader import DataLoader
        loader = DataLoader(config)
        _, labels_df = loader.load_data()
        df_raw = labels_df

    # Filter to rows with actual data
    df = df_raw[df_raw["C (wt.%)"].notna()].copy().reset_index(drop=True)
    print(f"  Usable samples: {len(df)} (from {len(df_raw)} raw rows)")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    env = parse_env()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    ensure_dir(run_dir)

    print("=" * 60)
    print("SEM Microstructure — CI/CD Training Run")
    print(f"Run ID: {run_id}")
    print("=" * 60)

    # -- Print config -------------------------------------------------------
    print("\n[1/8] Configuration")
    for k, v in env.items():
        print(f"  {k}: {v}")

    # -- Load data ----------------------------------------------------------
    print("\n[2/8] Loading data...")
    df = load_data(env)

    # -- Resolve target columns ---------------------------------------------
    print("\n[3/8] Resolving target columns...")
    resolved_targets = []
    for pattern in env["target_columns"]:
        actual = find_column(df, pattern)
        if actual is None:
            print(f"  ERROR: target column '{pattern}' not found in dataset")
            sys.exit(1)
        resolved_targets.append(actual)
        nn = df[actual].notna().sum()
        print(f"  {pattern} -> {repr(actual)}  ({nn}/{len(df)} available)")

    # Filter to samples with all targets non-null
    mask = pd.Series(True, index=df.index)
    for col in resolved_targets:
        mask &= df[col].notna()
    df_filtered = df[mask].copy().reset_index(drop=True)
    print(f"  Samples with all targets: {len(df_filtered)}")

    Y = df_filtered[resolved_targets].values.astype(np.float64)
    target_names = [re.sub(r"\s+", " ", c).strip() for c in env["target_columns"]]

    # -- Tabular features ---------------------------------------------------
    print("\n[4/8] Preprocessing tabular features...")
    feature_cols = [c for c in CHEMICAL_COLUMNS if c in df_filtered.columns]
    print(f"  Chemical input features: {len(feature_cols)}")

    preproc_config = PreprocessingConfig(
        missing_data=MissingDataConfig(
            column_drop_threshold=0.95,
            row_fill_threshold=0.50,
            numeric_fill_strategy=env["imputer_strategy"],
            categorical_fill_strategy="mode",
        ),
        scaling=ScalingConfig(method=env["scaling_method"], enabled=True),
        encoding=EncodingConfig(categorical="onehot", max_categories=50),
    )

    preprocessor = FeaturePreprocessor(preproc_config)
    X_tabular = preprocessor.fit_transform(df_filtered[feature_cols].copy())
    print(f"  Tabular features after preprocessing: {X_tabular.shape[1]}")

    # -- Image features -----------------------------------------------------
    print("\n[5/8] Image features...")

    # Compute expected feature dimension from backbone names
    backbone_dims = {
        "vgg16": 512, "vgg19": 512,
        "resnet18": 512, "resnet50": 2048, "resnet101": 2048,
        "densenet121": 1024,
        "efficientnet_b0": 1280, "efficientnet_b4": 1792,
        "convnext_tiny": 768, "mobilenet_v3": 960,
        "dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024,
    }
    feature_dim = sum(backbone_dims.get(b, 512) for b in env["backbones"])

    # TODO: Replace with actual image extraction when SEM images are available
    np.random.seed(env["random_seed"])
    X_images = np.random.randn(len(df_filtered), feature_dim).astype(np.float32)
    print(f"  Simulated image features: {X_images.shape} (backbones: {env['backbones']})")

    # -- Combine features ---------------------------------------------------
    print("\n[6/8] Combining features...")
    X_combined = np.concatenate([X_images, X_tabular], axis=1)
    print(f"  Combined: {X_images.shape[1]} (image) + {X_tabular.shape[1]} (tabular) = {X_combined.shape[1]}")

    # -- Train --------------------------------------------------------------
    print("\n[7/8] Training models...")
    config = Config(random_seed=env["random_seed"], model_dir=run_dir)
    trainer = ModelTrainer(
        config,
        n_estimators=env["n_estimators"],
        learning_rate=env["learning_rate"],
        model_selection=env["regression_models"],
    )

    X_train, X_val, X_test, Y_train, Y_val, Y_test = trainer.split_data(
        X_combined, Y,
        test_size=env["test_size"],
        val_size=env["val_size"],
    )

    start_time = time.time()
    best_name = trainer.train_and_evaluate(
        X_train, Y_train, X_val, Y_val, target_names,
        track_learning_curves=True,
    )
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.1f}s")

    # -- Evaluate all splits ------------------------------------------------
    print("\n[8/8] Evaluating & generating artifacts...")
    all_results = {}
    for name, model in trainer.fitted_models.items():
        all_results[name] = {}
        for split_name, X_split, Y_split in [
            ("train", X_train, Y_train),
            ("val", X_val, Y_val),
            ("test", X_test, Y_test),
        ]:
            metrics, _, per_target = evaluate_model(
                model, X_split, Y_split, target_names, split_name=split_name
            )
            all_results[name][split_name] = {
                "R2": round(metrics["R2"], 4),
                "MAE": round(metrics["MAE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
            }

    # Test results for comparison plot
    trainer.test_results = {}
    for name in trainer.fitted_models:
        trainer.test_results[name] = all_results[name]["test"]

    # -- Best model per-target breakdown ------------------------------------
    best_model = trainer.best_model
    Y_pred_test = best_model.predict(X_test)
    if Y_pred_test.ndim == 1:
        Y_pred_test = Y_pred_test.reshape(-1, 1)
    if Y_test.ndim == 1:
        Y_test_2d = Y_test.reshape(-1, 1)
    else:
        Y_test_2d = Y_test

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    per_target_test = {}
    for i, col in enumerate(target_names):
        per_target_test[col] = {
            "R2": round(r2_score(Y_test_2d[:, i], Y_pred_test[:, i]), 4),
            "MAE": round(mean_absolute_error(Y_test_2d[:, i], Y_pred_test[:, i]), 4),
            "RMSE": round(math.sqrt(mean_squared_error(Y_test_2d[:, i], Y_pred_test[:, i])), 4),
        }

    # -- Generate plots -----------------------------------------------------
    artifacts = []

    # Model comparison bar chart
    comp_path = os.path.join(run_dir, "model_comparison.png")
    plot_model_comparison(trainer.test_results, save_path=comp_path, show=False)
    artifacts.append("model_comparison.png")

    # Predicted vs actual (best model)
    pred_path = os.path.join(run_dir, f"predictions_{best_name}.png")
    plot_predictions(Y_test_2d, Y_pred_test, target_names, save_path=pred_path)
    artifacts.append(f"predictions_{best_name}.png")

    # Residual distribution (best model)
    resid_path = os.path.join(run_dir, f"residuals_{best_name}.png")
    plot_residuals(Y_test_2d, Y_pred_test, target_names, save_path=resid_path)
    artifacts.append(f"residuals_{best_name}.png")

    # Learning curves (boosting models, single-output only)
    for lc_name, history in trainer.learning_histories.items():
        lc_path = os.path.join(run_dir, f"learning_curves_{lc_name}.png")
        plot_learning_curves(history, lc_name, save_path=lc_path, show=False)
        artifacts.append(f"learning_curves_{lc_name}.png")

    # Save best model
    model_path = os.path.join(run_dir, "best_model.joblib")
    import joblib
    joblib.dump(best_model, model_path)
    artifacts.append("best_model.joblib")
    print(f"  Saved best model to {model_path}")

    # -- Write run_summary.json ---------------------------------------------
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": env["dataset_path"] or f"google_sheets:{env['worksheet_name']}",
            "worksheet_name": env["worksheet_name"],
            "target_columns": target_names,
            "backbones": env["backbones"],
            "regression_models": env["regression_models"],
            "scaler": env["scaling_method"],
            "imputer": env["imputer_strategy"],
            "n_estimators": env["n_estimators"],
            "learning_rate": env["learning_rate"],
            "test_size": env["test_size"],
            "val_size": env["val_size"],
            "random_seed": env["random_seed"],
            "feature_dimensions": {
                "tabular": int(X_tabular.shape[1]),
                "image": int(X_images.shape[1]),
                "total": int(X_combined.shape[1]),
            },
            "samples": {
                "total": int(len(df_filtered)),
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
        },
        "results": all_results,
        "best_model": {
            "name": best_name,
            "val_R2": all_results[best_name]["val"]["R2"],
            "test_R2": all_results[best_name]["test"]["R2"],
            "per_target": per_target_test,
        },
        "training_time_seconds": round(training_time, 2),
        "artifacts": artifacts,
    }

    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # -- Final output -------------------------------------------------------
    print("\n" + "=" * 60)
    print("Run complete!")
    print(f"  Run directory: {run_dir}")
    print(f"  Best model: {best_name}")
    print(f"  Test R2: {all_results[best_name]['test']['R2']:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Artifacts: {len(artifacts)} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
