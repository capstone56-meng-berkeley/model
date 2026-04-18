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
from sklearn.decomposition import PCA

from src.config import Config, ensure_dir, load_config
from src.model_trainer import (
    ModelTrainer,
    evaluate_model,
    plot_learning_curves,
    plot_model_comparison,
    plot_predictions,
)
from src.config import EncodingConfig, MissingDataConfig, PreprocessingConfig, ScalingConfig, ImageCleaningConfig
from src.preprocessing import FeaturePreprocessor
from src.column_sanitizer import sanitize_dataframe, sanitize_column
from src.extraction.morphology import MorphologicalExtractor
from src.extraction.morphology_config import MorphologyConfig
from src.image_cleaner import ImageCleaner


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------

def parse_env():
    """Read all configuration from environment variables with defaults."""
    target_str = os.getenv(
        "TARGET_COLUMNS",
        "cycle1_holdingtemp_degc,cycle1_holdingtime_min"
    )
    backbone_str = os.getenv("BACKBONES", "resnet50,vgg16")
    model_str = os.getenv("REGRESSION_MODELS", "RF,GBR,ABR")

    return {
        "worksheet_name": os.getenv("WORKSHEET_NAME", "Sheet1"),
        "dataset_path": os.getenv("DATASET_PATH", ""),
        "image_cache_path": os.getenv("IMAGE_CACHE_PATH", ""),
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
        # PCA components for image features (0 = disabled).
        # Recommended: 20–50 when n_samples < 200.
        "pca_components": int(os.getenv("PCA_COMPONENTS", "0")),
        # Path to morphological feature cache (.npz). Empty = use default.
        "morph_cache_path": os.getenv("MORPH_CACHE_PATH", "features/morph_features.npz"),
        # Directory containing downloaded images for morphological extraction.
        "images_dir": os.getenv("IMAGES_DIR", os.path.join("data", "temp_images")),
        # Image cleaning backend: "classical" (default) or "claude"
        "cleaning_backend": os.getenv("CLEANING_BACKEND", "classical"),
        # Output directory for cleaned images (used as images_dir for downstream steps)
        "cleaning_output_dir": os.getenv("CLEANING_OUTPUT_DIR", os.path.join("data", "images_cleaned")),
        # Set to "1" to skip cleaning and use images_dir directly
        "skip_cleaning": os.getenv("SKIP_CLEANING", "0") == "1",
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
    "c", "mn", "si", "cr", "p", "s",
    "mo", "cu", "ni", "al", "nb", "v", "ti", "fe",
]

# MICE imputation: correlated alloy elements with random missingness
MICE_COLUMNS = ["cr", "mo", "s", "ni", "al"]

# Binary presence indicators: structural missingness (element not in alloy)
INDICATOR_COLUMNS = ["ti", "nb", "v"]


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

    # Sanitize all column names to [a-z0-9_] immediately after loading.
    # All downstream code uses sanitized names exclusively.
    df_raw = sanitize_dataframe(df_raw)

    # Filter to rows with actual data
    df = df_raw[df_raw["c"].notna()].copy().reset_index(drop=True)
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
    print("\n[1/10] Configuration")  # steps: config, load, targets, split, tabular, clean, morph, cnn, combine, train
    for k, v in env.items():
        print(f"  {k}: {v}")

    # -- Load data ----------------------------------------------------------
    print("\n[2/10] Loading data...")
    df = load_data(env)

    # -- Resolve target columns ---------------------------------------------
    # Columns are already sanitized by load_data(); sanitize the env var
    # values too so raw names passed via TARGET_COLUMNS env var still work.
    print("\n[3/10] Resolving target columns...")
    resolved_targets = []
    for pattern in env["target_columns"]:
        col = sanitize_column(pattern)
        if col not in df.columns:
            print(f"  ERROR: target column '{col}' not found in dataset")
            sys.exit(1)
        resolved_targets.append(col)
        nn = df[col].notna().sum()
        print(f"  {pattern} -> {repr(col)}  ({nn}/{len(df)} available)")

    # Filter to samples with all targets non-null
    mask = pd.Series(True, index=df.index)
    for col in resolved_targets:
        mask &= df[col].notna()
    df_filtered = df[mask].copy().reset_index(drop=True)
    print(f"  Samples with all targets: {len(df_filtered)}")

    Y = df_filtered[resolved_targets].values.astype(np.float64)
    target_names = [re.sub(r"\s+", " ", c).strip() for c in env["target_columns"]]

    # -- Split rows BEFORE any preprocessing (prevents MICE/imputer leakage) --
    print("\n[4/10] Splitting data...")
    from sklearn.model_selection import train_test_split as _tts

    idx = np.arange(len(df_filtered))
    idx_trainval, idx_test = _tts(
        idx, test_size=env["test_size"], random_state=env["random_seed"]
    )
    idx_train, idx_val = _tts(
        idx_trainval, test_size=env["val_size"], random_state=env["random_seed"]
    )
    df_train = df_filtered.iloc[idx_train].reset_index(drop=True)
    df_val   = df_filtered.iloc[idx_val].reset_index(drop=True)
    df_test  = df_filtered.iloc[idx_test].reset_index(drop=True)
    Y_train  = Y[idx_train]
    Y_val    = Y[idx_val]
    Y_test   = Y[idx_test]
    print(f"  Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # -- Tabular features ---------------------------------------------------
    print("\n[5/10] Preprocessing tabular features...")
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

    mice_cols = [c for c in MICE_COLUMNS if c in feature_cols]
    indicator_cols = [c for c in INDICATOR_COLUMNS if c in feature_cols]

    preprocessor = FeaturePreprocessor(
        preproc_config,
        mice_columns=mice_cols,
        indicator_columns=indicator_cols,
    )
    # Fit only on training rows — val/test use transform() to avoid leakage
    X_tab_train = preprocessor.fit_transform(df_train[feature_cols].copy())
    X_tab_val   = preprocessor.transform(df_val[feature_cols].copy())
    X_tab_test  = preprocessor.transform(df_test[feature_cols].copy())
    print(f"  Tabular features after preprocessing: {X_tab_train.shape[1]}")

    # -- Image cleaning -----------------------------------------------------
    print("\n[6/10] Image cleaning...")

    raw_images_dir     = env["images_dir"]
    cleaning_output_dir = env["cleaning_output_dir"]

    if env["skip_cleaning"]:
        images_dir = raw_images_dir
        print(f"  Skipping (SKIP_CLEANING=1) — using raw images from {images_dir}")
    elif not os.path.isdir(raw_images_dir) or not any(
        f.endswith((".jpg", ".jpeg", ".png"))
        for f in os.listdir(raw_images_dir)
    ):
        images_dir = raw_images_dir
        print(f"  No images found in {raw_images_dir} — skipping cleaning")
    else:
        cleaning_cfg = ImageCleaningConfig(
            backend=env["cleaning_backend"],
            output_dir=cleaning_output_dir,
        )
        cleaner = ImageCleaner(cleaning_cfg)
        clean_report = cleaner.clean_directory(
            raw_images_dir, cleaning_output_dir, force=False
        )
        print(clean_report.summary())
        non_dp = clean_report.non_dp_images()
        if non_dp:
            print(f"  Non-DP images flagged (will be median-imputed): {len(non_dp)}")
        images_dir = cleaning_output_dir

    # -- Morphological features ---------------------------------------------
    print("\n[7/10] Morphological features...")

    X_morph_train = X_morph_val = X_morph_test = None
    morph_feature_names = []
    _F_RE_M = re.compile(r'_F_\d+\.[a-z]+$', re.IGNORECASE)

    def _img_key_m(path):
        return _F_RE_M.sub('', os.path.basename(path)).lower()

    def _id_key_m(row_id):
        return str(row_id).strip().lower().replace('-', '_').replace(' ', '_')

    import glob as _glob
    from collections import defaultdict as _defaultdict

    all_imgs_m = sorted(
        _glob.glob(os.path.join(images_dir, '*.jpg')) +
        _glob.glob(os.path.join(images_dir, '*.png'))
    )

    if not all_imgs_m:
        print(f"  No images found in {images_dir} — skipping morphological features")
    else:
        key_to_paths_m = _defaultdict(list)
        for p in all_imgs_m:
            key_to_paths_m[_img_key_m(p)].append(p)

        morph_cfg = MorphologyConfig(cache_path=env["morph_cache_path"])
        morph_extractor = MorphologicalExtractor(morph_cfg)
        morph_feature_names = morph_extractor.get_feature_names()
        n_feats_m = len(morph_feature_names)

        cache_m = env["morph_cache_path"]
        if cache_m and os.path.exists(cache_m):
            print(f"  Loading morphological features from cache: {cache_m}")
            _cd = np.load(cache_m, allow_pickle=True)
            X_morph_all = _cd["X"].astype(np.float64)
        else:
            X_morph_all = np.full((len(df_filtered), n_feats_m), np.nan, dtype=np.float64)
            id_col = "id" if "id" in df_filtered.columns else df_filtered.columns[0]
            for row_idx, row_id in enumerate(df_filtered[id_col]):
                paths = key_to_paths_m.get(_id_key_m(row_id), [])
                if not paths:
                    continue
                feats = np.vstack([morph_extractor.extract_single(p) for p in paths])
                X_morph_all[row_idx] = np.nanmean(feats, axis=0)
            if cache_m:
                ensure_dir(os.path.dirname(cache_m) or ".")
                np.savez(cache_m, X=X_morph_all)
                print(f"  Cached morphological features to {cache_m}")

        df_morph_all = pd.DataFrame(X_morph_all, columns=morph_feature_names)
        df_morph_train = df_morph_all.iloc[idx_train].reset_index(drop=True)
        df_morph_val   = df_morph_all.iloc[idx_val].reset_index(drop=True)
        df_morph_test  = df_morph_all.iloc[idx_test].reset_index(drop=True)

        morph_preproc_config = PreprocessingConfig(
            missing_data=MissingDataConfig(
                column_drop_threshold=0.95,
                row_fill_threshold=1.0,
                numeric_fill_strategy="median",
                categorical_fill_strategy="mode",
            ),
            scaling=ScalingConfig(method=env["scaling_method"], enabled=True),
            encoding=EncodingConfig(categorical="onehot", max_categories=50),
        )
        morph_preprocessor = FeaturePreprocessor(morph_preproc_config)
        X_morph_train = morph_preprocessor.fit_transform(df_morph_train)
        X_morph_val   = morph_preprocessor.transform(df_morph_val)
        X_morph_test  = morph_preprocessor.transform(df_morph_test)
        n_matched_m = int((~np.isnan(X_morph_all).any(axis=1)).sum())
        print(f"  Rows matched to images: {n_matched_m}/{len(df_filtered)}")
        print(f"  Morphological features: {X_morph_train.shape[1]}")

    # -- CNN image features -------------------------------------------------
    print("\n[8/10] CNN image features...")

    _F_RE_IMG = re.compile(r'_F_\d+\.[a-z]+$', re.IGNORECASE)

    def _img_key_cnn(filename):
        """Normalise 'row_id_F_N.jpg' → 'row_id' for grouping."""
        return _F_RE_IMG.sub('', os.path.basename(filename)).lower()

    def _id_key_cnn(row_id):
        return str(row_id).strip().lower().replace('-', '_').replace(' ', '_')

    cache_path = env["image_cache_path"]
    X_img_train = X_img_val = X_img_test = None

    if cache_path and os.path.exists(cache_path):
        print(f"  Loading image features from cache: {cache_path}")
        _cdata = np.load(cache_path, allow_pickle=True)
        X_cache = _cdata["X"].astype(np.float32)
        cache_filenames = [str(fn) for fn in _cdata["filenames"]]

        if X_cache.shape[0] == len(df_filtered):
            # Cache is already row-aligned (one entry per dataset row)
            X_img_all = X_cache
            print(f"  CNN features (pre-aligned): {X_img_all.shape}")
        else:
            # Cache has one entry per image file — aggregate to one per row
            from collections import defaultdict as _dd_cnn
            key_to_idxs = _dd_cnn(list)
            for fi, fn in enumerate(cache_filenames):
                key_to_idxs[_img_key_cnn(fn)].append(fi)

            id_col_cnn = "id" if "id" in df_filtered.columns else df_filtered.columns[0]
            n_img_feat = X_cache.shape[1]
            X_img_all = np.full((len(df_filtered), n_img_feat), np.nan, dtype=np.float32)
            for row_idx, row_id in enumerate(df_filtered[id_col_cnn]):
                idxs = key_to_idxs.get(_id_key_cnn(row_id), [])
                if idxs:
                    X_img_all[row_idx] = np.nanmean(X_cache[idxs], axis=0)

            n_matched_cnn = int((~np.isnan(X_img_all).any(axis=1)).sum())
            print(f"  CNN features: {n_img_feat} dims  |  "
                  f"rows matched: {n_matched_cnn}/{len(df_filtered)}")

            # Impute NaN rows with column means from matched rows
            col_means = np.nanmean(X_img_all, axis=0)
            nan_rows = np.isnan(X_img_all).any(axis=1)
            X_img_all[nan_rows] = col_means

        X_img_train = X_img_all[idx_train]
        X_img_val   = X_img_all[idx_val]
        X_img_test  = X_img_all[idx_test]
    else:
        if cache_path:
            print(f"  IMAGE_CACHE_PATH set but file not found: {cache_path}")
        print("  No CNN image cache available — continuing without CNN features")

    # -- Combine features ---------------------------------------------------
    print("\n[9/10] Combining features...")

    parts_train, parts_val, parts_test = [], [], []
    dim_log = []

    if X_img_train is not None:
        parts_train.append(X_img_train)
        parts_val.append(X_img_val)
        parts_test.append(X_img_test)
        image_dim = int(X_img_train.shape[1])
        dim_log.append(f"{image_dim} (CNN image)")
    else:
        image_dim = 0

    if X_morph_train is not None:
        parts_train.append(X_morph_train)
        parts_val.append(X_morph_val)
        parts_test.append(X_morph_test)
        dim_log.append(f"{X_morph_train.shape[1]} (morphological)")

    parts_train.append(X_tab_train)
    parts_val.append(X_tab_val)
    parts_test.append(X_tab_test)
    dim_log.append(f"{X_tab_train.shape[1]} (tabular)")

    X_train = np.concatenate(parts_train, axis=1)
    X_val   = np.concatenate(parts_val,   axis=1)
    X_test  = np.concatenate(parts_test,  axis=1)

    print(f"  Combined: {' + '.join(dim_log)} = {X_train.shape[1]} total")

    # -- PCA dimensionality reduction (optional) ----------------------------
    pca = None
    n_pca = env["pca_components"]
    if n_pca > 0 and X_train.shape[1] > n_pca:
        n_pca = min(n_pca, X_train.shape[0] - 1)  # can't exceed n_samples - 1
        print(f"\n  Applying PCA: {X_train.shape[1]} → {n_pca} components")
        pca = PCA(n_components=n_pca, random_state=env["random_seed"])
        X_train = pca.fit_transform(X_train)   # fit on train only
        X_val   = pca.transform(X_val)
        X_test  = pca.transform(X_test)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained:.1%}")

    # -- Train --------------------------------------------------------------
    print("\n[10/10] Training models...")
    config = Config(random_seed=env["random_seed"], model_dir=run_dir)
    trainer = ModelTrainer(
        config,
        n_estimators=env["n_estimators"],
        learning_rate=env["learning_rate"],
        model_selection=env["regression_models"],
    )

    start_time = time.time()
    best_name = trainer.train_and_evaluate(
        X_train, Y_train, X_val, Y_val, target_names,
        track_learning_curves=True,
    )
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.1f}s")

    # -- Evaluate all splits ------------------------------------------------
    print("\nEvaluating & generating artifacts...")
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
            "image_cache": env["image_cache_path"] or None,
            "pca_components": n_pca if pca is not None else None,
            "feature_dimensions": {
                "tabular": int(X_tab_train.shape[1]),
                "morphological": int(X_morph_train.shape[1]) if X_morph_train is not None else 0,
                "image": image_dim,
                "total": int(X_train.shape[1]),
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
