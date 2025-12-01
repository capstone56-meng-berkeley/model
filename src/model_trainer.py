"""Ensemble model training for heat treatment prediction."""

import math
import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config, ensure_dir


def build_ensemble_models(random_seed: int = 42, n_targets: int = 1) -> Dict:
    """
    Build ensemble regressor models.

    Uses MultiOutputRegressor only when n_targets > 1 and model doesn't
    support native multi-output. RandomForest supports native multi-output.

    Args:
        random_seed: Random seed for reproducibility
        n_targets: Number of target columns

    Returns:
        Dictionary of model name -> regressor
    """
    # Random Forest (supports native multi-output)
    rf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=random_seed,
            n_jobs=-1
        ))
    ])

    # Gradient Boosting (single output only)
    gbr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", GradientBoostingRegressor(random_state=random_seed))
    ])

    # AdaBoost (single output only)
    abr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", AdaBoostRegressor(
            random_state=random_seed,
            n_estimators=400
        ))
    ])

    if n_targets == 1:
        # Single target: no wrapper needed
        return {"RF": rf, "GBR": gbr, "ABR": abr}
    else:
        # Multi-target: RF native, others need wrapper
        return {
            "RF": rf,  # Native multi-output support
            "GBR": MultiOutputRegressor(gbr),
            "ABR": MultiOutputRegressor(abr),
        }


def evaluate_model(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    target_columns: List[str],
    split_name: str = "val"
) -> Tuple[Dict, np.ndarray, Dict]:
    """
    Evaluate a model's predictions.

    Args:
        model: Trained model
        X: Feature matrix
        Y: Target array (1D or 2D)
        target_columns: Names of target columns
        split_name: Name of the data split

    Returns:
        Tuple of (overall_metrics, predictions, per_target_metrics)
    """
    Y_pred = model.predict(X)

    # Ensure 2D for consistent handling
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    # Overall metrics
    r2 = r2_score(Y, Y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(Y, Y_pred, multioutput="uniform_average")
    rmse = math.sqrt(mean_squared_error(Y, Y_pred, multioutput="uniform_average"))

    overall_metrics = {
        "split": split_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }

    # Per-target metrics
    per_target = {}
    for i, col in enumerate(target_columns):
        per_target[col] = {
            "R2": r2_score(Y[:, i], Y_pred[:, i]),
            "MAE": mean_absolute_error(Y[:, i], Y_pred[:, i]),
            "RMSE": math.sqrt(mean_squared_error(Y[:, i], Y_pred[:, i])),
        }

    return overall_metrics, Y_pred, per_target


def plot_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_columns: List[str],
    save_path: str = None
) -> None:
    """
    Plot predicted vs true values for each target.

    Args:
        Y_true: True target values (1D or 2D)
        Y_pred: Predicted values (1D or 2D)
        target_columns: Names of target columns
        save_path: Optional path to save the figure
    """
    # Ensure 2D for consistent handling
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    n_targets = len(target_columns)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))

    if n_targets == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, target_columns)):
        y_true = Y_true[:, i]
        y_pred = Y_pred[:, i]

        ax.scatter(y_true, y_pred, alpha=0.6)

        # Identity line
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], linestyle="--", color="red")

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{col}: Pred vs True")

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved plot to {save_path}")

    plt.close()


class ModelTrainer:
    """Trains and evaluates ensemble models."""

    def __init__(self, config: Config):
        """
        Initialize the model trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.models = None  # Built lazily when n_targets is known
        self.fitted_models: Dict = {}
        self.best_model_name: str = None
        self.best_model = None
        self.n_targets: int = None

        # Set random seed
        np.random.seed(config.random_seed)

    def split_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            X: Feature matrix
            Y: Target matrix
            test_size: Test set proportion
            val_size: Validation set proportion (from remaining after test)

        Returns:
            Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test)
        """
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=self.config.random_seed
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=val_size,
            random_state=self.config.random_seed
        )

        print(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        target_columns: List[str]
    ) -> str:
        """
        Train all models and select the best one.

        Args:
            X_train: Training features
            Y_train: Training targets (1D or 2D)
            X_val: Validation features
            Y_val: Validation targets (1D or 2D)
            target_columns: Names of target columns

        Returns:
            Name of the best model
        """
        # Determine n_targets and build models lazily
        self.n_targets = len(target_columns)
        self.models = build_ensemble_models(
            random_seed=self.config.random_seed,
            n_targets=self.n_targets
        )
        print(f"Building models for {self.n_targets} target(s)...")

        best_val_r2 = -np.inf

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, Y_train)

            val_metrics, _, val_per_target = evaluate_model(
                model, X_val, Y_val, target_columns, split_name="val"
            )

            print(f"Validation: R2={val_metrics['R2']:.4f}, "
                  f"MAE={val_metrics['MAE']:.4f}, RMSE={val_metrics['RMSE']:.4f}")

            for col, metrics in val_per_target.items():
                print(f"  {col}: R2={metrics['R2']:.3f}, "
                      f"MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")

            self.fitted_models[name] = model

            if val_metrics["R2"] > best_val_r2:
                best_val_r2 = val_metrics["R2"]
                self.best_model_name = name

        self.best_model = self.fitted_models[self.best_model_name]
        print(f"\n✓ Best model: {self.best_model_name} (R2={best_val_r2:.4f})")

        return self.best_model_name

    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        target_columns: List[str]
    ) -> Tuple[Dict, np.ndarray]:
        """
        Evaluate the best model on test set.

        Args:
            X_test: Test features
            Y_test: Test targets
            target_columns: Names of target columns

        Returns:
            Tuple of (test_metrics, predictions)
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_and_evaluate first.")

        test_metrics, Y_pred, per_target = evaluate_model(
            self.best_model, X_test, Y_test, target_columns, split_name="test"
        )

        print(f"\nTest Results ({self.best_model_name}):")
        print(f"  R2={test_metrics['R2']:.4f}, "
              f"MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}")

        for col, metrics in per_target.items():
            print(f"  {col}: R2={metrics['R2']:.3f}, "
                  f"MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")

        return test_metrics, Y_pred

    def save_model(self, path: str = None) -> str:
        """
        Save the best model to disk.

        Args:
            path: Optional path (defaults to config.model_dir)

        Returns:
            Path to saved model
        """
        if self.best_model is None:
            raise ValueError("No model trained yet.")

        ensure_dir(self.config.model_dir)

        if path is None:
            path = os.path.join(
                self.config.model_dir,
                f"ensemble_{self.best_model_name}.joblib"
            )

        joblib.dump(self.best_model, path)
        print(f"✓ Saved model to {path}")

        return path

    def load_model(self, path: str) -> MultiOutputRegressor:
        """
        Load a model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded model
        """
        self.best_model = joblib.load(path)
        print(f"✓ Loaded model from {path}")
        return self.best_model
