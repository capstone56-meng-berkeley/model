"""Ensemble model training for heat treatment prediction."""

import math
import os

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


def build_ensemble_models(
    random_seed: int = 42,
    n_targets: int = 1,
    n_estimators: int = 400,
    learning_rate: float = 0.1,
    model_selection: list[str] = None
) -> dict:
    """
    Build ensemble regressor models.

    Uses MultiOutputRegressor only when n_targets > 1 and model doesn't
    support native multi-output. RandomForest supports native multi-output.

    Args:
        random_seed: Random seed for reproducibility
        n_targets: Number of target columns
        n_estimators: Number of trees/iterations for ensemble models
        learning_rate: Learning rate for GradientBoostingRegressor
        model_selection: List of model names to build (default: all)

    Returns:
        Dictionary of model name -> regressor
    """
    all_models = {}

    # Random Forest (supports native multi-output).
    # max_features="sqrt" and min_samples_leaf=3 reduce variance on small datasets.
    rf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=random_seed,
            n_jobs=-1
        ))
    ])
    all_models["RF"] = rf

    # Gradient Boosting (single output only).
    # max_depth=3, subsample=0.8, and min_samples_leaf=5 prevent memorisation
    # of the small training set.
    gbr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=random_seed
        ))
    ])

    # AdaBoost (single output only).
    # Shallow base estimator (max_depth=3) limits individual tree complexity.
    from sklearn.tree import DecisionTreeRegressor
    abr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=n_estimators,
            random_state=random_seed
        ))
    ])

    if n_targets == 1:
        all_models["GBR"] = gbr
        all_models["ABR"] = abr
    else:
        all_models["GBR"] = MultiOutputRegressor(gbr)
        all_models["ABR"] = MultiOutputRegressor(abr)

    # Filter to selected models
    if model_selection:
        all_models = {k: v for k, v in all_models.items() if k in model_selection}

    return all_models


def evaluate_model(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    target_columns: list[str],
    split_name: str = "val"
) -> tuple[dict, np.ndarray, dict]:
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


def compute_staged_metrics(
    model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray
) -> dict[str, list[float]]:
    """
    Compute metrics at each iteration for boosting models using staged_predict.

    Only works for GradientBoostingRegressor and AdaBoostRegressor (single target).

    Args:
        model: Fitted Pipeline with boosting regressor
        X_train: Training features
        Y_train: Training targets (1D)
        X_val: Validation features
        Y_val: Validation targets (1D)

    Returns:
        Dictionary with lists of metrics per iteration:
        {'train_r2': [...], 'val_r2': [...], 'train_rmse': [...], 'val_rmse': [...]}
    """
    # Get the regressor from the pipeline
    scaler = model.named_steps['scaler']
    reg = model.named_steps['reg']

    # Scale the data
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    history = {
        'train_r2': [],
        'val_r2': [],
        'train_rmse': [],
        'val_rmse': [],
        'iteration': []
    }

    # Use staged_predict to get predictions at each iteration
    train_staged = list(reg.staged_predict(X_train_scaled))
    val_staged = list(reg.staged_predict(X_val_scaled))

    for i, (y_train_pred, y_val_pred) in enumerate(zip(train_staged, val_staged, strict=False)):
        history['iteration'].append(i + 1)
        history['train_r2'].append(r2_score(Y_train, y_train_pred))
        history['val_r2'].append(r2_score(Y_val, y_val_pred))
        history['train_rmse'].append(math.sqrt(mean_squared_error(Y_train, y_train_pred)))
        history['val_rmse'].append(math.sqrt(mean_squared_error(Y_val, y_val_pred)))

    return history


def plot_learning_curves(
    history: dict[str, list[float]],
    model_name: str,
    save_path: str = None,
    show: bool = True
):
    """
    Plot learning curves showing metrics across iterations.

    Args:
        history: Dictionary with train/val metrics per iteration
        model_name: Name of the model for the title
        save_path: Optional path to save the figure
        show: Whether to display the plot (True for notebooks)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    iterations = history['iteration']

    # R2 plot
    axes[0].plot(iterations, history['train_r2'], label='Train', alpha=0.8)
    axes[0].plot(iterations, history['val_r2'], label='Validation', alpha=0.8)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('R2 Score')
    axes[0].set_title(f'{model_name}: R2 vs Iterations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMSE plot
    axes[1].plot(iterations, history['train_rmse'], label='Train', alpha=0.8)
    axes[1].plot(iterations, history['val_rmse'], label='Validation', alpha=0.8)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title(f'{model_name}: RMSE vs Iterations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved learning curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    save_path: str = None,
    show: bool = True
):
    """
    Plot bar chart comparing metrics across models.

    Args:
        results: Dictionary of {model_name: {metric_name: value}}
        save_path: Optional path to save the figure
        show: Whether to display the plot (True for notebooks)

    Returns:
        matplotlib Figure object
    """
    model_names = list(results.keys())
    metrics = ['R2', 'MAE', 'RMSE']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red

    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        bars = axes[idx].bar(model_names, values, color=colors[idx], alpha=0.8)
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} by Model')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=False):
            height = bar.get_height()
            axes[idx].annotate(f'{val:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)

    plt.suptitle('Model Comparison on Test Set', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved comparison plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_columns: list[str],
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

    for i, (ax, col) in enumerate(zip(axes, target_columns, strict=False)):
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

    def __init__(
        self,
        config: Config,
        n_estimators: int = 400,
        learning_rate: float = 0.1,
        model_selection: list[str] = None
    ):
        """
        Initialize the model trainer.

        Args:
            config: Configuration object
            n_estimators: Number of trees/iterations for ensemble models
            learning_rate: Learning rate for GradientBoostingRegressor
            model_selection: List of model names to train (default: all)
        """
        self.config = config
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_selection = model_selection
        self.models = None  # Built lazily when n_targets is known
        self.fitted_models: dict = {}
        self.best_model_name: str = None
        self.best_model = None
        self.n_targets: int = None
        self.learning_histories: dict[str, dict] = {}  # Learning curves for boosting models

        # Set random seed
        np.random.seed(config.random_seed)

    def split_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
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
        target_columns: list[str],
        track_learning_curves: bool = True
    ) -> str:
        """
        Train all models and select the best one.

        Args:
            X_train: Training features
            Y_train: Training targets (1D or 2D)
            X_val: Validation features
            Y_val: Validation targets (1D or 2D)
            target_columns: Names of target columns
            track_learning_curves: Whether to track learning curves for boosting models

        Returns:
            Name of the best model
        """
        # Determine n_targets and build models lazily
        self.n_targets = len(target_columns)
        self.models = build_ensemble_models(
            random_seed=self.config.random_seed,
            n_targets=self.n_targets,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            model_selection=self.model_selection
        )
        print(f"Building models for {self.n_targets} target(s) with {self.n_estimators} estimators...")

        best_val_r2 = -np.inf
        self.learning_histories = {}

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

            # Track learning curves for boosting models (single target only)
            if track_learning_curves and name in ("GBR", "ABR") and self.n_targets == 1:
                print(f"  Computing learning curves for {name}...")
                Y_train_1d = Y_train.ravel() if Y_train.ndim > 1 else Y_train
                Y_val_1d = Y_val.ravel() if Y_val.ndim > 1 else Y_val
                self.learning_histories[name] = compute_staged_metrics(
                    model, X_train, Y_train_1d, X_val, Y_val_1d
                )

            if val_metrics["R2"] > best_val_r2:
                best_val_r2 = val_metrics["R2"]
                self.best_model_name = name

        self.best_model = self.fitted_models[self.best_model_name]
        print(f"\n✓ Best model: {self.best_model_name} (R2={best_val_r2:.4f})")

        return self.best_model_name

    def plot_learning_curves(self, model_name: str = None, save_dir: str = None, show: bool = True) -> None:
        """
        Plot learning curves for boosting models.

        Args:
            model_name: Specific model to plot (default: all available)
            save_dir: Directory to save plots (default: don't save)
            show: Whether to display the plot (default: True for notebook use)
        """
        if not self.learning_histories:
            print("No learning histories available. Run train_and_evaluate with track_learning_curves=True")
            return

        models_to_plot = [model_name] if model_name else list(self.learning_histories.keys())

        for name in models_to_plot:
            if name not in self.learning_histories:
                print(f"No learning history for {name}")
                continue

            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{name}_learning_curves.png")

            plot_learning_curves(self.learning_histories[name], name, save_path, show=show)

    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        target_columns: list[str]
    ) -> tuple[dict, np.ndarray]:
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

    def evaluate_all_on_test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        target_columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate ALL fitted models on test set.

        Args:
            X_test: Test features
            Y_test: Test targets
            target_columns: Names of target columns

        Returns:
            Dictionary of {model_name: {metric_name: value}}
        """
        if not self.fitted_models:
            raise ValueError("No models trained yet. Call train_and_evaluate first.")

        self.test_results = {}

        print("\n" + "="*50)
        print("Test Set Evaluation - All Models")
        print("="*50)

        for name, model in self.fitted_models.items():
            test_metrics, Y_pred, per_target = evaluate_model(
                model, X_test, Y_test, target_columns, split_name="test"
            )

            self.test_results[name] = test_metrics

            print(f"\n{name}:")
            print(f"  R2={test_metrics['R2']:.4f}, "
                  f"MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}")

            for col, metrics in per_target.items():
                print(f"    {col}: R2={metrics['R2']:.3f}, "
                      f"MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")

        # Highlight best model
        best_name = max(self.test_results, key=lambda k: self.test_results[k]['R2'])
        print(f"\n✓ Best on test: {best_name} (R2={self.test_results[best_name]['R2']:.4f})")

        return self.test_results

    def plot_test_comparison(self, save_dir: str = None, show: bool = True) -> None:
        """
        Plot comparison of all models on test set.

        Args:
            save_dir: Directory to save the plot
            show: Whether to display the plot (True for notebooks)
        """
        if not hasattr(self, 'test_results') or not self.test_results:
            print("No test results available. Run evaluate_all_on_test first.")
            return

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, "model_comparison.png")

        plot_model_comparison(self.test_results, save_path, show=show)

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
