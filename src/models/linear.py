"""
Linear Regression Training Script for Student Test Scores with Optuna

This script trains a Linear Regression model with hyperparameter tuning
using Optuna and tracks experiments using MLflow.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import cross_val_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.mlflow_config import get_artifact_location, setup_mlflow
from preprocessing.preprocess import Preprocess


def load_processed_data(data_dir: str = "../../data/processed"):
    """Load preprocessed training and test data."""
    data_path = Path(data_dir)

    X_train = pd.read_parquet(data_path / "X_train.parquet")
    y_train = pd.read_parquet(data_path / "y_train.parquet").squeeze()
    X_test = pd.read_parquet(data_path / "X_test.parquet")
    y_test = pd.read_parquet(data_path / "y_test.parquet").squeeze()

    print(f"✓ Data loaded:")
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return metrics, y_pred


def objective(trial, X_train, y_train):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target

    Returns:
        Mean cross-validation score
    """
    params = {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "positive": trial.suggest_categorical("positive", [True, False]),
    }

    model = LinearRegression(**params)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    return cv_scores.mean()


def optimize_with_optuna(X_train, y_train, n_trials=100):
    """
    Optimize hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials

    Returns:
        Optuna study object
    """
    print(f"\n🔍 Starting Optuna optimization with {n_trials} trials...")

    study = optuna.create_study(
        direction="maximize",
        study_name="linear_regression_student_scores",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n✓ Optimization completed!")
    print(f"✓ Best trial: {study.best_trial.number}")
    print(f"✓ Best CV score: {study.best_value:.4f}")
    print(f"✓ Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study


def train_best_model(best_params, X_train, y_train, X_test, y_test):
    """Train final model with best parameters."""

    model = LinearRegression(**best_params)

    print("\n🚀 Training final model with best parameters...")
    model.fit(X_train, y_train)

    metrics, y_pred = evaluate_model(model, X_test, y_test)

    print(f"\n📊 Test Set Performance:")
    print(f"  - MSE:  {metrics['mse']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.4f}")
    print(f"  - MAE:  {metrics['mae']:.4f}")
    print(f"  - R²:   {metrics['r2']:.4f}")

    return model, metrics


def generate_submission(model, output_path="submission.csv"):
    """Generate submission file for Kaggle competition."""
    print("\n📝 Generating submission file...")

    # Load raw test data
    test_df = pd.read_csv("../../data/raw/playground-series-s6e1/test.csv")
    test_ids = test_df["id"].copy()

    # Load training data to fit the scaler
    train_df = pd.read_csv("../../data/raw/playground-series-s6e1/train.csv")

    # Preprocess training data to fit scaler
    train_prep = Preprocess(train_df.copy())
    train_prep.preprocess()
    X_train_full = train_prep.data.drop(columns=["exam_score", "id"])
    train_prep.scaler.fit(X_train_full)

    # Preprocess test data
    test_prep = Preprocess(test_df.copy())
    test_prep.preprocess()
    X_test_submission = test_prep.data.drop(columns=["id"])

    # Apply same scaling
    X_test_submission = pd.DataFrame(
        train_prep.scaler.transform(X_test_submission),
        columns=X_test_submission.columns,
    )

    # Make predictions
    predictions = model.predict(X_test_submission)

    # Create submission file
    submission = pd.DataFrame({"id": test_ids, "exam_score": predictions})

    submission.to_csv(output_path, index=False)
    print(f"✓ Submission file saved to {output_path}")
    print(f"  - Shape: {submission.shape}")
    print(f"  - Sample predictions: {predictions[:5]}")

    return submission


def train_with_mlflow(X_train, y_train, X_test, y_test, n_trials=100):
    """Train model with Optuna and MLflow tracking."""

    # Configure MLflow for PostgreSQL + MinIO
    setup_mlflow()

    # Set experiment with S3 artifact location
    experiment_name = "Student Scores - Linear Regression Optuna"
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=get_artifact_location(),
        )
        experiment = mlflow.get_experiment(experiment_id)
    except Exception:
        # Experiment already exists
        experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="LinearRegression_Optuna") as run:
        study = optimize_with_optuna(X_train, y_train, n_trials=n_trials)

        best_model, metrics = train_best_model(
            study.best_params, X_train, y_train, X_test, y_test
        )

        mlflow.log_params(study.best_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("best_trial_number", study.best_trial.number)

        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_cv_score", study.best_value)

        trials_df = study.trials_dataframe()
        trials_df.to_csv("optuna_trials.csv", index=False)
        mlflow.log_artifact("optuna_trials.csv")

        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name="LinearRegression_StudentScores_Optuna",
            input_example=X_train.iloc[:5],
            signature=mlflow.models.signature.infer_signature(X_train, y_train),
        )

        # Generate submission file
        generate_submission(best_model, "linear_regression_submission.csv")
        mlflow.log_artifact("linear_regression_submission.csv")

        try:
            import matplotlib.pyplot as plt

            fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig("optimization_history.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("optimization_history.png")
            plt.close()

            fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig("param_importances.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact("param_importances.png")
            plt.close()

            print("\n✓ Optimization plots saved")

        except Exception as e:
            print(f"\n⚠ Could not generate plots: {e}")

        print(f"\n✓ MLflow run completed: {run.info.run_id}")
        print(f"✓ Model logged and registered")

        return best_model, study, metrics


def main(n_trials=100):
    """
    Main execution function.

    Args:
        n_trials: Number of Optuna trials for optimization
    """

    print("=" * 60)
    print("STUDENT TEST SCORES PREDICTION - LINEAR REGRESSION (OPTUNA)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_processed_data()

    model, study, metrics = train_with_mlflow(
        X_train, y_train, X_test, y_test, n_trials=n_trials
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return model, study


if __name__ == "__main__":
    model, study = main(n_trials=4)
