"""
train.py
--------
End-to-end training script for the Medical Insurance Cost Prediction project.

Pipeline:
    1. Load data
    2. Clean data (duplicates, dtypes, missing values)
    3. Build a preprocessing + model Pipeline (ColumnTransformer)
    4. Train / compare multiple regression models
    5. Hyperparameter-tune the best model
    6. Evaluate on the held-out test set
    7. Save the final fitted pipeline with Joblib

Run:
    python src/train.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "medical_insurance_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw insurance dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------- #
# 2. Clean data
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    - Removes exact duplicate rows (documented, not silently dropped).
    - Confirms/casts dtypes for numeric and categorical columns.
    - No missing-value imputation is performed because the raw dataset
      contains no missing values (verified in EDA / notebook).
    """
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_removed = n_before - n_after
    print(f"Duplicate rows removed: {n_removed} (before={n_before}, after={n_after})")

    # Enforce expected dtypes
    df["age"] = df["age"].astype(int)
    df["bmi"] = df["bmi"].astype(float)
    df["children"] = df["children"].astype(int)
    df["charges"] = df["charges"].astype(float)
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str)

    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. Preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Scales numeric features with StandardScaler (helps Linear/Ridge
        Regression; harmless for tree-based models).
      - One-Hot Encodes categorical features (sex, smoker, region).
    Wrapping this inside the model Pipeline means the same transformer is
    fitted ONLY on the training data and re-used for every prediction,
    so raw user input can be passed straight in from Streamlit.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


# --------------------------------------------------------------------------- #
# 4. Train & compare models
# --------------------------------------------------------------------------- #
def get_candidate_models() -> dict:
    """Return a dictionary of candidate regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate_model(y_true, y_pred) -> dict:
    """Compute MAE, MSE, RMSE and R2 for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_models(X_train, y_train, X_test, y_test, preprocessor) -> tuple:
    """
    Train every candidate model inside a Pipeline(preprocessor + model),
    evaluate it on the test set, and return:
      - a results dataframe (model comparison table)
      - a dict of {model_name: fitted_pipeline}
    """
    results = []
    fitted_pipelines = {}

    for name, model in get_candidate_models().items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        metrics["Model"] = name
        results.append(metrics)
        fitted_pipelines[name] = pipe
        print(f"{name:30s} | MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)[["Model", "MAE", "MSE", "RMSE", "R2"]]
    results_df = results_df.sort_values("RMSE").reset_index(drop=True)
    return results_df, fitted_pipelines


# --------------------------------------------------------------------------- #
# 5. Hyperparameter tuning
# --------------------------------------------------------------------------- #
def tune_model(X_train, y_train, preprocessor, best_model_name: str):
    """
    Run GridSearchCV on the best-performing tree-based model
    (Random Forest or Gradient Boosting) with 5-fold cross-validation.
    """
    if best_model_name == "Random Forest Regressor":
        base_model = RandomForestRegressor(random_state=RANDOM_STATE)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 4],
        }
    else:
        base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        }

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV RMSE:", -grid_search.best_score_)

    return grid_search


# --------------------------------------------------------------------------- #
# 6. Save model
# --------------------------------------------------------------------------- #
def save_model(model, metadata: dict, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH):
    """Persist the fitted pipeline and its metadata to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)
    df = load_data()
    print(f"Raw shape: {df.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Cleaning data")
    print("=" * 70)
    df = clean_data(df)
    print(f"Clean shape: {df.shape}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    print("\n" + "=" * 70)
    print("STEP 3: Train-test split (80/20, random_state=42)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")

    print("\n" + "=" * 70)
    print("STEP 4: Building preprocessing pipeline")
    print("=" * 70)
    preprocessor = build_preprocessor()
    print("Numeric features (StandardScaler):", NUMERIC_FEATURES)
    print("Categorical features (OneHotEncoder):", CATEGORICAL_FEATURES)

    print("\n" + "=" * 70)
    print("STEP 5: Training & comparing baseline models")
    print("=" * 70)
    results_df, fitted_pipelines = train_models(X_train, y_train, X_test, y_test, preprocessor)
    print("\nModel comparison table:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest baseline model (lowest RMSE): {best_model_name}")

    print("\n" + "=" * 70)
    print(f"STEP 6: Hyperparameter tuning ({best_model_name})")
    print("=" * 70)
    # Tuning is only implemented for the two tree ensembles. If a linear
    # model happens to win, fall back to tuning Gradient Boosting since it
    # is the strongest general-purpose candidate for this dataset.
    tune_target = best_model_name if best_model_name in (
        "Random Forest Regressor", "Gradient Boosting Regressor"
    ) else "Gradient Boosting Regressor"

    grid_search = tune_model(X_train, y_train, preprocessor, tune_target)
    tuned_model = grid_search.best_estimator_

    print("\n" + "=" * 70)
    print("STEP 7: Final evaluation on test set")
    print("=" * 70)
    tuned_preds = tuned_model.predict(X_test)
    tuned_metrics = evaluate_model(y_test, tuned_preds)
    print(f"Tuned {tune_target}: {tuned_metrics}")

    baseline_metrics = results_df[results_df["Model"] == tune_target].iloc[0].to_dict()
    print(f"Baseline {tune_target}: {baseline_metrics}")

    # Decide the final model: keep the tuned model only if it is at least
    # as good as the baseline on RMSE (tuning should never make things worse).
    if tuned_metrics["RMSE"] <= baseline_metrics["RMSE"]:
        final_model = tuned_model
        final_metrics = tuned_metrics
        final_model_label = f"{tune_target} (tuned)"
    else:
        final_model = fitted_pipelines[tune_target]
        final_metrics = baseline_metrics
        final_model_label = f"{tune_target} (baseline)"

    print(f"\nFinal selected model: {final_model_label}")
    print(f"Final metrics: {final_metrics}")

    print("\n" + "=" * 70)
    print("STEP 8: Saving final model")
    print("=" * 70)
    metadata = {
        "final_model": final_model_label,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "best_params": grid_search.best_params_,
        "cv_best_rmse": float(-grid_search.best_score_),
        "test_metrics": {k: float(v) for k, v in final_metrics.items() if k in ("MAE", "MSE", "RMSE", "R2")},
        "model_comparison": results_df.to_dict(orient="records"),
    }
    save_model(final_model, metadata)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
