"""
predict.py
----------
Load the saved Medical Insurance Cost Prediction pipeline and generate
predictions for new customer records.

Can be used both as an importable module (e.g. from app.py) and as a
standalone script for a quick sanity check:

    python src/predict.py
"""

import os

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "medical_insurance_model.joblib")

FEATURE_ORDER = ["age", "bmi", "children", "sex", "smoker", "region"]


def load_model(model_path: str = MODEL_PATH):
    """Load the trained scikit-learn Pipeline (preprocessing + model) from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run 'python src/train.py' first."
        )
    return joblib.load(model_path)


def make_input_dataframe(age: int, sex: str, bmi: float, children: int, smoker: str, region: str) -> pd.DataFrame:
    """
    Convert raw customer inputs into the DataFrame format expected by the
    saved pipeline. Column names/order must match what the model was
    trained on; the pipeline itself handles scaling/encoding internally.
    """
    data = {
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [str(sex).lower()],
        "smoker": [str(smoker).lower()],
        "region": [str(region).lower()],
    }
    return pd.DataFrame(data)[FEATURE_ORDER]


def predict_charge(age: int, sex: str, bmi: float, children: int, smoker: str, region: str, model=None) -> float:
    """
    Predict the expected medical insurance charge for a single customer.

    Parameters
    ----------
    age : int            -- customer age in years (expected range ~18-64)
    sex : str             -- "male" or "female"
    bmi : float           -- body mass index (must be positive)
    children : int        -- number of dependents/children (>= 0)
    smoker : str           -- "yes" or "no"
    region : str          -- "southwest", "southeast", "northwest", "northeast"
    model : sklearn Pipeline, optional -- pass a pre-loaded model to avoid
            reloading it from disk on every call (recommended for Streamlit).

    Returns
    -------
    float : predicted insurance charge
    """
    if model is None:
        model = load_model()

    input_df = make_input_dataframe(age, sex, bmi, children, smoker, region)
    prediction = model.predict(input_df)[0]
    return float(prediction)


if __name__ == "__main__":
    model = load_model()

    sample_customers = [
        {"age": 19, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest"},
        {"age": 45, "sex": "male", "bmi": 28.0, "children": 2, "smoker": "no", "region": "northeast"},
        {"age": 60, "sex": "female", "bmi": 32.5, "children": 1, "smoker": "no", "region": "southeast"},
    ]

    print("Sample predictions:")
    for customer in sample_customers:
        cost = predict_charge(**customer, model=model)
        print(f"  {customer} -> Predicted charge: {cost:,.2f}")
