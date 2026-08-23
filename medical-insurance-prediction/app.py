"""
app.py
------
Streamlit application: Medical Insurance Cost Prediction.

Run locally with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import load_model, predict_charge  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "medical_insurance_model.joblib")

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


@st.cache_resource
def get_model():
    """Load and cache the trained pipeline so it isn't reloaded on every interaction."""
    return load_model(MODEL_PATH)


def main():
    st.title("🏥 Medical Insurance Cost Prediction")
    st.write(
        "Estimate the expected medical insurance charges for a customer based on "
        "age, sex, BMI, number of children, smoking status, and region. This app "
        "uses a Machine Learning regression model trained on the classic Medical "
        "Cost Personal Dataset."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model was found. Please run `python src/train.py` first "
            "to train and save the model before launching this app."
        )
        st.stop()

    model = get_model()

    st.header("Enter Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"])
        smoker = st.selectbox("Smoker", options=["No", "Yes"])
        region = st.selectbox("Region", options=["Southwest", "Southeast", "Northwest", "Northeast"])

    st.markdown("---")

    if st.button("Predict Insurance Cost", type="primary"):
        # Basic input validation (in addition to widget min/max constraints)
        errors = []
        if age < 18 or age > 100:
            errors.append("Age must be a sensible value (18-100).")
        if bmi <= 0:
            errors.append("BMI must be a positive number.")
        if children < 0:
            errors.append("Number of children cannot be negative.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            predicted_cost = predict_charge(
                age=age,
                sex=sex,
                bmi=bmi,
                children=children,
                smoker=smoker,
                region=region,
                model=model,
            )

            st.subheader("Estimated Medical Insurance Cost")
            st.markdown(f"## ₹ {predicted_cost:,.2f}")
            st.caption(
                "This estimate is generated in Indian Rupees (₹) for display purposes; "
                "the underlying model was trained on USD-denominated charges."
            )

    st.markdown("---")

    with st.expander("ℹ️ Model Information"):
        st.write(
            """
            - **Problem type:** Regression
            - **Target variable:** `charges` (medical insurance cost)
            - **Input features:** age, sex, bmi, children, smoker, region
            - **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical),
              wrapped in a scikit-learn Pipeline together with the model, so raw
              inputs from this form are transformed automatically.
            - See `models/model_metadata.json` and the project README for the
              exact model type and evaluation metrics (MAE, RMSE, R²).
            """
        )

    st.info(
        "⚠️ **Disclaimer:** This prediction is for educational/demo purposes only "
        "and should not be considered an actual insurance quotation or medical advice."
    )


if __name__ == "__main__":
    main()
