# Medical Insurance Cost Prediction using Machine Learning

## Overview

A complete, end-to-end Machine Learning project that predicts **medical
insurance charges** for a customer from demographic and health-related
attributes (age, sex, BMI, number of children, smoking status, region). The
project covers the full workflow — data understanding, EDA, cleaning,
preprocessing, model comparison, hyperparameter tuning, evaluation, model
saving, and a Streamlit web app for interactive predictions.

## Problem Statement

Insurance companies need to estimate the expected medical insurance cost for
an individual based on factors such as age, gender, BMI, number of children,
smoking status, and region. The goal is to build a regression model that
predicts expected insurance charges for a new customer.

This is a **regression problem** because the target variable, `charges`, is a
continuous numerical value.

## Business Objective

- Predict medical insurance costs accurately.
- Help insurance companies estimate expected healthcare expenses.
- Support better pricing and risk assessment.
- Provide a simple UI for entering customer information and obtaining an
  estimated insurance cost.
- Demonstrate an end-to-end ML workflow from data collection to deployment.

## Dataset

| | |
|---|---|
| **Name** | Medical Cost Personal Dataset (`insurance.csv`) |
| **Common source** | [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (originally compiled for *Machine Learning with R* by Brett Lantz) |
| **Type** | Publicly available benchmark dataset commonly used for regression practice; not an official real-world insurer billing feed |
| **Rows** | 1,338 (1,337 after removing 1 exact duplicate row) |
| **Columns** | 7 (6 features + 1 target) |
| **Features** | `age`, `sex`, `bmi`, `children`, `smoker`, `region` |
| **Target** | `charges` |
| **License** | Distributed for educational/practice use on Kaggle — check the Kaggle dataset page before any commercial use |

All dataset statistics above and throughout this project were computed from
the actual file in `data/insurance.csv`, not fabricated.

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter

## Machine Learning Workflow

```
Data Collection
      ↓
Data Understanding
      ↓
EDA
      ↓
Data Cleaning
      ↓
Preprocessing
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Model Saving
      ↓
Streamlit Application
      ↓
Deployment
```

## Model Comparison

Four regression models were trained inside identical preprocessing pipelines
and evaluated on a held-out 20% test set:

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Gradient Boosting Regressor** | **2,517.47** | **18,218,239.92** | **4,268.28** | **0.9009** |
| Random Forest Regressor | 2,663.33 | 22,372,580.00 | 4,729.97 | 0.8782 |
| Linear Regression | 4,177.05 | 35,478,020.00 | 5,956.34 | 0.8069 |
| Ridge Regression | 4,193.89 | 35,663,130.00 | 5,971.86 | 0.8059 |

**Best baseline model:** Gradient Boosting Regressor.

## Hyperparameter Tuning

`GridSearchCV` (5-fold cross-validation, scoring = `neg_root_mean_squared_error`)
was run on the Gradient Boosting Regressor.

- **Best parameters:** `learning_rate=0.05`, `max_depth=2`, `n_estimators=200`
- **Best CV RMSE:** 4,624.95

On the held-out test set, the tuned model (RMSE ≈ 4,299.28) did not beat the
untuned baseline (RMSE ≈ 4,268.28), so `train.py` automatically keeps
whichever of the two actually performs better on the test set. **The final
saved model is the baseline Gradient Boosting Regressor.**

## Final Model Performance (on test set)

| Metric | Value |
|---|---|
| **Model** | Gradient Boosting Regressor |
| **MAE** | 2,517.47 |
| **MSE** | 18,218,239.92 |
| **RMSE** | 4,268.28 |
| **R²** | 0.9009 |

In plain terms: the model's predictions are, on average, off by about
**$2,517** (MAE), and the model explains about **90%** of the variance in
insurance charges (R²).

## Key EDA Insight

`smoker` status is by far the strongest driver of insurance charges
(correlation ≈ 0.79 with `charges` when label-encoded), followed by `age`
(≈ 0.30) and `bmi` (≈ 0.20). `children`, `region`, and `sex` show weak
correlation with charges individually. See the notebook for full EDA,
visualizations, and per-chart insights.

## How to Run Locally

```bash
git clone <repository-url>
cd medical-insurance-prediction

pip install -r requirements.txt

# Train the model (regenerates models/medical_insurance_model.joblib)
python src/train.py

# Launch the Streamlit app
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

### Quick prediction check

```bash
python src/predict.py
```

This loads the saved model and prints predictions for 3 sample customers.

## Streamlit Deployment (Streamlit Community Cloud)

> Deployment credentials/account access were not available in this
> environment, so the app has **not** been deployed live. Follow these exact
> steps to deploy it yourself:

1. Push this project to a public (or Community-Cloud-accessible) GitHub
   repository (see "GitHub Upload Instructions" below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
3. Click **"New app"**, then select:
   - **Repository:** `<your-username>/medical-insurance-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**. Streamlit Community Cloud will install packages from
   `requirements.txt` and start the app automatically.
5. Once live, the trained model is loaded from `models/medical_insurance_model.joblib`,
   which is committed to the repo — no separate training step is needed on
   the cloud. If you change the training data, re-run `python src/train.py`
   locally and commit the updated `.joblib` file before redeploying.

## GitHub Upload Instructions

```bash
cd medical-insurance-prediction
git init
git add .
git commit -m "Initial commit: Medical Insurance Cost Prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/medical-insurance-prediction.git
git push -u origin main
```

## Project Structure

```
medical-insurance-prediction/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   └── medical_insurance_prediction.ipynb
│
├── models/
│   ├── medical_insurance_model.joblib
│   └── model_metadata.json
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── plots/                     # EDA / evaluation charts exported as PNGs
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Limitations

- The dataset is a single public benchmark of ~1,300 rows — modest in size
  and may not represent every population or region.
- Only 6 features are available; real-world underwriting typically also uses
  pre-existing conditions, detailed medical history, occupation, and
  insurer-specific pricing rules, none of which are present here.
- Predicted values carry uncertainty (RMSE ≈ $4,268 on the test set) and
  should **not** be treated as an actual insurance quotation or medical
  advice.
- The Streamlit app displays predictions with a ₹ symbol for presentation
  purposes only; the underlying model was trained on USD-denominated charges
  and no currency conversion is applied — the numeric value is not an actual
  INR estimate.

## Future Improvements

- Train on larger, more diverse, and more recent datasets.
- Add features such as pre-existing conditions, occupation, or income.
- Try additional/advanced models (e.g. XGBoost, LightGBM, CatBoost, stacking).
- Add explainability (e.g. SHAP values) so predictions can be broken down
  per feature.
- Add proper uncertainty estimates (e.g. prediction intervals) instead of a
  single point estimate.
- Add model monitoring / drift detection if deployed against live data.

## Disclaimer

This project and its predictions are for **educational and portfolio
demonstration purposes only**. They do not constitute an actual insurance
quotation, financial advice, or medical advice.
