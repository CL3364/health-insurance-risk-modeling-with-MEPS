# Health Insurance Risk Modeling with MEPS

Predicting individual annual healthcare expenditure (`TOTEXP`) from demographic, socioeconomic, clinical, and insurance-coverage features in the US Medical Expenditure Panel Survey (MEPS). The trained ensemble ships as an interactive Streamlit app.

---

## Highlights

- **Target:** Total annual healthcare expenditure (insurance + out-of-pocket) from MEPS 2022 — median **$1,538**, mean **$6,859**, 15.1% of respondents with $0 spending.
- **Primary model:** Weighted LightGBM + XGBoost v2 ensemble on a 63-feature extended set (55 base + 8 derived).
- **Zero-inflation:** Handled with Tweedie (p=1.5) and two-stage Hurdle models as comparison points.
- **Validation design:**
  - **Train / validate:** MEPS HC-243 (2022 full-year consolidated file, 22,431 individuals)
  - **Real-world test:** MEPS HC-251 (2023 full-year consolidated file) — one-year-forward generalization check, no data leakage.
- **Streamlit web app** (`app/app.py`): interactive input form with MEPS-grounded defaults, percentile positioning, and side-by-side predictions from every trained model.

---

## Repository Structure

```
health-insurance-risk-modeling-with-MEPS/
│
├── README.md
├── .gitignore
│
├── app/
│   └── app.py                      # Streamlit web app (primary = LGB+XGB ensemble)
│
├── artifacts/                      # pipeline outputs (produced by notebooks 03–05)
│   ├── X_train.csv                 # 63-feature train matrix
│   ├── X_test.csv                  # 63-feature test matrix
│   ├── y_train.csv                 # log1p(TOTEXP) train
│   ├── y_test.csv                  # log1p(TOTEXP) test
│   ├── scaler.pkl                  # StandardScaler fit on training features (LR/SVR only)
│   └── web_app_constants.json      # age gates, imputation defaults, income transforms
│
├── data/
│   └── raw/                        # MEPS .dta files (not tracked — download separately)
│
├── docs/
│   ├── MEPS_HC_243_Documentation.pdf   # 2022 codebook
│   └── MEPS_HC_251_Documentation.pdf   # 2023 codebook (real-world test set)
│
├── models/
│   ├── primary/                    # production ensemble components
│   │   ├── lightgbm.pkl            # LightGBM on 63-feature set, log1p target
│   │   └── xgboost_v2.pkl          # XGBoost on 63-feature set, log1p target
│   ├── baseline/                   # 55-feature log-space baselines (require scaler)
│   │   ├── linear_regression.pkl
│   │   ├── svr.pkl
│   │   └── xgboost.pkl
│   ├── hurdle/                     # models built for zero-inflated spending
│   │   ├── xgboost_tweedie.pkl     # Tweedie regression (p=1.5)
│   │   ├── hurdle_classifier.pkl   # stage 1: P(spend > 0)
│   │   └── hurdle_regressor.pkl    # stage 2: E[spend | spend > 0]
│   ├── experimental/               # exploratory, not loaded by the app
│   │   ├── catboost.pkl
│   │   ├── lightgbm_opt.pkl        # Optuna-tuned LGB variant
│   │   ├── xgboost_opt.pkl         # Optuna-tuned XGB variant
│   │   └── stacking_meta.pkl       # stacking meta-learner
│   ├── figures/                    # feature importance plots
│   │   ├── feature_importance.png
│   │   └── feature_importance_lgb.png
│   └── meta/                       # configs + held-out results
│       ├── model_meta.json         # feature_cols (55) and feature_cols_extended (63), ensemble weights
│       ├── model_results.json      # test-set R² / RMSE / MAE for every model
│       ├── percentile_lookup.json  # MEPS spending percentiles (for app UI)
│       └── smearing_factors.json   # Duan smearing factors for log-space models
│
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_eda.ipynb
    ├── 03_feature_engineering.ipynb    # builds the 55 base features; writes artifacts/*.csv, scaler.pkl
    ├── 04_modeling.ipynb                # baseline + hurdle + Tweedie (log-space, 55 feat)
    ├── 05_improved_modeling.ipynb       # LightGBM, XGBoost v2, ensemble (63 feat) ★ best
    └── 06_advanced_modeling.ipynb       # CatBoost, Optuna tuning, stacking (experimental)
```

---

## Running the Web App

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn
streamlit run app/app.py
```

The app loads the ensemble and all comparison models from `models/`, reads the fitted scaler from `artifacts/scaler.pkl`, and renders an input form that mirrors the MEPS questionnaire. The headline prediction is the **LGB+XGB ensemble**.

---

## Notebook Pipeline

The notebooks are numbered to reflect the data flow. Each one consumes outputs from its predecessor.

| # | Notebook | What it does | Key outputs |
|---|---|---|---|
| 01 | `01_eda.ipynb` | Target distribution, missing-data patterns | — |
| 02 | `02_eda.ipynb` | Variable relationships, correlations | — |
| 03 | `03_feature_engineering.ipynb` | Age-gating, transformations (signed log1p on income, log1p on target), StandardScaler | `artifacts/X_{train,test}.csv`, `artifacts/y_{train,test}.csv`, `artifacts/scaler.pkl`, `artifacts/web_app_constants.json` |
| 04 | `04_modeling.ipynb` | LR, SVR, RF, XGBoost (log-space + Duan smearing), Tweedie, two-stage Hurdle — **55 features** | `models/baseline/*.pkl`, `models/hurdle/*.pkl`, `models/meta/model_meta.json`, `models/meta/smearing_factors.json`, `models/meta/percentile_lookup.json` |
| 05 | `05_improved_modeling.ipynb` | Adds 8 derived features, trains LightGBM + XGBoost v2, builds weighted ensemble on held-out validation — **63 features** ★ | `models/primary/*.pkl`, updates `models/meta/model_meta.json`, `models/figures/feature_importance_lgb.png` |
| 06 | `06_advanced_modeling.ipynb` | CatBoost, Optuna hyperparameter tuning, stacking meta-learner — exploratory | `models/experimental/*.pkl`, updates `models/meta/model_results.json` |

Run sequentially `03 → 04 → 05`. Notebook `06` is independent and not required for the app.

---

## Feature Engineering

**Base set (55 features):** demographics, socioeconomic, clinical diagnoses (12 `dx_` flags), insurance coverage, functional limitation, mental-health screens (K6 / PHQ-2), race/marital/region one-hots.

**Age-gating** (reflects MEPS survey design — some features only defined for certain ages):
- K6 / PHQ-2 distress scores: **adults 18+**
- Student status: **ages 17–23**
- Adult chronic conditions (hypertension, high cholesterol, arthritis, cancer, CHD, angina, MI, stroke, emphysema): **18+**
- ADHD: **ages 5–17**

**Derived set (+8 features = 63 total):** added in `05_improved_modeling.ipynb`, also recomputed at app inference time in `app/app.py:add_derived_features()`:

| Feature | Definition | Rationale |
|---|---|---|
| `num_conditions` | Count of 12 `dx_` flags | Comorbidity burden |
| `is_elderly` | `age >= 65` | Medicare-eligibility discontinuity |
| `bmi_obese` | `BMI >= 30` | Obesity-related spend |
| `dual_eligible` | Medicare AND Medicaid | 2–3× per-capita spending among duals |
| `cardiac_burden` | CHD + angina + MI + stroke | Cardiac disease density |
| `age_sq` | `age² / 1000` | Non-linear aging effect |
| `age_x_conditions` | `age × num_conditions / 10` | Interaction term |
| `health_burden` | `self_rated_health × num_conditions / 10` | Self-report × objective count |

**Target transformation:** `TOTEXP` is log1p-transformed before model fitting. Legacy (55-feat) models use **Duan's smearing factors** (stored in `models/meta/smearing_factors.json`) to correct retransformation bias. Primary (63-feat) models use direct `expm1` — smearing is not needed once the feature set richer-fits the mean.

---

## Why Two Feature Sets?

The 55-feature baseline is a fair comparison to prior MEPS modeling literature, most of which uses the standard set of questionnaire fields. The 8 derived features in the 63-feature set are all computable from the 55-feature inputs — no extra user input — but they give the tree models the non-linearities and interactions they otherwise have to learn from scratch. Holding the input set constant, the extended set lifts ensemble test R² noticeably without inflating complexity.

---

## Model Limitations

Even the best model here captures roughly **half** the variance in annual spending. Healthcare spending is intrinsically hard to predict from cross-sectional survey data because the strongest predictors — **prior-year spending, count of active prescriptions, planned procedures, visit frequency** — are **not available** in a single-year MEPS snapshot. Use the predictions as a rough estimate, not a precise forecast.

---

## Data

MEPS is published by the Agency for Healthcare Research and Quality (AHRQ) and is publicly downloadable. Place the `.dta` files under `data/raw/`:

- [HC-243 (2022 Full-Year Consolidated)](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-243)
- [HC-251 (2023 Full-Year Consolidated)](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-251)

See `docs/MEPS_HC_243_Documentation.pdf` and `docs/MEPS_HC_251_Documentation.pdf` for the codebooks.

---

## Stack

Python 3 · pandas · numpy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · Streamlit · joblib · matplotlib · seaborn

---

## Author

**Caleb Lee**
