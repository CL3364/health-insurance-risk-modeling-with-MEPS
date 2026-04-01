# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Healthcare spending prediction using MEPS (Medical Expenditure Panel Survey) public-use data. Predicts total annual healthcare expenditures (`TOTEXP`) for individuals.

- **Train/validate:** HC-243 (2022 MEPS data, `notebooks/h243.dta` — not tracked)
- **Real-world test:** HC-251 (2023 MEPS data, `data/raw/h251.dta` — not tracked)

## Running the Web App

```bash
streamlit run app/app.py
```

The app loads all trained models from `models/` and a scaler, then provides an interactive form for healthcare spending prediction. The primary prediction is the **Ensemble (LGB+XGB)** model.

## Notebook Workflow

The notebooks follow a numbered sequence:

1. `01_eda.ipynb` / `02_eda.ipynb` — exploratory data analysis
2. `03_feature_engineering.ipynb` — feature creation, transformations, age-gating; saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `scaler.pkl`
3. `04_modeling.ipynb` — baseline models (LR, SVR, RF, XGBoost, Tweedie, Hurdle); saves legacy model files
4. `05_improved_modeling.ipynb` — **run this for best performance**; adds 8 derived features, trains LightGBM + XGBoost v2, builds weighted ensemble; saves `lightgbm.pkl`, `xgboost_v2.pkl`, updates `model_meta.json`

Run in order. Notebooks 3→4→5 each consume outputs from the prior step.

## Architecture

**Two feature sets — critical distinction:**
- **55-feature base set** (`feature_cols` in `model_meta.json`): all models from `04_modeling.ipynb` (LR, SVR, RF, XGBoost, Tweedie, Hurdle). Requires `scaler.pkl` for LR/SVR.
- **63-feature extended set** (`feature_cols_extended` in `model_meta.json`): models from `05_improved_modeling.ipynb` (LightGBM, XGBoost v2, Ensemble). 8 derived features added on top of the base 55 — all computable from existing inputs, no new user input needed.

**Data flow — primary path (app):**
User Input → 55 Base Features → `add_derived_features()` → 63 Extended Features → LGB+XGB Ensemble → expm1 → Dollar Output → Percentile Lookup

**Data flow — legacy path (app, for comparison display):**
User Input → 55 Base Features → [Signed Log1p → StandardScaler for LR/SVR] → Model → expm1 × Smearing Factor → Dollar Output

**Models (all in `models/`):**
- `lightgbm.pkl`, `xgboost_v2.pkl` — **primary models** (63 feat, no smearing); ensemble weights in `model_meta.json` under `ensemble.weights`
- `linear_regression.pkl`, `svr.pkl`, `random_forest.pkl`, `xgboost.pkl` — baseline log-space models (55 feat)
- `xgboost_tweedie.pkl` — Tweedie regression for zero-inflated spending data (55 feat)
- `hurdle_classifier.pkl` + `hurdle_regressor.pkl` — two-stage hurdle model (55 feat)

**Key config files:**
- `models/model_meta.json` — feature column lists (base + extended), model metadata, smearing factors, ensemble weights
- `notebooks/web_app_constants.json` — age gates, imputation defaults, income transform rules
- `models/model_results.json` — test set performance metrics for all models

## Feature Engineering Notes

**Age-gating** (MEPS survey design — features only apply within certain age ranges):
- K6/PHQ-2 mental health scores: adults 18+
- Student status: ages 17–23
- Adult chronic conditions: 18+
- ADHD: ages 5–17

**Transformations:**
- Target variable (`TOTEXP`): log1p-transformed before model fitting; Duan's smearing factors (stored in `smearing_factors.json`) correct for the retransformation bias in legacy models; new models (LGB, XGB v2) use direct expm1 with no smearing
- Income: signed log1p applied to handle negative and zero values before StandardScaler

**8 derived features** (added in `05_improved_modeling.ipynb`, also computed in `app.py:add_derived_features()`):
- `num_conditions`: count of all 12 dx_ flags (comorbidity burden)
- `is_elderly`: age ≥ 65 (Medicare eligibility discontinuity)
- `bmi_obese`: BMI ≥ 30
- `dual_eligible`: Medicare AND Medicaid (2–3× per-capita spending)
- `cardiac_burden`: CHD + angina + MI + stroke (0–4)
- `age_sq`: age² / 1000 (captures non-linear aging effect)
- `age_x_conditions`: age × num_conditions / 10
- `health_burden`: self_rated_health × num_conditions / 10

**55 base features** in `model_meta.json → feature_cols`; **63 extended features** in `feature_cols_extended`.

## Stack

Python 3 · pandas · numpy · scikit-learn · XGBoost · Streamlit · joblib · matplotlib · seaborn
