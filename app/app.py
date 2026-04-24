import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Spending Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, '..', 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))

    # Legacy log-space models (55 features, scaled)
    log_models = {}
    for name, fname in [
        ('Linear Regression', 'baseline/linear_regression.pkl'),
        ('SVR',               'baseline/svr.pkl'),
        ('Random Forest',     'random_forest.pkl'),
        ('XGBoost',           'baseline/xgboost.pkl'),
    ]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            log_models[name] = joblib.load(p)

    # Improved models — trained on 63-feature extended set
    improved = {}
    for key, fname in [
        ('tweedie',    'hurdle/xgboost_tweedie.pkl'),
        ('hurdle_clf', 'hurdle/hurdle_classifier.pkl'),
        ('hurdle_reg', 'hurdle/hurdle_regressor.pkl'),
        ('lgb',        'primary/lightgbm.pkl'),
        ('xgb_v2',     'primary/xgboost_v2.pkl'),
    ]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            improved[key] = joblib.load(p)

    with open(os.path.join(MODELS_DIR, 'meta/model_meta.json'))        as f: meta          = json.load(f)
    with open(os.path.join(MODELS_DIR, 'meta/smearing_factors.json'))  as f: smearing_data  = json.load(f)
    with open(os.path.join(MODELS_DIR, 'meta/percentile_lookup.json')) as f: pct_data       = json.load(f)

    return scaler, log_models, improved, meta, smearing_data, pct_data

scaler, log_models, improved, meta, smearing_data, pct_data = load_assets()

feature_cols     = meta['feature_cols']
feature_cols_ext = meta.get('feature_cols_extended', feature_cols)
smearing         = smearing_data['smearing_factors']
hurdle_sf        = smearing_data['hurdle_smearing']
meps_mean        = pct_data['meps_mean']
meps_median      = pct_data['meps_median']
pct_lookup       = pct_data['percentiles']

ensemble_cfg  = meta.get('ensemble', {})
ensemble_w    = ensemble_cfg.get('weights', [0.5, 0.5])
has_ensemble  = 'lgb' in improved and 'xgb_v2' in improved
models_ready  = len(log_models) > 0 or has_ensemble

# ── Helpers ───────────────────────────────────────────────────────────────────
DX_COLS = [
    'dx_hypertension', 'dx_coronary_heart_disease', 'dx_angina',
    'dx_myocardial_infarction', 'dx_stroke', 'dx_emphysema',
    'dx_high_cholesterol', 'dx_cancer', 'dx_arthritis',
    'dx_asthma', 'dx_adhd_add', 'dx_diabetes',
]
CARDIAC_COLS = [
    'dx_coronary_heart_disease', 'dx_angina',
    'dx_myocardial_infarction', 'dx_stroke',
]

def signed_log1p(x):
    return float(np.sign(x) * np.log1p(abs(x)))

def get_percentile(dollar_pred):
    for p in range(100, -1, -1):
        if dollar_pred >= pct_lookup[str(p)]:
            return p
    return 0

def add_derived_features(row: dict) -> dict:
    """Compute the 8 derived features from existing input values.
    These match exactly what 05_improved_modeling.ipynb uses during training.
    """
    num_cond  = sum(row[c] for c in DX_COLS if c in row)
    cardiac   = sum(row[c] for c in CARDIAC_COLS if c in row)
    age       = row['age']
    bmi       = row['bmi']
    srh       = row['self_rated_health']

    row['num_conditions']   = float(num_cond)
    row['is_elderly']       = float(age >= 65)
    row['bmi_obese']        = float(bmi >= 30)
    row['dual_eligible']    = float(row.get('has_medicare', 0) == 1 and
                                    row.get('has_medicaid', 0) == 1)
    row['cardiac_burden']   = float(cardiac)
    row['age_sq']           = float(age ** 2) / 1000.0
    row['age_x_conditions'] = float(age * num_cond) / 10.0
    row['health_burden']    = float(srh * num_cond) / 10.0
    return row

def predict_all(X_raw_55, X_scaled, X_raw_63):
    preds = {}

    # Original log-space models + Duan's smearing correction
    for name, model in log_models.items():
        X      = X_scaled if name in ('Linear Regression', 'SVR') else X_raw_55
        log_p  = float(model.predict(X)[0])
        dollar = max(0.0, np.expm1(log_p) * smearing.get(name, 1.0))
        preds[name] = dollar

    # XGBoost Tweedie (raw dollar output — no expm1 needed)
    if 'tweedie' in improved:
        preds['XGBoost Tweedie'] = max(0.0, float(improved['tweedie'].predict(X_raw_55)[0]))

    # Two-Stage Hurdle: P(spend>0) × smearing-corrected amount
    if 'hurdle_clf' in improved and 'hurdle_reg' in improved:
        p_spend = float(improved['hurdle_clf'].predict_proba(X_raw_55)[0, 1])
        log_amt = float(improved['hurdle_reg'].predict(X_raw_55)[0])
        preds['Two-Stage Hurdle'] = max(0.0, p_spend * np.expm1(log_amt) * hurdle_sf)

    # LightGBM (63-feature extended set, log1p → expm1)
    if 'lgb' in improved:
        log_p = float(improved['lgb'].predict(X_raw_63)[0])
        preds['LightGBM'] = max(0.0, float(np.expm1(log_p)))

    # XGBoost v2 (63-feature extended set, log1p → expm1)
    if 'xgb_v2' in improved:
        log_p = float(improved['xgb_v2'].predict(X_raw_63)[0])
        preds['XGBoost v2'] = max(0.0, float(np.expm1(log_p)))

    # Ensemble: weighted average of LGB + XGB v2 in log space
    if 'LightGBM' in preds and 'XGBoost v2' in preds:
        lgb_log  = float(improved['lgb'].predict(X_raw_63)[0])
        xgb_log  = float(improved['xgb_v2'].predict(X_raw_63)[0])
        w_lgb, w_xgb = ensemble_w[0], ensemble_w[1]
        ens_log  = w_lgb * lgb_log + w_xgb * xgb_log
        preds['Ensemble (LGB+XGB)'] = max(0.0, float(np.expm1(ens_log)))

    return preds

# ── Sidebar: User Inputs ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("Your Information")
    st.caption("Fill in your details to estimate annual healthcare spending.")

    st.subheader("Demographics")
    age      = st.slider("Age", 0, 85, 35)
    sex      = st.radio("Sex", ["Male", "Female"], horizontal=True)
    hispanic = st.radio("Hispanic / Latino", ["Yes", "No"], horizontal=True, index=1)
    race     = st.selectbox("Race", [
        "White", "Black", "Asian Indian", "Chinese", "Filipino",
        "Other Asian / Pacific Islander", "Native American", "Multiple races",
    ])
    marital  = st.selectbox("Marital Status", [
        "Married", "Never Married", "Divorced", "Widowed", "Separated",
    ])
    region = st.selectbox("Region", ["South", "West", "Midwest", "Northeast"])

    st.subheader("Socioeconomic")
    education_years = st.slider("Years of Education", 0, 17, 12)
    family_size     = st.slider("Family Size", 1, 14, 3)
    poverty_label   = st.selectbox("Household Income Level", [
        "Poor / Negative income", "Near Poor", "Low Income", "Middle Income", "High Income",
    ], index=3)
    poverty_category = {
        "Poor / Negative income": 1, "Near Poor": 2, "Low Income": 3,
        "Middle Income": 4, "High Income": 5,
    }[poverty_label]

    family_income       = st.number_input("Annual Family Income ($)",        -50000, 500000, 60000, step=1000)
    total_person_income = st.number_input("Your Annual Personal Income ($)", -50000, 300000, 30000, step=1000)

    emp_label = st.selectbox("Employment Status", [
        "Not Applicable (under 16)", "Employed", "Has job to return to",
        "Worked during year", "Not Employed",
    ], index=1)
    employment_status = {
        "Not Applicable (under 16)": 0, "Employed": 1, "Has job to return to": 2,
        "Worked during year": 3, "Not Employed": 4,
    }[emp_label]

    if 17 <= age <= 23:
        s_label    = st.radio("Student Status", ["Not a Student", "Part-time Student", "Full-time Student"])
        is_student = {"Not a Student": 0, "Part-time Student": 1, "Full-time Student": 2}[s_label]
    else:
        is_student = 0

    st.subheader("Health Status")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, step=0.1)

    h_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    self_rated_health        = h_map[st.selectbox("Overall Physical Health", list(h_map), index=2)]
    self_rated_mental_health = h_map[st.selectbox("Overall Mental Health",   list(h_map), index=2)]

    needs_help_iadl = 1 if st.checkbox("Needs help with daily activities (IADL)") else 0
    needs_help_adl  = 1 if st.checkbox("Needs help with personal care (ADL)") else 0
    any_limitation  = 1 if st.checkbox("Has any functional limitation") else 0

    if age >= 18:
        k6_distress_score     = st.slider("K6 Psychological Distress (0–24)", 0, 24, 2,
                                           help="Score ≥ 13 = serious distress.")
        phq2_depression_score = st.slider("PHQ-2 Depression Score (0–6)",     0,  6, 0,
                                           help="Score ≥ 3 suggests screening.")
    else:
        k6_distress_score = phq2_depression_score = 0

    st.subheader("Chronic Conditions")
    dx_asthma   = 1 if st.checkbox("Asthma") else 0
    dx_diabetes = 1 if st.checkbox("Diabetes") else 0

    if age >= 18:
        dx_hypertension           = 1 if st.checkbox("Hypertension") else 0
        dx_high_cholesterol       = 1 if st.checkbox("High Cholesterol") else 0
        dx_arthritis              = 1 if st.checkbox("Arthritis") else 0
        dx_cancer                 = 1 if st.checkbox("Cancer") else 0
        dx_coronary_heart_disease = 1 if st.checkbox("Coronary Heart Disease") else 0
        dx_angina                 = 1 if st.checkbox("Angina") else 0
        dx_myocardial_infarction  = 1 if st.checkbox("Heart Attack (MI)") else 0
        dx_stroke                 = 1 if st.checkbox("Stroke") else 0
        dx_emphysema              = 1 if st.checkbox("Emphysema / COPD") else 0
    else:
        dx_hypertension = dx_high_cholesterol = dx_arthritis = dx_cancer = 0
        dx_coronary_heart_disease = dx_angina = dx_myocardial_infarction = 0
        dx_stroke = dx_emphysema = 0

    dx_adhd_add = 1 if (5 <= age <= 17 and st.checkbox("ADHD / ADD")) else 0

    st.subheader("Insurance")
    has_private_insurance = 1 if st.checkbox("Private Insurance", value=True) else 0
    has_tricare           = 1 if st.checkbox("TRICARE (military)") else 0
    has_medicare          = 1 if st.checkbox("Medicare") else 0
    has_medicaid          = 1 if st.checkbox("Medicaid") else 0
    has_va_coverage       = 1 if st.checkbox("VA Coverage") else 0
    is_uninsured          = 1 if st.checkbox("Uninsured") else 0
    has_usual_care        = 1 if st.checkbox("Has a usual care provider", value=True) else 0

# ── Build Feature Vector ──────────────────────────────────────────────────────
def build_input():
    # Race one-hot
    race_white = race_black = race_native_american = 0
    race_asian_indian = race_chinese = race_filipino = race_other_asian_pi = race_multiple = 0
    if race == "White":                             race_white          = 1
    elif race == "Black":                           race_black          = 1
    elif race == "Native American":                 race_native_american = 1
    elif race == "Asian Indian":                    race_asian_indian   = 1
    elif race == "Chinese":                         race_chinese        = 1
    elif race == "Filipino":                        race_filipino       = 1
    elif race == "Other Asian / Pacific Islander":  race_other_asian_pi = 1
    elif race == "Multiple races":                  race_multiple       = 1

    # Marital one-hot
    marital_married = marital_divorced = marital_never_married = 0
    marital_separated = marital_widowed = marital_under_16 = 0
    if age < 16:
        marital_under_16 = 1
    elif marital == "Married":         marital_married       = 1
    elif marital == "Divorced":        marital_divorced      = 1
    elif marital == "Never Married":   marital_never_married = 1
    elif marital == "Separated":       marital_separated     = 1
    elif marital == "Widowed":         marital_widowed       = 1

    # Region one-hot
    region_south = region_west = region_midwest = region_northeast = 0
    if region == "South":       region_south     = 1
    elif region == "West":      region_west      = 1
    elif region == "Midwest":   region_midwest   = 1
    elif region == "Northeast": region_northeast = 1

    # Base 55-feature row
    row = {
        'age':                      age,
        'sex':                      1 if sex == "Male" else 0,
        'hispanic':                 1 if hispanic == "Yes" else 0,
        'bmi':                      bmi,
        'education_years':          education_years,
        'family_size':              family_size,
        'poverty_category':         poverty_category,
        'family_income':            signed_log1p(family_income),
        'total_person_income':      signed_log1p(total_person_income),
        'employment_status':        employment_status,
        'is_student':               is_student,
        'self_rated_health':        self_rated_health,
        'self_rated_mental_health': self_rated_mental_health,
        'needs_help_iadl':          needs_help_iadl,
        'needs_help_adl':           needs_help_adl,
        'any_limitation':           any_limitation,
        'k6_distress_score':        k6_distress_score,
        'phq2_depression_score':    phq2_depression_score,
        'dx_hypertension':             dx_hypertension,
        'dx_coronary_heart_disease':   dx_coronary_heart_disease,
        'dx_angina':                   dx_angina,
        'dx_myocardial_infarction':    dx_myocardial_infarction,
        'dx_stroke':                   dx_stroke,
        'dx_emphysema':                dx_emphysema,
        'dx_high_cholesterol':         dx_high_cholesterol,
        'dx_cancer':                   dx_cancer,
        'dx_arthritis':                dx_arthritis,
        'dx_asthma':                   dx_asthma,
        'dx_adhd_add':                 dx_adhd_add,
        'dx_diabetes':                 dx_diabetes,
        'has_private_insurance':   has_private_insurance,
        'has_tricare':             has_tricare,
        'has_medicare':            has_medicare,
        'has_medicaid':            has_medicaid,
        'has_va_coverage':         has_va_coverage,
        'is_uninsured':            is_uninsured,
        'has_usual_care':          has_usual_care,
        'race_white':              race_white,
        'race_other_asian_pi':     race_other_asian_pi,
        'race_multiple':           race_multiple,
        'race_black':              race_black,
        'race_native_american':    race_native_american,
        'race_asian_indian':       race_asian_indian,
        'race_chinese':            race_chinese,
        'race_filipino':           race_filipino,
        'marital_divorced':        marital_divorced,
        'marital_married':         marital_married,
        'marital_never_married':   marital_never_married,
        'marital_separated':       marital_separated,
        'marital_under_16':        marital_under_16,
        'marital_widowed':         marital_widowed,
        'region_midwest':          region_midwest,
        'region_northeast':        region_northeast,
        'region_south':            region_south,
        'region_west':             region_west,
    }

    # Add 8 derived features for extended models
    row_ext = add_derived_features(dict(row))

    X_raw_55 = pd.DataFrame([row])[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X_raw_55), columns=feature_cols)
    X_raw_63 = pd.DataFrame([row_ext])[feature_cols_ext]

    return X_raw_55, X_scaled, X_raw_63

# ── Main Page ─────────────────────────────────────────────────────────────────
st.title("Healthcare Spending Predictor")
st.markdown(
    "Estimate your **annual total healthcare spending** based on your health profile. "
    "Trained on **22,431 respondents** from the 2022 Medical Expenditure Panel Survey (MEPS)."
)

if not models_ready:
    st.error("Models not found. Run `notebooks/04_modeling.ipynb` and `notebooks/05_improved_modeling.ipynb` first.")
    st.stop()

X_raw_55, X_sc, X_raw_63 = build_input()
predictions = predict_all(X_raw_55, X_sc, X_raw_63)

# Primary prediction: ensemble if available, otherwise best single model
if 'Ensemble (LGB+XGB)' in predictions:
    primary_pred = predictions['Ensemble (LGB+XGB)']
    primary_label = 'Ensemble (LGB+XGB)'
elif 'LightGBM' in predictions:
    primary_pred = predictions['LightGBM']
    primary_label = 'LightGBM'
else:
    primary_pred = predictions.get('XGBoost', list(predictions.values())[0])
    primary_label = 'XGBoost'

percentile = get_percentile(primary_pred)

# ── Hero metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Predicted Annual Spending",
        f"${primary_pred:,.0f}",
        help=f"Primary model: {primary_label}. Weighted ensemble of LightGBM + XGBoost v2 "
             "trained on 63 features including comorbidity count, age², and interaction terms.",
    )
with c2:
    st.metric("Monthly Estimate", f"${primary_pred / 12:,.0f}")

with c3:
    if percentile <= 25:   tier = "Low spender"
    elif percentile <= 50: tier = "Below median"
    elif percentile <= 75: tier = "Above median"
    elif percentile <= 90: tier = "High spender"
    else:                  tier = "Very high spender"
    st.metric(
        "MEPS Percentile",
        f"{percentile}th percentile",
        help=f"{tier} among 2022 MEPS respondents.",
    )

with c4:
    diff = primary_pred - meps_median
    sign = "+" if diff >= 0 else ""
    st.metric(
        f"vs. MEPS Median (${meps_median:,.0f})",
        f"{sign}${abs(diff):,.0f}",
        delta=f"{sign}{diff / meps_median * 100:.0f}%" if meps_median > 0 else "N/A",
        delta_color="inverse",
        help=f"MEPS 2022: median ${meps_median:,.0f} · mean ${meps_mean:,.0f}. "
             "Note: the commonly cited '$13,493 US average' is the CMS national per-capita figure "
             "which includes insurer overhead and admin costs — not individual care received.",
    )

st.divider()

st.caption(
    f"**MEPS 2022 spending distribution** — "
    f"10th pct: $0 · 25th: ${pct_lookup['25']:,.0f} · "
    f"Median: ${meps_median:,.0f} · "
    f"75th: ${pct_lookup['75']:,.0f} · "
    f"90th: ${pct_lookup['90']:,.0f} · "
    f"Mean: ${meps_mean:,.0f}"
)

# ── All model predictions ─────────────────────────────────────────────────────
st.subheader("All Model Predictions")
st.caption(
    "Legacy models (LR/SVR/RF/XGBoost) use Duan's smearing correction on the original 55 features. "
    "LightGBM, XGBoost v2, and the Ensemble use 63 features with no smearing correction."
)

MODEL_ORDER = [
    ('Linear Regression',    'log1p + smearing, 55 feat'),
    ('SVR',                  'log1p + smearing, 55 feat'),
    ('Random Forest',        'log1p + smearing, 55 feat'),
    ('XGBoost',              'log1p + smearing, 55 feat'),
    ('XGBoost Tweedie',      'Tweedie p=1.5, 55 feat'),
    ('Two-Stage Hurdle',     'P(spend>0) × amount, 55 feat'),
    ('LightGBM',             '63 feat, no smearing'),
    ('XGBoost v2',           '63 feat, no smearing'),
    ('Ensemble (LGB+XGB)',   '★ Primary — 63 feat, weighted blend'),
]

available = [(n, m) for n, m in MODEL_ORDER if n in predictions]
pred_cols = st.columns(min(len(available), 5))
for i, (name, method) in enumerate(available):
    with pred_cols[i % len(pred_cols)]:
        is_primary = (name == primary_label)
        st.metric(
            label=f"{'★ ' if is_primary else ''}{name}",
            value=f"${predictions[name]:,.0f}",
            help=f"Method: {method}",
        )

chart_data = pd.DataFrame({
    'Model':                  [n for n, _ in available],
    'Predicted Spending ($)': [predictions[n] for n, _ in available],
})
st.bar_chart(chart_data.set_index('Model'))

st.divider()

# ── Limitations ───────────────────────────────────────────────────────────────
st.info(
    "**Model limitations:** The strongest predictors of healthcare spending — prior-year spending, "
    "number of prescriptions, planned procedures, and number of visits — are not captured in MEPS "
    "cross-sectional survey data. Even the best model explains ~50% of spending variance. "
    "Healthcare spending is inherently hard to predict due to random health events. "
    "Use this as a rough estimate, not a precise forecast."
)

# ── Model performance ─────────────────────────────────────────────────────────
with st.expander("Model Performance on Test Set (4,487 people)", expanded=False):
    results_path = os.path.join(MODELS_DIR, 'meta/model_results.json')
    if os.path.exists(results_path):
        res_df = pd.read_json(results_path)
        if not res_df.empty:
            disp = res_df[['model', 'r2', 'rmse_dollar', 'mae_dollar']].copy()
            disp.columns = ['Model', 'R² (log scale)', 'RMSE ($)', 'MAE ($)']
            disp['RMSE ($)']      = disp['RMSE ($)'].apply(lambda x: f"${x:,.0f}")
            disp['MAE ($)']       = disp['MAE ($)'].apply(lambda x: f"${x:,.0f}")
            disp['R² (log scale)'] = disp['R² (log scale)'].apply(lambda x: f"{x:.4f}")
            st.dataframe(disp.set_index('Model'), use_container_width=True)
    st.caption(
        "R² on log1p scale. Dollar figures for original models are before smearing correction. "
        "LightGBM, XGBoost v2, and Ensemble use direct expm1 output (no smearing)."
    )

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("About this tool", expanded=False):
    w_lgb = ensemble_cfg.get('weights', [0.5, 0.5])[0]
    w_xgb = ensemble_cfg.get('weights', [0.5, 0.5])[1]
    st.markdown(f"""
    **Data**: MEPS 2022 Full-Year Consolidated File (h243.dta) — 22,431 civilian non-institutionalized
    US residents. Agency for Healthcare Research and Quality.

    **Target**: Total annual healthcare expenditure (insurance + out-of-pocket payments).
    MEPS 2022: median **${meps_median:,.0f}** · mean **${meps_mean:,.0f}** · 15.1% had $0 spending.

    **Why not $13,493?** The CMS national per-capita figure includes insurer overhead, administrative
    costs, and government program spending that does not directly flow to individual care. MEPS
    measures payments for actual services received — the correct denominator for individual prediction.

    **Primary model — Ensemble (LGB+XGB)** (`{w_lgb:.2f} × LightGBM + {w_xgb:.2f} × XGBoost v2`):
    - Trained on 63 features: 55 base features + 8 derived
      (`num_conditions`, `is_elderly`, `bmi_obese`, `dual_eligible`, `cardiac_burden`,
      `age_sq`, `age_x_conditions`, `health_burden`)
    - Proper train/validation split for early stopping — no test-set leakage
    - Ensemble weights chosen on held-out validation set

    **Legacy models** (for comparison only):
    - **LR / SVR / RF / XGBoost** — trained on log1p(expenditure); Duan smearing applied at inference
    - **XGBoost Tweedie** (p=1.5) — compound Poisson-Gamma for zero-inflated data
    - **Two-Stage Hurdle** — XGBClassifier (any spending?) + XGBRegressor (how much?)

    **Age-gated inputs** (MEPS survey design): student status 17–23 · K6/PHQ-2 adults 18+ ·
    adult chronic conditions 18+ · ADHD 5–17.
    """)
