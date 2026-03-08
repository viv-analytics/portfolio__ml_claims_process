"""Streamlit demo app — Motor Vehicle Claims ML Pipeline.

Run with:
    uv run streamlit run streamlit_app.py

Improvements over v1:
  - Preset scenario buttons (Low Risk / Young Driver / Senior Driver / High Risk)
  - Severity model → E[severity | claim] and expected loss (pure premium proxy)
  - Portfolio distribution chart: where does this policy rank?
  - 4-metric summary row including expected loss
  - Session-state-backed form so presets populate inputs immediately
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="Motor Claims ML",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from claims.data import load_fremtpl2, build_claims_dataset, claims_only
from claims.features import (
    CATEGORICAL_FEATURES,
    ENG_NUMERIC_FEATURES,
    engineer_features,
)
from claims.process.severity import catboost_severity

ALL_ENG = ENG_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ─── Preset policy profiles ──────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "🟢 Low Risk":       {"driv_age": 45, "veh_age": 8,  "bonus_malus": 80,  "exposure": 1.0, "density": 250},
    "🧑 Young Driver":  {"driv_age": 22, "veh_age": 2,  "bonus_malus": 115, "exposure": 0.5, "density": 3000},
    "👴 Senior Driver": {"driv_age": 76, "veh_age": 6,  "bonus_malus": 95,  "exposure": 0.8, "density": 400},
    "🔴 High Risk":     {"driv_age": 24, "veh_age": 1,  "bonus_malus": 150, "exposure": 1.0, "density": 9000},
}

_NUMERIC_DEFAULTS = {"driv_age": 35, "veh_age": 5, "bonus_malus": 100, "exposure": 0.5, "density": 500}


# ─── Cached model training (runs once; ~4 min on first load) ─────────────────
@st.cache_resource(show_spinner="Training models on freMTPL2 (one-time setup ~4 min)…")
def _train_models():
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split

    from claims.classification.calibration import calibrate_beta

    freq, sev = load_fremtpl2()
    df = build_claims_dataset(freq, sev)
    df = engineer_features(df)

    # ── Classification: strict 4-way split ───────────────────────────────────
    df_pool, _ = train_test_split(df, test_size=0.10, stratify=df["HasClaim"], random_state=42)
    df_train, df_temp = train_test_split(df_pool, test_size=0.333, stratify=df_pool["HasClaim"], random_state=42)
    df_early, df_cal = train_test_split(df_temp, test_size=0.50, stratify=df_temp["HasClaim"], random_state=42)

    cb = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=7,
        auto_class_weights="Balanced", eval_metric="AUC",
        early_stopping_rounds=40, random_seed=42, verbose=0,
    )
    cb.fit(
        df_train[ALL_ENG], df_train["HasClaim"].values,
        eval_set=(df_early[ALL_ENG], df_early["HasClaim"].values),
        cat_features=CATEGORICAL_FEATURES, verbose=False,
    )
    beta_cal = calibrate_beta(cb.predict_proba(df_cal[ALL_ENG])[:, 1], df_cal["HasClaim"].values)

    # ── Severity model: CatBoost on log1p(AvgSeverity) ───────────────────────
    df_sev = claims_only(df).dropna(subset=["AvgSeverity"])
    df_sev = df_sev[df_sev["AvgSeverity"] > 0].reset_index(drop=True)
    sev_tr, sev_te = train_test_split(df_sev, test_size=0.2, random_state=42)
    y_log_tr = np.log1p(sev_tr["AvgSeverity"].values)
    y_log_te = np.log1p(sev_te["AvgSeverity"].values)
    cb_sev = catboost_severity(iterations=300)
    cb_sev.fit(
        sev_tr[ALL_ENG], y_log_tr,
        eval_set=(sev_te[ALL_ENG], y_log_te),
        cat_features=CATEGORICAL_FEATURES, verbose=False,
    )

    # ── Portfolio sample for distribution chart (5k calibrated probs) ────────
    sample = df.sample(5_000, random_state=42)
    s_raw = cb.predict_proba(sample[ALL_ENG])[:, 1]
    portfolio_probs = np.clip(beta_cal.predict(s_raw.reshape(-1, 1)), 0, 1).astype(float)

    cat_opts = {
        col: sorted(df[col].astype(str).dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    portfolio_stats = {
        "n_policies": len(df),
        "claim_rate":  float(df["HasClaim"].mean()),
        "avg_severity": float(df["AvgSeverity"].dropna().mean()),
        "avg_exposure": float(pd.to_numeric(df["Exposure"], errors="coerce").mean()),
    }

    return cb, beta_cal, cb_sev, cat_opts, portfolio_stats, portfolio_probs


cb_model, beta_cal, cb_sev, cat_opts, stats, portfolio_probs = _train_models()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _stp_lane(p: float) -> str:
    if p < 0.10:  return "auto-settle"
    if p >= 0.40: return "investigate"
    return "review"


def _portfolio_chart(cal_prob: float) -> plt.Figure:
    """Histogram of 5k portfolio calibrated probs with this policy marked."""
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(portfolio_probs, bins=60, color="#4472C4", alpha=0.75, edgecolor="white", density=True)
    ax.axvline(cal_prob, color="#ED7D31", lw=2.5, linestyle="--", label=f"This policy  p={cal_prob:.3f}")
    pct = float((portfolio_probs <= cal_prob).mean()) * 100
    ymax = ax.get_ylim()[1]
    ax.text(cal_prob + 0.002, ymax * 0.82, f"{pct:.0f}th\npct", color="#ED7D31", fontsize=9, fontweight="bold")
    ax.set_xlabel("Calibrated Claim Probability", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Portfolio Distribution — Where does this policy sit?", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🚗 Motor Vehicle Claims — ML Decision Support")
st.caption(
    "French Motor TPL (freMTPL2) · 678k policies · "
    "CatBoost + Beta calibration · SHAP explainability"
)

tab_predict, tab_portfolio = st.tabs(["🔮 Predict", "📊 Portfolio Overview"])


# ─── Tab: Portfolio Overview ──────────────────────────────────────────────────
with tab_portfolio:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Policies",     f"{stats['n_policies']:,}")
    c2.metric("Claim Rate",         f"{stats['claim_rate']:.2%}")
    c3.metric("Avg Claim Severity", f"€{stats['avg_severity']:,.0f}")
    c4.metric("Avg Exposure",       f"{stats['avg_exposure']:.2f} years")

    st.markdown("---")

    fig_dir = Path(__file__).parent / "reports" / "figures"
    for fname, caption in [
        ("01_eda_overview.png",      "EDA — Claim distribution & feature overview"),
        ("02_classification.png",    "Classification — Beta & Venn-ABERS calibration, STP routing"),
        ("03_fraud_detection.png",   "Fraud Detection — Anomaly scoring + cost-sensitive CatBoost"),
        ("04_process_analytics.png", "Process Analytics — Lorenz curve, severity models, P50/P75/P95 reserves"),
        ("05_summary.png",           "Results Summary — CatBoost feature importance & metrics"),
        ("06_shap_beeswarm.png",     "SHAP Beeswarm — Global feature impact (500 test samples)"),
        ("06_shap_waterfall.png",    "SHAP Waterfall — Highest-risk prediction breakdown"),
    ]:
        fpath = fig_dir / fname
        if fpath.exists():
            st.image(str(fpath), caption=caption, width="stretch")
            st.markdown("")
        else:
            st.info(f"📁 `{fname}` not found — run `uv run python reports/make_figures.py` first.")


# ─── Tab: Predict ─────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Single-Policy Risk Assessment")

    # ── Initialise session-state defaults (first run only) ───────────────────
    for k, v in _NUMERIC_DEFAULTS.items():
        st.session_state.setdefault(k, v)

    # ── Preset buttons ────────────────────────────────────────────────────────
    st.markdown("**Quick scenario presets**")
    preset_cols = st.columns(len(PRESETS))
    for col, (name, vals) in zip(preset_cols, PRESETS.items()):
        if col.button(name, width="stretch"):
            for k, v in vals.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("---")

    col_form, col_result = st.columns([1, 2], gap="large")

    with col_form:
        st.markdown("**Driver & Vehicle**")
        driv_age    = st.number_input("Driver Age",                  min_value=18,  max_value=100,   step=1,    key="driv_age")
        veh_age     = st.number_input("Vehicle Age (years)",          min_value=0,   max_value=30,    step=1,    key="veh_age")
        bonus_malus = st.number_input("BonusMalus (100 = neutral)",   min_value=50,  max_value=350,   step=1,    key="bonus_malus")
        exposure    = st.slider(      "Exposure (policy years)",       0.0, 1.0,                      step=0.01, key="exposure")
        density     = st.number_input("Area Density (pop/km²)",        min_value=1,   max_value=30000, step=100,  key="density")

        st.markdown("**Categorical**")
        veh_brand = st.selectbox("Vehicle Brand", cat_opts["VehBrand"], key="veh_brand")
        veh_gas   = st.selectbox("Fuel Type",      cat_opts["VehGas"],   key="veh_gas")
        area      = st.selectbox("Area Code",      cat_opts["Area"],     key="area")
        region    = st.selectbox("Region",         cat_opts["Region"],   key="region")
        veh_power = st.selectbox("Vehicle Power",  cat_opts["VehPower"], key="veh_power")

        predict_btn = st.button("🔮 Predict Risk", type="primary", width="stretch")

    with col_result:
        if predict_btn:
            row = pd.DataFrame([{
                "VehAge": veh_age, "DrivAge": driv_age, "BonusMalus": bonus_malus,
                "Density": density, "Exposure": exposure,
                "VehBrand": veh_brand, "VehGas": veh_gas,
                "Area": area, "Region": region, "VehPower": veh_power,
            }])
            row_eng  = engineer_features(row)
            raw_prob = float(cb_model.predict_proba(row_eng[ALL_ENG])[:, 1][0])
            cal_prob = float(np.clip(beta_cal.predict(np.array([[raw_prob]])), 0, 1)[0])
            sev_est  = float(np.expm1(cb_sev.predict(row_eng[ALL_ENG]))[0])
            exp_loss = cal_prob * sev_est
            lane     = _stp_lane(cal_prob)

            # ── 4 key metrics ─────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Raw P(claim)",        f"{raw_prob:.3f}")
            m2.metric("Calibrated P (Beta)", f"{cal_prob:.3f}")
            m3.metric("E[Severity | claim]", f"€{sev_est:,.0f}",
                      help="Expected claim amount given a claim occurs")
            m4.metric("Expected Loss",       f"€{exp_loss:,.0f}",
                      help="P(claim) × E[severity | claim] — pure premium proxy")

            # ── STP decision banner ───────────────────────────────────────────
            lane_cfg = {
                "auto-settle": ("✅", "success", "Auto-settle — Instant payout, no adjuster needed."),
                "review":      ("⚠️", "warning", "Review — Manual adjuster assigned."),
                "investigate": ("🚨", "error",   "Investigate — Detailed review required before settlement."),
            }
            icon, alert_fn, msg = lane_cfg[lane]
            getattr(st, alert_fn)(f"{icon} **{lane.title()}** ({cal_prob:.1%})  —  {msg}")

            progress_val = min(cal_prob / 0.30, 1.0)
            st.progress(
                progress_val,
                text=f"Risk level: {cal_prob:.1%}   (portfolio mean: {stats['claim_rate']:.1%})",
            )

            # ── Portfolio context ─────────────────────────────────────────────
            st.markdown("---")
            pct_rank = float((portfolio_probs <= cal_prob).mean()) * 100
            st.markdown(
                f"**Portfolio Context** — riskier than **{pct_rank:.0f}%** of the sampled portfolio"
            )
            portfolio_fig = _portfolio_chart(cal_prob)
            st.pyplot(portfolio_fig, width="stretch")
            plt.close(portfolio_fig)

            # ── SHAP waterfall ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**SHAP Feature Contributions**")
            st.caption("Positive SHAP value = feature raises claim probability | Negative = lowers it")
            try:
                import shap
                shap_expl = shap.TreeExplainer(cb_model)
                sv = shap_expl(row_eng[ALL_ENG])
                plt.close("all")  # ensure SHAP starts with a clean figure
                shap.plots.waterfall(sv[0], max_display=12, show=False)
                st.pyplot(plt.gcf(), width="stretch")
                plt.close()
            except Exception as e:
                st.info(f"SHAP plot unavailable: {e}")

        else:
            st.info(
                "👈 Pick a **preset** or configure the policy on the left, then click **Predict Risk**.\n\n"
                "**What this app shows:**\n"
                "- **Calibrated P(claim)** — Beta-calibrated probability (ECE < 0.001 on test set)\n"
                "- **E[Severity | claim]** — Expected claim amount (CatBoost on log-transformed target)\n"
                "- **Expected Loss** — P × E[sev] = actuarial pure premium proxy\n"
                "- **STP Lane** — Auto-settle / Review / Investigate routing decision\n"
                "- **Portfolio rank** — Where this policy sits vs 5k sampled policies\n"
                "- **SHAP waterfall** — Which features drove this specific prediction\n\n"
                "**STP thresholds:** `p < 0.10` → Auto-settle · `0.10–0.40` → Review · `p ≥ 0.40` → Investigate"
            )
