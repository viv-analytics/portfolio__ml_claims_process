"""Generate all portfolio figures and save to reports/figures/.

Run with:  uv run python reports/make_figures.py

Produces 8 figures covering EDA, classification, fraud, and process analytics.
Takes ~3-5 minutes (includes freMTPL2 download on first run).
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import train_test_split as sk_split
from sklearn.calibration import calibration_curve as sk_cal_curve

from claims.data import load_fremtpl2, build_claims_dataset, split_dataset, claims_only
from claims.features import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENG_NUMERIC_FEATURES,
    engineer_features, get_preprocessor, add_fraud_features,
)
from claims.evaluation import (
    classification_report_dict, gini_coefficient,
    lorenz_curve, stp_routing, stp_summary,
)
from claims.classification.models import logistic_pipeline, catboost_classifier
from claims.classification.calibration import (
    calibrate_beta, calibrate_venn_abers, calibration_metrics,
    plot_reliability_diagram,
)
from claims.fraud.anomaly import create_fraud_proxy_labels, compute_anomaly_scores
from claims.fraud.supervised import cost_sensitive_catboost, optimal_threshold, business_impact
from claims.process.severity import gamma_glm_pipeline, catboost_severity, poisson_glm_pipeline

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

STYLE = "seaborn-v0_8-whitegrid"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

ALL_FEATURES     = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_ENG_FEATURES = ENG_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
print("[1/8] Loading & engineering freMTPL2 dataset …")
freq, sev = load_fremtpl2()
df_raw = build_claims_dataset(freq, sev)
df = engineer_features(df_raw)   # adds log_Density, DrivAge_sq, BM_excess, etc.
print(f"      {len(df):,} policies | claim rate: {df['HasClaim'].mean():.2%}")
print(f"      Engineered features: {ENG_NUMERIC_FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
print("[2/8] Figure 1 — EDA: claim distribution & feature overview …")
with plt.style.context(STYLE):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1a — HasClaim bar
    ax1 = fig.add_subplot(gs[0, 0])
    vc = df["HasClaim"].value_counts().sort_index()
    ax1.bar(["No Claim (0)", "Claim (1)"], vc.values,
            color=["#4472C4", "#ED7D31"], edgecolor="white")
    for i, v in enumerate(vc.values):
        ax1.text(i, v + 5000, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=9)
    ax1.set_title("Claim Occurrence Distribution", fontweight="bold")
    ax1.set_ylabel("Number of Policies")

    # 1b — ClaimNb distribution (zero-inflated)
    ax2 = fig.add_subplot(gs[0, 1])
    cnb = df["ClaimNb"].clip(upper=4).value_counts().sort_index()
    ax2.bar(cnb.index, cnb.values, color="#4472C4", edgecolor="white")
    ax2.set_title("Claim Count Distribution\n(zero-inflated Poisson)", fontweight="bold")
    ax2.set_xlabel("Number of Claims")
    ax2.set_ylabel("Number of Policies")
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(["0", "1", "2", "3", "4+"])

    # 1c — DrivAge distribution
    ax3 = fig.add_subplot(gs[0, 2])
    driv_age = pd.to_numeric(df["DrivAge"], errors="coerce").dropna()
    ax3.hist(driv_age, bins=40, color="#4472C4", edgecolor="white", alpha=0.85)
    ax3.axvline(driv_age.mean(), color="#ED7D31", lw=2, label=f"Mean {driv_age.mean():.0f}")
    ax3.axvline(driv_age.median(), color="#A9D18E", lw=2, linestyle="--",
                label=f"Median {driv_age.median():.0f}")
    ax3.set_title("Driver Age Distribution", fontweight="bold")
    ax3.set_xlabel("Driver Age")
    ax3.legend(fontsize=9)

    # 1d — Claim rate by VehBrand
    ax4 = fig.add_subplot(gs[1, 0])
    df["VehBrand_str"] = df["VehBrand"].astype(str)
    brand_rate = (df.groupby("VehBrand_str")["HasClaim"]
                  .mean().sort_values(ascending=False).head(10))
    ax4.barh(brand_rate.index, brand_rate.values, color="#4472C4", edgecolor="white")
    ax4.axvline(df["HasClaim"].mean(), color="#ED7D31", lw=2, linestyle="--",
                label=f"Overall {df['HasClaim'].mean():.2%}")
    ax4.set_title("Claim Rate by Vehicle Brand\n(top 10)", fontweight="bold")
    ax4.set_xlabel("Claim Rate")
    ax4.legend(fontsize=8)

    # 1e — BonusMalus distribution
    ax5 = fig.add_subplot(gs[1, 1])
    bm = pd.to_numeric(df["BonusMalus"], errors="coerce").dropna()
    ax5.hist(bm.clip(upper=200), bins=40, color="#4472C4", edgecolor="white", alpha=0.85)
    ax5.axvline(100, color="#ED7D31", lw=2, linestyle="--", label="Neutral (100)")
    ax5.set_title("BonusMalus Distribution\n(100 = neutral, >100 = penalised)", fontweight="bold")
    ax5.set_xlabel("BonusMalus Score")
    ax5.legend(fontsize=9)

    # 1f — Claim severity (log scale)
    ax6 = fig.add_subplot(gs[1, 2])
    sev_vals = df["AvgSeverity"].dropna()
    ax6.hist(np.log1p(sev_vals), bins=50, color="#4472C4", edgecolor="white", alpha=0.85)
    ax6.set_title("Claim Severity Distribution\n(log scale — heavy right tail)", fontweight="bold")
    ax6.set_xlabel("log(1 + Claim Amount €)")
    ax6.set_ylabel("Number of Claims")

    fig.suptitle("Figure 1 — Exploratory Data Analysis: freMTPL2 (678k Policies)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(FIG_DIR / "01_eda_overview.png", bbox_inches="tight")
    plt.close()
print("      Saved 01_eda_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
print("[3/8] Training classification models …")

# ── 4-way stratified split: train 60% / early-stop 20% / calibrate 10% / test 10%
# Calibration set is strictly isolated from model training — no data leakage.
df_pool, df_test_clf = sk_split(df, test_size=0.10, stratify=df["HasClaim"], random_state=42)
df_train_clf, df_temp = sk_split(df_pool, test_size=0.333, stratify=df_pool["HasClaim"], random_state=42)
df_early, df_cal = sk_split(df_temp, test_size=0.50, stratify=df_temp["HasClaim"], random_state=42)

y_train_clf = df_train_clf["HasClaim"].values
y_early     = df_early["HasClaim"].values
y_cal       = df_cal["HasClaim"].values
y_test_clf  = df_test_clf["HasClaim"].values

print(f"      Split sizes — train:{len(df_train_clf):,} early-stop:{len(df_early):,} "
      f"cal:{len(df_cal):,} test:{len(df_test_clf):,}")
print(f"      Positive rate — train:{y_train_clf.mean():.2%} cal:{y_cal.mean():.2%} "
      f"test:{y_test_clf.mean():.2%}")

# Feature matrices (engineered features for all models)
X_train_clf = df_train_clf[ALL_ENG_FEATURES].copy()
X_early     = df_early[ALL_ENG_FEATURES].copy()
X_cal       = df_cal[ALL_ENG_FEATURES].copy()
X_test_clf  = df_test_clf[ALL_ENG_FEATURES].copy()

# Logistic Regression with engineered features
print("      Fitting Logistic Regression …")
lr_pipe = logistic_pipeline()
lr_pipe.fit(X_train_clf, y_train_clf)
lr_prob = lr_pipe.predict_proba(X_test_clf)[:, 1]
print(f"      LR AUC: {classification_report_dict(y_test_clf, lr_prob)['roc_auc']:.4f}")

# CatBoost — auto_class_weights, more iterations, deeper trees
print("      Fitting CatBoost (800 iter, depth=7, auto_class_weights) …")
from catboost import CatBoostClassifier as _CB
cb_clf = _CB(
    iterations=800,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    auto_class_weights="Balanced",   # principled alternative to scale_pos_weight
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=0,
)
cb_clf.fit(
    X_train_clf, y_train_clf,
    eval_set=(X_early, y_early),
    cat_features=CATEGORICAL_FEATURES,
    verbose=False,
)
cb_prob_raw = cb_clf.predict_proba(X_test_clf)[:, 1]
print(f"      CatBoost AUC: {classification_report_dict(y_test_clf, cb_prob_raw)['roc_auc']:.4f}")

# ── Calibration — Beta & Venn-ABERS on dedicated held-out set (no leakage)
print("      Calibrating — Beta & Venn-ABERS …")
cb_prob_cal = cb_clf.predict_proba(X_cal)[:, 1]

cb_beta = calibrate_beta(cb_prob_cal, y_cal, parameters="abm")
cb_va   = calibrate_venn_abers(cb_clf, X_cal, y_cal)

cb_prob_beta = cb_beta.predict(cb_prob_raw.reshape(-1, 1))
cb_prob_va   = cb_va.predict_proba(X_test_clf)[:, 1]

for name, prob in [("Uncalibrated", cb_prob_raw), ("Beta", cb_prob_beta), ("Venn-ABERS", cb_prob_va)]:
    m   = classification_report_dict(y_test_clf, prob)
    cal = calibration_metrics(y_test_clf, prob)
    print(f"      {name:12s}  AUC={m['roc_auc']:.4f}  Brier={m['brier_score']:.4f}  "
          f"ECE={cal['ece']:.4f}  Kappa={m['cohen_kappa']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("[4/8] Figure 2 — Classification: model comparison & calibration …")
with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2a — Model comparison: AUC / Gini / ECE grouped bars
    ax = axes[0]
    model_labels = ["LogReg", "CatBoost\nraw", "CatBoost\n+Beta", "CatBoost\n+Venn-ABERS"]
    probs_list   = [lr_prob, cb_prob_raw, cb_prob_beta, cb_prob_va]
    aucs   = [classification_report_dict(y_test_clf, p)["roc_auc"]     for p in probs_list]
    ginis  = [classification_report_dict(y_test_clf, p)["gini"]        for p in probs_list]
    briers = [classification_report_dict(y_test_clf, p)["brier_score"] for p in probs_list]
    eces   = [calibration_metrics(y_test_clf, p)["ece"]                for p in probs_list]

    x = np.arange(len(model_labels))
    w = 0.2
    ax.bar(x - 1.5*w, aucs,   w, label="ROC-AUC",  color="#4472C4")
    ax.bar(x - 0.5*w, ginis,  w, label="Gini",      color="#ED7D31")
    ax.bar(x + 0.5*w, briers, w, label="Brier↓",    color="#A9D18E")
    ax.bar(x + 1.5*w, eces,   w, label="ECE↓",      color="#FF9999")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_title("Model & Calibration Comparison\n(↓ = lower is better)", fontweight="bold")
    ax.set_ylim(0, max(aucs) * 1.2)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylabel("Score")

    # 2b — Reliability diagram: Uncalibrated / Beta / Venn-ABERS
    ax = axes[1]
    plot_reliability_diagram(
        probs_dict={
            "Uncalibrated": (y_test_clf, cb_prob_raw),
            "Beta":         (y_test_clf, cb_prob_beta),
            "Venn-ABERS":   (y_test_clf, cb_prob_va),
        },
        n_bins=10, zoom_max=0.25, ax=ax,
    )

    # 2c — STP routing using Venn-ABERS (lowest ECE)
    ax = axes[2]
    best_prob = cb_prob_va if calibration_metrics(y_test_clf, cb_prob_va)["ece"] \
                           <= calibration_metrics(y_test_clf, cb_prob_beta)["ece"] \
                           else cb_prob_beta
    stp = stp_summary(best_prob, y_test_clf)
    colors = {"auto-settle": "#A9D18E", "review": "#FFD966", "investigate": "#ED7D31"}
    lane_colors = [colors[l] for l in stp.index]
    bars = ax.bar(stp.index, stp["pct"], color=lane_colors, edgecolor="white", width=0.5)
    for bar, (lane, row) in zip(bars, stp.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row['n_policies']:,}\nclaim rate:\n{row['claim_rate']:.1%}",
                ha="center", fontsize=8.5)
    ax.set_title("STP Routing (Beta-Calibrated)\nStraight-Through Processing", fontweight="bold")
    ax.set_ylabel("% of Policies")
    ax.set_ylim(0, stp["pct"].max() * 1.4)

    fig.suptitle("Figure 2 — Claim Occurrence Classification (Module 1)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_classification.png", bbox_inches="tight")
    plt.close()
print("      Saved 02_classification.png")

# ─────────────────────────────────────────────────────────────────────────────
print("[5/8] Figure 3 — Fraud detection …")
df_claims = claims_only(df)
df_claims = add_fraud_features(df_claims)
df_claims = create_fraud_proxy_labels(df_claims, severity_col="AvgSeverity")

fraud_rate = df_claims["FraudProxy"].mean()
print(f"      Fraud proxy rate in claims: {fraud_rate:.2%}")

# Anomaly scoring on numeric features only
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_feats = [f for f in NUMERIC_FEATURES if f in df_claims.columns]
X_fraud_num = df_claims[num_feats].apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy="median")
X_fraud_imp = imputer.fit_transform(X_fraud_num)
scaler = StandardScaler()
X_fraud_scaled = scaler.fit_transform(X_fraud_imp)

print("      Computing anomaly scores …")
anomaly_scores = compute_anomaly_scores(X_fraud_scaled, contamination=0.05)
df_claims["AnomalyScore"] = anomaly_scores

# Supervised fraud model
print("      Fitting cost-sensitive CatBoost fraud model …")
fraud_feats = num_feats + CATEGORICAL_FEATURES + ["AnomalyScore"]
available_feats = [f for f in fraud_feats if f in df_claims.columns]

train_f, test_f = split_dataset(df_claims, target="FraudProxy", test_size=0.25)
X_tf = train_f[available_feats]
X_ef = test_f[available_feats]
y_tf = train_f["FraudProxy"].values
y_ef = test_f["FraudProxy"].values

cat_feats_f = [f for f in CATEGORICAL_FEATURES if f in available_feats]
fraud_model = cost_sensitive_catboost(cat_features=cat_feats_f, iterations=200)
fraud_model.fit(X_tf, y_tf, eval_set=(X_ef, y_ef), cat_features=cat_feats_f, verbose=False)
fraud_prob = fraud_model.predict_proba(X_ef)[:, 1]

opt_t = optimal_threshold(y_ef, fraud_prob)
impact_df = business_impact(y_ef, fraud_prob, threshold=opt_t)

with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 3a — Anomaly score distribution by fraud label
    ax = axes[0]
    ax.hist(anomaly_scores[df_claims["FraudProxy"] == 0], bins=50,
            alpha=0.65, label="Legitimate (0)", color="#4472C4", density=True)
    ax.hist(anomaly_scores[df_claims["FraudProxy"] == 1], bins=50,
            alpha=0.65, label="Suspicious (1)", color="#ED7D31", density=True)
    ax.set_title("Anomaly Score Distribution\n(IsolationForest + LOF)", fontweight="bold")
    ax.set_xlabel("Combined Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # 3b — Fraud probability distribution
    ax = axes[1]
    ax.hist(fraud_prob[y_ef == 0], bins=40, alpha=0.65,
            label="Legitimate", color="#4472C4", density=True)
    ax.hist(fraud_prob[y_ef == 1], bins=40, alpha=0.65,
            label="Fraud proxy", color="#ED7D31", density=True)
    ax.axvline(opt_t, color="black", lw=2, linestyle="--",
               label=f"Optimal threshold = {opt_t:.2f}")
    ax.set_title("Fraud Probability Scores\n(Cost-Sensitive CatBoost)", fontweight="bold")
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # 3c — Business impact table
    ax = axes[2]
    ax.axis("off")
    tbl = ax.table(
        cellText=impact_df.values,
        colLabels=impact_df.columns,
        cellLoc="left",
        loc="center",
        bbox=[0, 0.1, 1, 0.8],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF3FB")
    ax.set_title("Estimated Business Impact\nvs. No-Model Baseline", fontweight="bold")

    fig.suptitle("Figure 3 — Fraud Detection: Anomaly + Cost-Sensitive Classification (Module 2)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_fraud_detection.png", bbox_inches="tight")
    plt.close()
print("      Saved 03_fraud_detection.png")

# ─────────────────────────────────────────────────────────────────────────────
print("[6/8] Figure 4 — Severity modeling …")
df_sev = claims_only(df).dropna(subset=["AvgSeverity"])
df_sev = df_sev[df_sev["AvgSeverity"] > 0].reset_index(drop=True)

train_s, test_s = split_dataset(df_sev, target="HasClaim", test_size=0.25)

# Use engineered features for severity
X_ts = train_s[ALL_ENG_FEATURES]
X_es = test_s[ALL_ENG_FEATURES]
y_es_true = test_s["AvgSeverity"].values

# Gamma GLM on raw severity (log link handles skew internally)
print("      Fitting Gamma GLM …")
gamma_pipe = gamma_glm_pipeline()
gamma_pipe.fit(X_ts, train_s["AvgSeverity"].values)
gamma_pred = gamma_pipe.predict(X_es)

# CatBoost on log1p-transformed target — RMSE on log scale ≈ minimises RMSLE
# on original scale, which is far more robust to the heavy tail than raw RMSE.
print("      Fitting CatBoost severity (log1p target) …")
y_log_train = np.log1p(train_s["AvgSeverity"].values)
y_log_eval  = np.log1p(y_es_true)
cb_sev = catboost_severity(iterations=500)
cb_sev.fit(
    X_ts, y_log_train,
    eval_set=(X_es, y_log_eval),
    cat_features=CATEGORICAL_FEATURES,
    verbose=False,
)
cb_sev_pred = np.expm1(cb_sev.predict(X_es))   # back to original scale

from sklearn.metrics import mean_absolute_error, mean_squared_error
gamma_mae  = mean_absolute_error(y_es_true, gamma_pred)
cb_mae     = mean_absolute_error(y_es_true, cb_sev_pred)
gamma_rmse = np.sqrt(mean_squared_error(y_es_true, gamma_pred))
cb_rmse    = np.sqrt(mean_squared_error(y_es_true, cb_sev_pred))

print(f"      Gamma GLM  — MAE: €{gamma_mae:,.0f}  RMSE: €{gamma_rmse:,.0f}")
print(f"      CatBoost   — MAE: €{cb_mae:,.0f}  RMSE: €{cb_rmse:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
print("[7/8] Figure 5 — Process analytics: Lorenz + reserves …")

# Frequency model on full dataset (for pure premium)
print("      Fitting frequency models …")
from claims.process.severity import poisson_glm_pipeline, catboost_frequency
train_pp, test_pp = split_dataset(df, target="HasClaim", test_size=0.2)

glm_freq = poisson_glm_pipeline()
glm_freq.fit(train_pp[ALL_ENG_FEATURES], train_pp["ClaimFrequency"].values)
glm_freq_pred = glm_freq.predict(test_pp[ALL_ENG_FEATURES])

cb_freq = catboost_frequency(iterations=400)
cb_freq.fit(
    train_pp[ALL_ENG_FEATURES], train_pp["ClaimFrequency"].values,
    eval_set=(test_pp[ALL_ENG_FEATURES], test_pp["ClaimFrequency"].values),
    cat_features=CATEGORICAL_FEATURES,
    verbose=False,
)
cb_freq_pred = cb_freq.predict(test_pp[ALL_ENG_FEATURES])

# Quantile regression for reserves
print("      Fitting quantile models …")
from claims.process.reserving import quantile_model
quantiles = [0.50, 0.75, 0.95]
q_preds = {}
for q in quantiles:
    qm = quantile_model(quantile=q, iterations=300)
    qm.fit(
        X_ts, train_s["AvgSeverity"].values,
        eval_set=(X_es, y_es_true),
        cat_features=CATEGORICAL_FEATURES,
        verbose=False,
    )
    q_preds[q] = qm.predict(X_es)

with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 5a — Lorenz curve
    ax = axes[0]
    cum_exp_glm, cum_cls_glm = lorenz_curve(test_pp["HasClaim"].values, glm_freq_pred)
    cum_exp_cb,  cum_cls_cb  = lorenz_curve(test_pp["HasClaim"].values, cb_freq_pred)
    gini_glm = gini_coefficient(test_pp["HasClaim"].values, glm_freq_pred)
    gini_cb  = gini_coefficient(test_pp["HasClaim"].values, cb_freq_pred)
    ax.plot(cum_exp_glm, cum_cls_glm, lw=2, label=f"Poisson GLM (Gini={gini_glm:.3f})")
    ax.plot(cum_exp_cb,  cum_cls_cb,  lw=2, label=f"CatBoost   (Gini={gini_cb:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random model")
    ax.fill_between(cum_exp_cb, cum_exp_cb, cum_cls_cb, alpha=0.08)
    ax.set_xlabel("Cumulative Exposure Share")
    ax.set_ylabel("Cumulative Claims Share")
    ax.set_title("Lorenz Curve — Actuarial Lift\n(Frequency Model)", fontweight="bold")
    ax.legend(fontsize=9)

    # 5b — Severity: actual vs predicted scatter
    ax = axes[1]
    sample_idx = np.random.default_rng(0).integers(0, len(y_es_true),
                                                    min(500, len(y_es_true)))
    ax.scatter(gamma_pred[sample_idx], y_es_true[sample_idx],
               alpha=0.3, s=8, label=f"GLM  MAE=€{gamma_mae:,.0f}", color="#4472C4")
    ax.scatter(cb_sev_pred[sample_idx], y_es_true[sample_idx],
               alpha=0.3, s=8, label=f"CatBoost MAE=€{cb_mae:,.0f}", color="#ED7D31")
    lim = np.percentile(y_es_true, 95)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Predicted Severity (€)")
    ax.set_ylabel("Actual Severity (€)")
    ax.set_title("Severity Model: Actual vs Predicted\n(Gamma GLM vs CatBoost)", fontweight="bold")
    ax.legend(fontsize=8)

    # 5c — Reserve quantile bands
    ax = axes[2]
    sort_idx = np.argsort(q_preds[0.50])
    x_plot = np.arange(len(sort_idx))
    y_actual_sorted = y_es_true[sort_idx]
    ax.fill_between(x_plot, q_preds[0.50][sort_idx], q_preds[0.95][sort_idx],
                    alpha=0.25, color="#4472C4", label="P50–P95 band")
    ax.fill_between(x_plot, q_preds[0.50][sort_idx], q_preds[0.75][sort_idx],
                    alpha=0.40, color="#4472C4", label="P50–P75 band")
    ax.plot(x_plot, q_preds[0.50][sort_idx], lw=1.5, color="#4472C4", label="P50 reserve")
    ax.scatter(x_plot[::5], y_actual_sorted[::5], s=3, color="#ED7D31",
               alpha=0.4, label="Actual claim (sample)")
    ax.set_xlim(0, len(sort_idx))
    ax.set_ylim(0, np.percentile(y_es_true, 97))
    ax.set_xlabel("Policies (sorted by P50 reserve)")
    ax.set_ylabel("Claim Amount (€)")
    ax.set_title("Reserve Estimates with Uncertainty\n(P50 / P75 / P95 Quantile Regression)",
                 fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 4 — Process Analytics: Actuarial Lift, Severity & Reserves (Module 3)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_process_analytics.png", bbox_inches="tight")
    plt.close()
print("      Saved 04_process_analytics.png")

# ─────────────────────────────────────────────────────────────────────────────
print("[8/8] Figure 5 — Feature importance & model metrics summary …")

# CatBoost feature importances
feat_imp = pd.Series(
    cb_clf.get_feature_importance(),
    index=ALL_ENG_FEATURES,
).sort_values(ascending=True).tail(10)

with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature importance
    ax = axes[0]
    bars = ax.barh(feat_imp.index, feat_imp.values, color="#4472C4", edgecolor="white")
    ax.set_title("CatBoost Feature Importance\n(Claim Occurrence Model)",
                 fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=9)

    # Summary metrics table across all modules
    ax = axes[1]
    ax.axis("off")
    _m_lr   = classification_report_dict(y_test_clf, lr_prob)
    _m_beta = classification_report_dict(y_test_clf, cb_prob_beta)
    _ece_beta = calibration_metrics(y_test_clf, cb_prob_beta)
    summary_data = [
        ["Module 1", "Logistic Reg (baseline)", "ROC-AUC",     f"{_m_lr['roc_auc']:.4f}"],
        ["Module 1", "Logistic Reg (baseline)", "Gini",         f"{_m_lr['gini']:.4f}"],
        ["Module 1", "CatBoost + Beta cal.",    "ROC-AUC",     f"{_m_beta['roc_auc']:.4f}"],
        ["Module 1", "CatBoost + Beta cal.",    "Gini",         f"{_m_beta['gini']:.4f}"],
        ["Module 1", "CatBoost + Beta cal.",    "Brier Score",  f"{_m_beta['brier_score']:.4f}"],
        ["Module 1", "CatBoost + Beta cal.",    "ECE",          f"{_ece_beta['ece']:.4f}"],
        ["Module 1", "CatBoost + Beta cal.",    "Cohen Kappa",  f"{_m_beta['cohen_kappa']:.4f}"],
        ["Module 2", "CatBoost cost-sensitive", "Opt threshold",f"{opt_t:.3f}"],
        ["Module 3", "Gamma GLM",               "Severity MAE", f"€{gamma_mae:,.0f}"],
        ["Module 3", "CatBoost log-target",     "Severity MAE", f"€{cb_mae:,.0f}"],
        ["Module 3", "CatBoost Poisson freq.",  "Gini",         f"{gini_cb:.4f}"],
    ]
    tbl = ax.table(
        cellText=summary_data,
        colLabels=["Module", "Model", "Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF3FB")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Results Summary — All Modules", fontweight="bold", pad=12)

    fig.suptitle("Figure 5 — Feature Importance & Portfolio Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_summary.png", bbox_inches="tight")
    plt.close()
print("      Saved 05_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
print("[SHAP] Figure 6 — CatBoost SHAP explainability (beeswarm + waterfall) …")
try:
    import shap

    shap_n = min(500, len(X_test_clf))
    rng_s  = np.random.default_rng(0)
    s_idx  = rng_s.integers(0, len(X_test_clf), shap_n)
    X_shap = X_test_clf.iloc[s_idx].copy()

    shap_explainer = shap.TreeExplainer(cb_clf)
    shap_vals      = shap_explainer(X_shap)

    # 6a — Beeswarm: global feature impact across 500 test samples
    shap.plots.beeswarm(shap_vals, max_display=12, show=False)
    plt.gcf().suptitle(
        "Figure 6a — SHAP Beeswarm: Global Feature Impact\n"
        "(CatBoost Claim Occurrence · 500 test samples)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.savefig(FIG_DIR / "06_shap_beeswarm.png", bbox_inches="tight", dpi=130)
    plt.close()

    # 6b — Waterfall: feature contributions for the highest-risk prediction
    hr_local = int(np.argmax(cb_prob_raw[s_idx]))
    shap.plots.waterfall(shap_vals[hr_local], max_display=10, show=False)
    plt.gcf().suptitle(
        f"Figure 6b — SHAP Waterfall: Highest-Risk Prediction "
        f"(p={float(cb_prob_raw[s_idx[hr_local]]):.3f})\n"
        "(positive = raises claim probability, negative = lowers it)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.savefig(FIG_DIR / "06_shap_waterfall.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("      Saved 06_shap_beeswarm.png + 06_shap_waterfall.png")

except Exception as e:
    print(f"      SHAP figures skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\nDone! Figures saved to reports/figures/:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")
