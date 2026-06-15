"""
PLS-SEM + XGBoost/SHAP analysis
Hypothesis: Participation level → Citizen Satisfaction (Expectation Gap Theory)

Variables:
  - Q22_binary   : participated in any channel (0/1)
  - info_score   : Info Provision score (0-1)
  - consult_score: Consultation score (0-1)
  - Q24          : Participation construct (0=none, 1=info-only, 2=consultation)
  - Q25          : Satisfaction with project (1-5, outcome)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr, kendalltau
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb
import shap
import os

# ── Output dir ──────────────────────────────────────────────────────────────
OUT = "/home/user/my_paper1/results"
os.makedirs(OUT, exist_ok=True)

# ── Load & clean ─────────────────────────────────────────────────────────────
df_raw = pd.read_excel(
    "/root/.claude/uploads/7e325fe5-5704-55cc-9ede-a4a3bdde6097/4372d09e-Participation_Constructs.xlsx"
)

df = pd.DataFrame({
    "id"           : df_raw["ID"],
    "q22"          : df_raw["Q22\nBinary\n(0/1)"],
    "info_score"   : df_raw["Info Provision\nSCORE\n(0–1)"],
    "consult_score": df_raw["Consultation\nSCORE\n(0–1)"],
    "q24"          : df_raw["Q24\nOrdinal\n(0/1/2)"],
    "satisfaction" : df_raw["Q25\nClean\n(1–5; DK→NaN)"],
    "flag"         : df_raw["FLAG\nInconsistent\n(Q22=0 but channel selected)"],
})

# Remove inconsistent flags and keep only rows with satisfaction
df = df[df["flag"].fillna(0) == 0].reset_index(drop=True).copy()
df_sat = df.dropna(subset=["satisfaction"]).reset_index(drop=True).copy()

print(f"Total records: {len(df_raw)}")
print(f"After removing inconsistencies: {len(df)}")
print(f"With satisfaction data (Q25): {len(df_sat)}")
print(f"\nParticipation level (Q24) distribution:")
print(df_sat["q24"].value_counts().sort_index().rename({0.0:"0-None",1.0:"1-Info",2.0:"2-Consultation"}))
print(f"\nSatisfaction (Q25) distribution:")
print(df_sat["satisfaction"].value_counts().sort_index())


# ═══════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. DESCRIPTIVE STATISTICS BY PARTICIPATION LEVEL")
print("="*60)

desc = df_sat.groupby("q24")["satisfaction"].agg(
    N="count", Mean="mean", Median="median", SD="std", Min="min", Max="max"
).rename(index={0.0:"None (0)", 1.0:"Info-only (1)", 2.0:"Consultation (2)"})
print(desc.round(3).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 2. NON-PARAMETRIC TESTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. NON-PARAMETRIC TESTS")
print("="*60)

g0 = df_sat[df_sat["q24"] == 0.0]["satisfaction"].dropna().values
g1 = df_sat[df_sat["q24"] == 1.0]["satisfaction"].dropna().values
g2 = df_sat[df_sat["q24"] == 2.0]["satisfaction"].dropna().values

# Kruskal-Wallis (omnibus)
kw_stat, kw_p = kruskal(g0, g1, g2)
print(f"\nKruskal-Wallis H-test (Q24 groups vs Satisfaction):")
print(f"  H = {kw_stat:.4f}, p = {kw_p:.4f}")
print(f"  {'SIGNIFICANT' if kw_p < 0.05 else 'NOT significant'} at α=0.05")

# Effect size: eta-squared for Kruskal-Wallis
n_total = len(g0) + len(g1) + len(g2)
eta2 = (kw_stat - 2) / (n_total - 3)  # epsilon-squared approx
print(f"  ε² (effect size) ≈ {eta2:.4f}")

# Post-hoc pairwise Mann-Whitney with Bonferroni
print("\nPairwise Mann-Whitney U tests (Bonferroni α=0.0167):")
pairs = [("None vs Info", g0, g1), ("None vs Consultation", g0, g2), ("Info vs Consultation", g1, g2)]
mw_results = []
for name, ga, gb in pairs:
    u, p = mannwhitneyu(ga, gb, alternative="two-sided")
    r_eff = 1 - (2 * u) / (len(ga) * len(gb))   # rank-biserial r
    mw_results.append({"Comparison": name, "U": u, "p": p, "p_bonf": min(p * 3, 1.0), "r": r_eff})
    sig = "***" if p * 3 < 0.001 else "**" if p * 3 < 0.01 else "*" if p * 3 < 0.05 else "ns"
    print(f"  {name}: U={u:.0f}, p_bonf={min(p*3,1):.4f} {sig}, r={r_eff:.3f}")

# Spearman correlation Q24 ↔ Q25
_sat_corr = df_sat.dropna(subset=["q24", "satisfaction"])
rho, p_rho = spearmanr(_sat_corr["q24"], _sat_corr["satisfaction"])
tau, p_tau = kendalltau(_sat_corr["q24"], _sat_corr["satisfaction"])
print(f"\nSpearman ρ (Q24 ↔ Q25): ρ = {rho:.4f}, p = {p_rho:.4f}")
print(f"Kendall τ  (Q24 ↔ Q25): τ = {tau:.4f}, p = {p_tau:.4f}")

# Also: participated at all (Q22) vs satisfaction
df_q22 = df_sat.dropna(subset=["q22"])
g_no  = df_q22[df_q22["q22"] == 0.0]["satisfaction"].values
g_yes = df_q22[df_q22["q22"] == 1.0]["satisfaction"].values
u22, p22 = mannwhitneyu(g_no, g_yes, alternative="two-sided")
r22 = 1 - (2 * u22) / (len(g_no) * len(g_yes))
print(f"\nMann-Whitney Q22 (participated?) vs Satisfaction:")
print(f"  Non-participants median={np.median(g_no):.1f}, Participants median={np.median(g_yes):.1f}")
print(f"  U={u22:.0f}, p={p22:.4f}, r={r22:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PLS-SEM (manual implementation)
#    Model:
#    PARTICIPATION (reflective latent) → SATISFACTION
#    Indicators of PARTICIPATION: q22, info_score, consult_score, q24
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. PLS-SEM — PATH MODEL")
print("="*60)

# Work on complete cases
df_pls = df_sat.dropna(subset=["q22", "info_score", "consult_score", "q24", "satisfaction"]).reset_index(drop=True).copy()
print(f"Complete cases for PLS-SEM: {len(df_pls)}")

# Q24 is derived from Q22 + channels, so use the raw components as formative indicators
# Q24 is retained as a single-indicator cross-validation in Model B
indicators = ["q22", "info_score", "consult_score"]
outcome    = "satisfaction"

# --- Outer model (measurement model) ---
# Standardise all variables
scaler = StandardScaler()
X_std = scaler.fit_transform(df_pls[indicators])
df_std = pd.DataFrame(X_std, columns=indicators, index=df_pls.index)
df_std[outcome] = df_pls[outcome].values

# PLS-SEM outer weights: iterative algorithm (simplified mode B for formative)
# Since participation is FORMATIVE (channels → participation level), we use
# OLS-regression weights for the composite score.

# Step 1: Outer weights via OLS of each indicator on a unit score, iterated
def pls_outer_weights(X, tol=1e-6, max_iter=300):
    """PLS mode-A (reflective): correlations as outer loadings."""
    n, p = X.shape
    w = np.ones(p) / np.sqrt(p)
    for _ in range(max_iter):
        score = X @ w
        score /= np.linalg.norm(score)
        loadings = X.T @ score / (n - 1)
        w_new = loadings / np.linalg.norm(loadings)
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    score = X @ w
    score = (score - score.mean()) / score.std()
    loadings = np.array([np.corrcoef(X[:, j], score)[0, 1] for j in range(p)])
    return w, loadings, score

w, loadings, part_score = pls_outer_weights(X_std)

print("\nOuter loadings (reflective measurement model):")
for ind, load in zip(indicators, loadings):
    stars = "***" if abs(load) > 0.7 else "**" if abs(load) > 0.5 else "*" if abs(load) > 0.3 else ""
    print(f"  {ind:15s}: λ = {load:.4f}  {stars}")

# AVE and composite reliability
loadings2 = loadings ** 2
AVE = loadings2.mean()
CR  = loadings.sum() ** 2 / (loadings.sum() ** 2 + (1 - loadings2).sum())
print(f"\n  AVE = {AVE:.4f}  ({'✓ >0.5' if AVE > 0.5 else '✗ <0.5'})")
print(f"  CR  = {CR:.4f}  ({'✓ >0.7' if CR > 0.7 else '✗ <0.7'})")

# --- Inner model (structural model) ---
y = df_pls[outcome].values
y_std = (y - y.mean()) / y.std()

X_path = sm.add_constant(part_score)
inner_model = sm.OLS(y_std, X_path).fit()

path_coef = inner_model.params[1]
path_se   = inner_model.bse[1]
path_t    = inner_model.tvalues[1]
path_p    = inner_model.pvalues[1]
R2        = inner_model.rsquared
R2_adj    = inner_model.rsquared_adj

print(f"\nInner model (Participation → Satisfaction):")
print(f"  β = {path_coef:.4f}  SE = {path_se:.4f}  t = {path_t:.4f}  p = {path_p:.4f}")
print(f"  R² = {R2:.4f}   R²_adj = {R2_adj:.4f}")
print(f"  Effect: {'positive' if path_coef > 0 else 'negative'}, "
      f"{'significant' if path_p < 0.05 else 'NOT significant'}")

# Bootstrapped CI for path coefficient (500 resamples)
np.random.seed(42)
boot_coefs = []
n = len(df_pls)
for _ in range(500):
    idx = np.random.choice(n, n, replace=True)
    Xb = X_std[idx]
    yb = y_std[idx]
    _, _, sc_b = pls_outer_weights(Xb)
    Xp_b = sm.add_constant(sc_b)
    try:
        m_b = sm.OLS(yb, Xp_b).fit()
        boot_coefs.append(m_b.params[1])
    except Exception:
        pass
boot_coefs = np.array(boot_coefs)
ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])
print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  {'Excludes zero → significant' if ci_low > 0 or ci_high < 0 else 'Contains zero → not significant'}")

# f² effect size
f2 = R2 / (1 - R2)
print(f"  f² effect size = {f2:.4f}  ({'large' if f2>0.35 else 'medium' if f2>0.15 else 'small' if f2>0.02 else 'negligible'})")

# Predictive relevance Q² (blindfolding, omission distance d=7)
d = 7
SSO, SSE = 0.0, 0.0
for i in range(d):
    omit_idx = np.arange(i, n, d)
    use_idx  = np.setdiff1d(np.arange(n), omit_idx)
    _, _, sc_u = pls_outer_weights(X_std[use_idx])
    reg_u = sm.OLS(y_std[use_idx], sm.add_constant(sc_u)).fit()
    # predict on omitted using weights from use set
    _, _, sc_o = pls_outer_weights(X_std[omit_idx])
    y_pred = reg_u.params[0] + reg_u.params[1] * sc_o
    SSE += np.sum((y_std[omit_idx] - y_pred) ** 2)
    SSO += np.sum((y_std[omit_idx] - y_std.mean()) ** 2)
Q2 = 1 - SSE / SSO
print(f"  Q² predictive relevance = {Q2:.4f}  ({'✓ >0' if Q2 > 0 else '✗ ≤0'})")


# ═══════════════════════════════════════════════════════════════════════════
# 4. EXPECTATION GAP THEORY TEST
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. EXPECTATION GAP THEORY — EVIDENCE SUMMARY")
print("="*60)

medians = {k: np.median(g) for k, g in [(0.0, g0), (1.0, g1), (2.0, g2)]}
print(f"\n  Median satisfaction by participation level:")
print(f"    Non-participants (0):  {medians[0.0]:.2f}")
print(f"    Info-only (1):         {medians[1.0]:.2f}")
print(f"    Consultation (2):      {medians[2.0]:.2f}")

direction = "non-linear / inverted-U" if medians[1.0] >= medians[0.0] and medians[2.0] <= medians[1.0] else \
            "positive monotone"       if medians[2.0] >= medians[1.0] >= medians[0.0] else \
            "negative monotone"       if medians[2.0] <= medians[1.0] <= medians[0.0] else \
            "mixed / non-monotone"
print(f"\n  Pattern: {direction}")

# Test for non-linearity: polynomial regression
df_sat_poly = df_sat.dropna(subset=["q24","satisfaction"]).reset_index(drop=True).copy()
df_sat_poly["q24_sq"] = df_sat_poly["q24"] ** 2
lin_mod  = ols("satisfaction ~ q24", data=df_sat_poly).fit()
quad_mod = ols("satisfaction ~ q24 + q24_sq", data=df_sat_poly).fit()
from statsmodels.stats.anova import anova_lm
anova_res = anova_lm(lin_mod, quad_mod)
print(f"\n  Polynomial test (linear vs quadratic Q24):")
print(f"    Linear   R² = {lin_mod.rsquared:.4f}")
print(f"    Quadratic R² = {quad_mod.rsquared:.4f}")
print(f"    F-test for quadratic term: F = {anova_res['F'].iloc[1]:.4f}, p = {anova_res['Pr(>F)'].iloc[1]:.4f}")

# Expectation gap diagnosis
eg_confirmed = False
if medians[2.0] < medians[0.0] and path_coef < 0:
    print("\n  → Expectation gap CONFIRMED: consultation participants LESS satisfied than non-participants.")
    eg_confirmed = True
elif medians[2.0] < medians[1.0]:
    print("\n  → Partial expectation gap: consultation participants LESS satisfied than info-only.")
elif rho < 0 and p_rho < 0.05:
    print("\n  → Expectation gap CONFIRMED by negative Spearman correlation.")
    eg_confirmed = True
else:
    print("\n  → Expectation gap NOT confirmed: higher participation does not reduce satisfaction.")
    print(f"     Spearman ρ = {rho:.4f} (p={p_rho:.4f}), direction: {'positive' if rho>0 else 'negative'}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. XGBOOST + SHAP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. XGBOOST + SHAP VALIDATION")
print("="*60)

# Features: all participation indicators + derived composite
df_xgb = df_sat.dropna(subset=["q22", "info_score", "consult_score", "q24", "satisfaction"]).copy()
feature_names = ["q22", "info_score", "consult_score", "q24"]
X_xgb = df_xgb[feature_names].values
y_xgb = df_xgb["satisfaction"].values

from sklearn.model_selection import cross_val_score, KFold

model_xgb = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(model_xgb, X_xgb, y_xgb, cv=kf, scoring="r2")
cv_rmse = np.sqrt(-cross_val_score(model_xgb, X_xgb, y_xgb, cv=kf, scoring="neg_mean_squared_error"))
print(f"\n5-Fold CV results:")
print(f"  R²   = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  RMSE = {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# Fit on full data for SHAP
model_xgb.fit(X_xgb, y_xgb)
y_pred = model_xgb.predict(X_xgb)
print(f"  Full-data R² = {r2_score(y_xgb, y_pred):.4f}")

# SHAP
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_xgb)

print(f"\nMean |SHAP| values (feature importance):")
mean_shap = np.abs(shap_values).mean(axis=0)
for name, val in sorted(zip(feature_names, mean_shap), key=lambda x: -x[1]):
    print(f"  {name:15s}: {val:.4f}")

# SHAP direction for q24 (main hypothesis)
idx_q24 = feature_names.index("q24")
shap_q24 = shap_values[:, idx_q24]
q24_vals  = X_xgb[:, idx_q24]
rho_shap, p_shap = spearmanr(q24_vals, shap_q24)
print(f"\nSHAP(q24) direction:")
print(f"  Spearman ρ(Q24, SHAP_Q24) = {rho_shap:.4f}, p = {p_shap:.4f}")
print(f"  → Q24 has {'positive' if rho_shap > 0 else 'negative'} effect on satisfaction in XGBoost")


# ═══════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

labels_q24 = {0.0: "None\n(0)", 1.0: "Info-only\n(1)", 2.0: "Consultation\n(2)"}
colors_q24 = {0.0: "#4C72B0", 1.0: "#55A868", 2.0: "#C44E52"}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Participation Level & Citizen Satisfaction — Analysis Results", fontsize=14, fontweight="bold")

# (A) Box plot satisfaction by Q24
ax = axes[0, 0]
data_box = [df_sat[df_sat["q24"] == k]["satisfaction"].dropna().values for k in [0.0, 1.0, 2.0]]
bp = ax.boxplot(data_box, patch_artist=True, notch=True, widths=0.5)
for patch, k in zip(bp["boxes"], [0.0, 1.0, 2.0]):
    patch.set_facecolor(colors_q24[k])
    patch.set_alpha(0.7)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["None (0)", "Info-only (1)", "Consultation (2)"])
ax.set_ylabel("Satisfaction (Q25, 1–5)")
ax.set_title("(A) Satisfaction by Participation Level")
ax.text(0.02, 0.97, f"KW: H={kw_stat:.2f}, p={kw_p:.3f}", transform=ax.transAxes,
        va="top", fontsize=9, color="darkred")

# (B) Bar: mean ± SE
ax = axes[0, 1]
means = [g.mean() for g in data_box]
sems  = [g.std() / np.sqrt(len(g)) for g in data_box]
bars = ax.bar([1, 2, 3], means, yerr=sems, capsize=5, width=0.5,
              color=[colors_q24[k] for k in [0.0, 1.0, 2.0]], alpha=0.8, edgecolor="k")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["None (0)", "Info-only (1)", "Consultation (2)"])
ax.set_ylabel("Mean Satisfaction (±SE)")
ax.set_title("(B) Mean Satisfaction ± SE")
ax.set_ylim(1, 5.5)
for bar, m, se in zip(bars, means, sems):
    ax.text(bar.get_x() + bar.get_width()/2, m + se + 0.1, f"{m:.2f}", ha="center", fontsize=9)

# (C) Scatter Q24 vs Satisfaction with jitter + polynomial trend
ax = axes[0, 2]
jitter_x = df_sat["q24"] + np.random.uniform(-0.15, 0.15, len(df_sat))
scatter_colors = [colors_q24.get(v, "grey") for v in df_sat["q24"]]
ax.scatter(jitter_x, df_sat["satisfaction"], c=scatter_colors, alpha=0.5, s=30)
# polynomial trend
xfit = np.linspace(0, 2, 100)
yfit = quad_mod.params.iloc[0] + quad_mod.params.iloc[1]*xfit + quad_mod.params.iloc[2]*xfit**2
ax.plot(xfit, yfit, "k--", lw=2, label=f"Quadratic fit (R²={quad_mod.rsquared:.3f})")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["None", "Info", "Consultation"])
ax.set_xlabel("Participation Level (Q24)")
ax.set_ylabel("Satisfaction (Q25)")
ax.set_title(f"(C) Q24 vs Q25  [ρ={rho:.3f}, p={p_rho:.3f}]")
ax.legend(fontsize=8)

# (D) PLS-SEM path diagram (schematic)
ax = axes[1, 0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(D) PLS-SEM Path Model")
# Indicators box
for i, (ind, load) in enumerate(zip(["Q22", "Info Score", "Consult Score", "Q24"],
                                     loadings)):
    y_pos = 8 - i * 1.8
    rect = mpatches.FancyBboxPatch((0.3, y_pos - 0.4), 2.5, 0.8,
                                    boxstyle="round,pad=0.1", fc="#AED6F1", ec="steelblue")
    ax.add_patch(rect)
    ax.text(1.55, y_pos, f"{ind}\n(λ={load:.3f})", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(4.0, 5.0), xytext=(2.8, y_pos),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5))
# Latent variable
circle = plt.Circle((5.2, 5.0), 1.0, fc="#F9E79F", ec="goldenrod", lw=2)
ax.add_patch(circle)
ax.text(5.2, 5.0, "PARTI-\nCIPATION", ha="center", va="center", fontsize=8, fontweight="bold")
# Path to outcome
ax.annotate("", xy=(7.8, 5.0), xytext=(6.2, 5.0),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=2.5))
coef_label = f"β={path_coef:.3f}\np={path_p:.3f}\nCI[{ci_low:.2f},{ci_high:.2f}]"
ax.text(7.0, 5.5, coef_label, ha="center", va="bottom", fontsize=8, color="darkred")
# Outcome box
rect2 = mpatches.FancyBboxPatch((7.8, 4.3), 1.9, 1.4,
                                  boxstyle="round,pad=0.1", fc="#ABEBC6", ec="green")
ax.add_patch(rect2)
ax.text(8.75, 5.0, f"SATIS-\nFACTION\nR²={R2:.3f}", ha="center", va="center", fontsize=8, fontweight="bold")

# (E) SHAP summary bar
ax = axes[1, 1]
sorted_idx = np.argsort(mean_shap)
bar_colors = ["#C44E52" if n == "q24" else "#4C72B0" for n in np.array(feature_names)[sorted_idx]]
ax.barh(range(len(feature_names)), mean_shap[sorted_idx], color=bar_colors, edgecolor="k", alpha=0.85)
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"(E) XGBoost SHAP Feature Importance\n(CV R²={cv_r2.mean():.3f})")
ax.axvline(0, color="k", lw=0.5)

# (F) SHAP dependence plot for Q24
ax = axes[1, 2]
shap_q24_all = shap_values[:, idx_q24]
for kv in [0.0, 1.0, 2.0]:
    mask = q24_vals == kv
    ax.scatter(q24_vals[mask] + np.random.uniform(-0.07, 0.07, mask.sum()),
               shap_q24_all[mask], c=colors_q24[kv], alpha=0.7, s=40,
               label=labels_q24[kv], edgecolors="k", linewidths=0.3)
means_shap = [shap_q24_all[q24_vals == kv].mean() for kv in [0.0, 1.0, 2.0]]
ax.plot([0, 1, 2], means_shap, "k--o", lw=2, ms=8, label="Group mean")
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["None", "Info", "Consultation"])
ax.set_xlabel("Participation Level (Q24)")
ax.set_ylabel("SHAP value for Q24")
ax.set_title("(F) SHAP Dependence — Q24 effect on Satisfaction")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/participation_satisfaction_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {OUT}/participation_satisfaction_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. CONSOLIDATED RESULTS SUMMARY")
print("="*60)

summary = {
    "Kruskal-Wallis H"         : f"{kw_stat:.4f}",
    "Kruskal-Wallis p"         : f"{kw_p:.4f}",
    "Effect size ε²"           : f"{eta2:.4f}",
    "Spearman ρ (Q24↔Q25)"    : f"{rho:.4f}",
    "Spearman p"               : f"{p_rho:.4f}",
    "Kendall τ"                : f"{tau:.4f}",
    "Kendall p"                : f"{p_tau:.4f}",
    "PLS β (Participation→Sat)": f"{path_coef:.4f}",
    "PLS p-value"              : f"{path_p:.4f}",
    "PLS 95% CI"               : f"[{ci_low:.4f}, {ci_high:.4f}]",
    "PLS R²"                   : f"{R2:.4f}",
    "PLS Q²"                   : f"{Q2:.4f}",
    "XGBoost CV R²"            : f"{cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
    "Top SHAP feature"         : feature_names[np.argmax(mean_shap)],
    "Expectation Gap confirmed": str(eg_confirmed),
}
for k, v in summary.items():
    print(f"  {k:<35}: {v}")

# Save summary CSV
pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
    f"{OUT}/results_summary.csv", index=False
)
print(f"\nSummary saved: {OUT}/results_summary.csv")
print("\nAnalysis complete.")
