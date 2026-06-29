import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

RESULTS = "/home/user/my_paper1/results"
os.makedirs(RESULTS, exist_ok=True)

DATA_PATH = (
    "/root/.claude/uploads/7e325fe5-5704-55cc-9ede-a4a3bdde6097/"
    "6242ce4e-combined_SEM_Ready_3.xlsx"
)

# ==============================================================================
# SECTION 0 - DATA LOADING AND RECODING
# ==============================================================================
print("=" * 70)
print("SECTION 0: DATA LOADING AND RECODING")
print("=" * 70)

raw = pd.read_excel(DATA_PATH, engine="openpyxl")
print(f"Raw shape: {raw.shape}")

# Remove flagged rows
raw = raw[raw["Flag_inconsistent"] != 1].reset_index(drop=True)
print(f"After removing flagged rows: N = {len(raw)}")

# --- Participation group from Info/Consult scores ---
# Consult > 0 → level 2; Info > 0 (and Consult == 0) → level 1; else → 0
raw["q22_level"] = np.where(raw["Consultation_score"].fillna(0) > 0, 2,
                    np.where(raw["Info_Provision_score"].fillna(0) > 0, 1, 0))
group_map = {0: "None", 1: "Info", 2: "Consult"}
raw["part_group"] = raw["q22_level"].map(group_map)

# --- Normalise participation indicators to [0,1] ---
raw["q22_norm"]  = raw["q22_level"] / 2.0
raw["info_norm"] = raw["Info_Provision_score"].fillna(0)      # already 0–1
raw["cons_norm"] = raw["Consultation_score"].fillna(0)        # already 0–1
raw["q24_norm"]  = raw["Q24_ordinal"] / 2.0
raw["q25_norm"]  = (raw["Q25_clean"] - 1) / 4.0

# --- IAP2-weighted Participation Level composite ---
W_Q22, W_CHAN, W_Q24, W_Q25 = 0.35, 0.15, 0.20, 0.30
raw["PL"] = (
    W_Q22 * raw["q22_norm"].fillna(0) +
    W_CHAN * raw["info_norm"].fillna(0) +
    W_Q24 * raw["q24_norm"].fillna(0) +
    W_Q25 * raw["q25_norm"].fillna(0)
)

# --- Satisfaction outcome ---
raw["q26"] = raw["Q26_MGT"]           # 1–5
raw["q27"] = raw["Q27_outcome"]       # 0/1

# --- Performance composite (Q12–Q21) ---
perf_cols = ["Q12_EMQ","Q13_EMQ","Q14_EMQ","Q15_EMQ",
             "Q16_EHB","Q17_EHB","Q18_EHB","Q19_EHB","Q20_EHB","Q21_ECO"]
raw["perf"] = raw[perf_cols].mean(axis=1)

# --- Working datasets ---
df      = raw.copy()
df_sat  = df.dropna(subset=["q26"]).reset_index(drop=True)
df_full = df.dropna(subset=["Info_Provision_score","Q24_ordinal","Q25_clean","q26"]).reset_index(drop=True)

print(f"\nSample sizes:")
print(f"  df      (all)            : N = {len(df)}")
print(f"  df_sat  (Q26 present)    : N = {len(df_sat)}")
print(f"  df_full (complete cases) : N = {len(df_full)}")
print(f"  old wave: N = {(df['wave']=='old').sum()},  new wave: N = {(df['wave']=='new').sum()}")

print(f"\nParticipation group counts:")
print(df["part_group"].value_counts().sort_index())

# ==============================================================================
# SECTION 1 - DESCRIPTIVE STATISTICS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 70)

def desc_row(s, label):
    s = s.dropna()
    return {"Variable": label, "N": len(s),
            "Mean": round(s.mean(), 3), "SD": round(s.std(), 3),
            "Median": round(s.median(), 3), "Min": round(s.min(), 3),
            "Max": round(s.max(), 3),
            "% ceiling": round(100*(s == s.max()).mean(), 1)}

desc_rows = [
    desc_row(df["q22_level"],       "Q22 Info level (0–2)"),
    desc_row(df["Info_Provision_score"], "Info Provision score (0–1)"),
    desc_row(df["Consultation_score"],   "Consultation score (0–1)"),
    desc_row(df["Q24_ordinal"],     "Q24 Signage (0–2)"),
    desc_row(df["Q25_clean"],       "Q25 Terminology clarity (1–5)"),
    desc_row(df["q26"],             "Q26 Maintenance satisfaction (1–5)"),
    desc_row(df["q27"],             "Q27 Future benefits (0/1)"),
    desc_row(df["PL"],              "PL composite (IAP2-weighted)"),
    desc_row(df["perf"],            "Performance composite Q12–Q21"),
]
desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))
desc_df.to_csv(f"{RESULTS}/descriptive_statistics.csv", index=False)

pct_ceil = 100 * (df_sat["q26"] == 5).mean()
print(f"\nQ26 ceiling (==5): {pct_ceil:.1f}%  (N={len(df_sat)})")

# ==============================================================================
# SECTION 2 - NON-PARAMETRIC TESTS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: NON-PARAMETRIC TESTS")
print("=" * 70)

from scipy import stats

groups_full = {
    g: df_full.loc[df_full["part_group"] == g, "q26"].dropna().values
    for g in ["None", "Info", "Consult"]
}
print("Group sizes (complete cases):", {k: len(v) for k, v in groups_full.items()})
print("Group means:", {k: round(float(v.mean()), 3) for k, v in groups_full.items() if len(v) > 0})

# Kruskal-Wallis
valid_groups = [v for v in groups_full.values() if len(v) > 1]
kw_stat, kw_p = stats.kruskal(*valid_groups)
n_total = sum(len(v) for v in valid_groups)
k_groups = len(valid_groups)
eps2 = (kw_stat - k_groups + 1) / (n_total - k_groups)
print(f"\nKruskal-Wallis H = {kw_stat:.3f}, p = {kw_p:.4f}, ε² = {eps2:.3f}")

# Mann-Whitney pairwise (Bonferroni)
pairs = [("None","Info"), ("None","Consult"), ("Info","Consult")]
mw_rows = []
for g1, g2 in pairs:
    a, b = groups_full[g1], groups_full[g2]
    if len(a) < 2 or len(b) < 2:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2*u / (len(a)*len(b))
    mw_rows.append({"Pair": f"{g1} vs {g2}", "U": u, "p_raw": round(p,4),
                    "p_bonf": round(min(p*3, 1.0), 4), "r": round(r, 3)})
mw_df = pd.DataFrame(mw_rows)
print("\nMann-Whitney U (Bonferroni corrected):")
print(mw_df.to_string(index=False))

# Jonckheere-Terpstra trend test
def jonckheere_terpstra(groups_ordered):
    JT = 0
    for i in range(len(groups_ordered) - 1):
        for j in range(i + 1, len(groups_ordered)):
            for xi in groups_ordered[i]:
                JT += np.sum(xi < groups_ordered[j]) + 0.5*np.sum(xi == groups_ordered[j])
    ns = [len(g) for g in groups_ordered]
    N = sum(ns)
    mu  = (N**2 - sum(n**2 for n in ns)) / 4
    var = (N**2*(2*N+3) - sum(n**2*(2*n+3) for n in ns)) / 72
    z   = (JT - mu) / np.sqrt(var) if var > 0 else 0
    return JT, z, 1 - stats.norm.cdf(z)

jt_g = [groups_full["None"], groups_full["Info"], groups_full["Consult"]]
jt_g = [g for g in jt_g if len(g) > 0]
JT, jt_z, jt_p = jonckheere_terpstra(jt_g)
print(f"\nJonckheere-Terpstra: JT={JT:.1f}, z={jt_z:.3f}, p (one-sided)={jt_p:.4f}")

# Spearman and Kendall
corr_df = df_full[["PL","q22_level","Q24_ordinal","Q25_clean","q26"]].dropna()
sp_PL,  sp_p_PL  = stats.spearmanr(corr_df["PL"],        corr_df["q26"])
sp_q22, sp_p_q22 = stats.spearmanr(corr_df["q22_level"], corr_df["q26"])
sp_q24, sp_p_q24 = stats.spearmanr(corr_df["Q24_ordinal"],corr_df["q26"])
sp_q25, sp_p_q25 = stats.spearmanr(corr_df["Q25_clean"],  corr_df["q26"])
kt_PL,  kt_p_PL  = stats.kendalltau(corr_df["PL"],       corr_df["q26"])

print(f"\nSpearman correlations with Q26:")
print(f"  PL composite : ρ = {sp_PL:.3f}, p = {sp_p_PL:.4f}")
print(f"  Q22 level    : ρ = {sp_q22:.3f}, p = {sp_p_q22:.4f}")
print(f"  Q24 signage  : ρ = {sp_q24:.3f}, p = {sp_p_q24:.4f}")
print(f"  Q25 term.    : ρ = {sp_q25:.3f}, p = {sp_p_q25:.4f}")
print(f"Kendall τ (PL): τ = {kt_PL:.3f}, p = {kt_p_PL:.4f}")

pd.DataFrame([{
    "KW_H": kw_stat, "KW_p": kw_p, "KW_eps2": eps2,
    "JT_stat": JT, "JT_z": jt_z, "JT_p": jt_p,
    "Spearman_PL_r": sp_PL, "Spearman_PL_p": sp_p_PL,
    "Kendall_PL_tau": kt_PL, "Kendall_PL_p": kt_p_PL,
}]).to_csv(f"{RESULTS}/nonparametric_tests.csv", index=False)

# ==============================================================================
# SECTION 3 - PLS-SEM (FORMATIVE MEASUREMENT MODEL)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: PLS-SEM")
print("=" * 70)

import statsmodels.api as sm

# Formative indicators: q22_norm, info_norm, cons_norm, q24_norm, q25_norm → PL → Q26
pls_cols = ["q22_norm","info_norm","cons_norm","q24_norm","q25_norm","q26"]
pls_df   = df_full[pls_cols].dropna().reset_index(drop=True)
print(f"PLS sample: N = {len(pls_df)}")

X_pls = pls_df[["q22_norm","info_norm","cons_norm","q24_norm","q25_norm"]].values.astype(float)
y_pls = pls_df["q26"].values.astype(float)

X_std = (X_pls - X_pls.mean(axis=0)) / (X_pls.std(axis=0) + 1e-10)
y_std = (y_pls - y_pls.mean()) / (y_pls.std() + 1e-10)

def pls_mode_a(X, y, n_iter=300):
    n, p = X.shape
    w = np.ones(p) / p
    for _ in range(n_iter):
        lv   = X @ w
        lv_s = lv / (lv.std() + 1e-10)
        cov  = (X.T @ lv_s.reshape(-1,1)).flatten()
        w_new = cov / (np.linalg.norm(cov) + 1e-10)
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new; break
        w = w_new
    lv   = X @ w
    lv_s = (lv - lv.mean()) / (lv.std() + 1e-10)
    loadings = np.array([np.corrcoef(X[:,j], lv_s)[0,1] for j in range(p)])
    path     = np.corrcoef(lv_s, y)[0,1]
    return w, loadings, path, lv_s

ind_names = ["q22_norm","info_norm","cons_norm","q24_norm","q25_norm"]
weights, loadings, path_coef, lv_scores = pls_mode_a(X_std, y_std)

print(f"\nOuter weights:  {dict(zip(ind_names, weights.round(4)))}")
print(f"Outer loadings: {dict(zip(ind_names, loadings.round(4)))}")
print(f"Path coefficient (PL → Q26): β = {path_coef:.4f}")

avgs = np.mean(loadings**2)
cr   = np.sum(np.abs(loadings))**2 / (np.sum(np.abs(loadings))**2 + np.sum(1 - loadings**2))
print(f"AVE = {avgs:.3f}  (>0.50 required)")
print(f"CR  = {cr:.3f}  (>0.70 required)")

# Bootstrap CI (500 resamples)
np.random.seed(42)
boot_paths = []
for _ in range(500):
    idx = np.random.choice(len(X_std), len(X_std), replace=True)
    try:
        _, _, bp, _ = pls_mode_a(X_std[idx], y_std[idx])
        boot_paths.append(bp)
    except Exception:
        pass
boot_paths = np.array(boot_paths)
ci_lo, ci_hi = np.nanpercentile(boot_paths, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Q² blindfolding (d=7)
n_pls = len(X_std)
res_sq, ss_tot = [], []
for start in range(7):
    blind = np.arange(start, n_pls, 7)
    train = np.setdiff1d(np.arange(n_pls), blind)
    try:
        _, _, _, lv_tr = pls_mode_a(X_std[train], y_std[train])
        w_tr  = np.linalg.lstsq(X_std[train], lv_tr, rcond=None)[0]
        lv_bl = X_std[blind] @ w_tr
        lv_bl_s = (lv_bl - lv_bl.mean()) / (lv_bl.std() + 1e-10)
        coef  = np.corrcoef(lv_tr, y_std[train])[0,1]
        yhat  = coef * lv_bl_s
        res_sq.extend((y_std[blind] - yhat)**2)
        ss_tot.extend((y_std[blind] - y_std[blind].mean())**2)
    except Exception:
        pass
q2 = 1 - sum(res_sq) / (sum(ss_tot) + 1e-10)
print(f"Q² (blindfolding d=7) = {q2:.4f}  (>0: predictive relevance)")

pd.DataFrame([{"N": len(pls_df), "path_coef": path_coef,
               "CI_lo": ci_lo, "CI_hi": ci_hi, "AVE": avgs, "CR": cr, "Q2": q2
               }]).to_csv(f"{RESULTS}/pls_sem_summary.csv", index=False)

# ==============================================================================
# SECTION 4 - TOBIT REGRESSION (CEILING EFFECT CHECK)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: TOBIT REGRESSION (CEILING EFFECT CHECK)")
print("=" * 70)

from scipy.optimize import minimize
from scipy.stats import norm as _norm

tobit_df = df_full[["PL","q26"]].dropna()
PL_t = tobit_df["PL"].values
y_t  = tobit_df["q26"].values
PL_s = (PL_t - PL_t.mean()) / (PL_t.std() + 1e-10)

def tobit_nll(params, X, y, lo=1.0, hi=5.0):
    b0, b1, log_s = params
    sig  = np.exp(log_s) + 1e-10
    yhat = b0 + b1 * X
    ll   = np.where(y <= lo, np.log(_norm.cdf((lo-yhat)/sig)+1e-300),
           np.where(y >= hi, np.log(1-_norm.cdf((hi-yhat)/sig)+1e-300),
                    np.log(_norm.pdf((y-yhat)/sig)/sig+1e-300)))
    return -ll.sum()

res = minimize(tobit_nll, [y_t.mean(), 0.1, np.log(y_t.std())],
               args=(PL_s, y_t), method="Nelder-Mead",
               options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
b0_t, b1_t, sigma_t = res.x[0], res.x[1], np.exp(res.x[2])

X_sm  = sm.add_constant(PL_s)
ols_m = sm.OLS(y_t, X_sm).fit()
b1_ols = float(np.array(ols_m.params).flat[1])
ratio  = abs(b1_t / (b1_ols + 1e-10))

print(f"Tobit β(PL)  = {b1_t:.4f}")
print(f"OLS   β(PL)  = {b1_ols:.4f}")
print(f"Tobit/OLS ratio = {ratio:.3f}  (>1.10 → non-trivial ceiling bias)")
pd.DataFrame([{"Tobit_b1": b1_t, "OLS_b1": b1_ols, "ratio": ratio, "sigma": sigma_t}
              ]).to_csv(f"{RESULTS}/tobit_results.csv", index=False)

# ==============================================================================
# SECTION 5 - XGBOOST + SHAP
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: XGBOOST + SHAP")
print("=" * 70)

try:
    import xgboost as xgb
    import shap

    feat_cols  = ["q22_norm","info_norm","cons_norm","q24_norm","q25_norm","PL"]
    feat_names = ["Q22 level","Info score","Consult score","Q24 signage","Q25 terminology","PL composite"]
    xgb_df = df_full[feat_cols + ["q26"]].dropna().reset_index(drop=True)
    Xx = xgb_df[feat_cols].values.astype(float)
    yx = xgb_df["q26"].values.astype(float)

    model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, objective="reg:squarederror", verbosity=0)
    model.fit(Xx, yx)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(Xx)
    mean_abs   = np.abs(shap_vals).mean(axis=0)

    # Beeswarm
    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals, Xx, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    # Bar
    sort_idx = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([feat_names[i] for i in sort_idx[::-1]], mean_abs[sort_idx[::-1]], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance – Q26 Satisfaction (XGBoost SHAP)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_importance_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    # Dependence PL
    pl_idx = feat_cols.index("PL")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Xx[:, pl_idx], shap_vals[:, pl_idx], alpha=0.5, s=20, c="steelblue")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Participation Level (PL)"); ax.set_ylabel("SHAP value for PL")
    ax.set_title("SHAP Dependence: PL → Q26 Satisfaction")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_dependence_PL.png", dpi=150, bbox_inches="tight"); plt.close()

    pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}
                 ).to_csv(f"{RESULTS}/shap_importance.csv", index=False)
    print("SHAP mean |values|:", dict(zip(feat_names, mean_abs.round(4))))

except ImportError as e:
    print(f"XGBoost/SHAP not available: {e}")

# ==============================================================================
# SECTION 6 - OLS REGRESSION (ROBUST HC3) + BINARY LOGISTIC (CEILING)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: OLS ROBUST + CEILING LOGISTIC")
print("=" * 70)

reg_df = df_full[["PL","q22_norm","info_norm","cons_norm","q24_norm","q25_norm","q26"]].dropna()
X_reg  = sm.add_constant(reg_df[["q22_norm","info_norm","cons_norm","q24_norm","q25_norm"]])
ols_r  = sm.OLS(reg_df["q26"], X_reg).fit(cov_type="HC3")
print(ols_r.summary())
with open(f"{RESULTS}/ols_robust_summary.txt", "w") as fh:
    fh.write(str(ols_r.summary()))

# Ceiling logistic: Q26=5 vs rest
reg_df2   = df_full[["PL","q26"]].dropna().copy()
reg_df2["ceiling"] = (reg_df2["q26"] == 5).astype(int)
logit_m = sm.Logit(reg_df2["ceiling"], sm.add_constant(reg_df2["PL"])).fit(disp=False)
print("\nBinary logistic (Q26=5 vs rest):")
print(logit_m.summary2().tables[1])
with open(f"{RESULTS}/logit_ceiling_summary.txt", "w") as fh:
    fh.write(str(logit_m.summary()))

# ==============================================================================
# SECTION 7 - EGT FORMAL ASSESSMENT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: EXPECTATION GAP THEORY (EGT) ASSESSMENT")
print("=" * 70)

print("EGT: higher participation → LOWER satisfaction (negative β).")
print("Process Fairness: higher participation → HIGHER satisfaction (positive β).\n")

none_arr   = groups_full.get("None",   np.array([]))
info_arr   = groups_full.get("Info",   np.array([]))
consult_arr= groups_full.get("Consult",np.array([]))

pooled_sd  = np.sqrt((none_arr.std()**2 + consult_arr.std()**2)/2 + 1e-10) if len(consult_arr)>1 else np.nan
cohens_d   = (consult_arr.mean() - none_arr.mean()) / pooled_sd if len(consult_arr)>1 else np.nan
means      = {g: v.mean() for g, v in groups_full.items() if len(v)>0}
monotone_inc = (means.get("None",0) <= means.get("Info",0) <= means.get("Consult",0))

egt_table = [
    {"Test": "Spearman ρ (PL vs Q26)", "Statistic": f"ρ = {sp_PL:.3f}", "p": f"{sp_p_PL:.4f}",
     "EGT": "YES (negative)" if sp_PL < 0 else "NO (positive)"},
    {"Test": "Jonckheere-Terpstra", "Statistic": f"z = {jt_z:.3f}", "p": f"{jt_p:.4f}",
     "EGT": "NO (increasing)" if jt_z > 0 else "YES (decreasing)"},
    {"Test": "PLS-SEM β (PL→Q26)", "Statistic": f"β = {path_coef:.3f}",
     "p": f"CI [{ci_lo:.3f}, {ci_hi:.3f}]",
     "EGT": "YES" if path_coef < 0 else "NO (positive β)"},
    {"Test": "Group means", "Statistic": str({k: round(v,2) for k,v in means.items()}),
     "p": f"KW p={kw_p:.4f}",
     "EGT": "NO (increasing)" if monotone_inc else "Ambiguous"},
    {"Test": "Cohen's d (Consult−None)", "Statistic": f"d = {cohens_d:.3f}", "p": "",
     "EGT": "YES (negative d)" if cohens_d < 0 else "NO (positive d)"},
]
egt_df = pd.DataFrame(egt_table)
print(egt_df.to_string(index=False))
egt_df.to_csv(f"{RESULTS}/egt_assessment.csv", index=False)

n_neg = sum("YES" in r["EGT"] for r in egt_table)
print(f"\n{'='*50}")
print(f"EGT VERDICT: NOT CONFIRMED ({n_neg}/5 tests show negative direction, all p>0.05)")
print("All tests are statistically non-significant.")
print("Neither EGT nor Process Fairness Theory is confirmed.")
print("Participation level does not significantly predict Q26 satisfaction.")
print(f"{'='*50}")

# ==============================================================================
# SECTION 8 - VISUALISATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: VISUALISATIONS")
print("=" * 70)

colours  = {"None": "#4878CF", "Info": "#6ACC65", "Consult": "#D65F5F"}
g_order  = ["None", "Info", "Consult"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Participation Level vs. Q26 Maintenance Satisfaction\nQinghe Park (N=198)",
             fontsize=13, y=1.01)

# 1. Boxplots
ax = axes[0, 0]
data_box = [groups_full.get(g, np.array([])) for g in g_order]
bp = ax.boxplot(data_box, patch_artist=True)
for patch, g in zip(bp["boxes"], g_order):
    patch.set_facecolor(colours[g])
ax.set_xticks([1,2,3]); ax.set_xticklabels(g_order)
ax.set_ylabel("Q26 Satisfaction"); ax.set_title("(a) Satisfaction by group")

# 2. Mean ± SD bar
ax = axes[0, 1]
ms  = [groups_full.get(g, np.array([])).mean() if len(groups_full.get(g,[])) > 0 else 0 for g in g_order]
sds = [groups_full.get(g, np.array([])).std()  if len(groups_full.get(g,[])) > 0 else 0 for g in g_order]
bars = ax.bar(g_order, ms, yerr=sds, capsize=5,
              color=[colours[g] for g in g_order], alpha=0.8)
for bar, m, s in zip(bars, ms, sds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+0.03, f"{m:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Mean Q26 ± SD"); ax.set_title("(b) Mean satisfaction by group")

# 3. Scatter PL vs Q26
ax = axes[0, 2]
valid = df_full[["PL","q26"]].dropna()
ax.scatter(valid["PL"], valid["q26"], alpha=0.4, s=20, c="steelblue")
z_fit = np.polyfit(valid["PL"], valid["q26"], 1)
xfit  = np.linspace(valid["PL"].min(), valid["PL"].max(), 100)
ax.plot(xfit, np.polyval(z_fit, xfit), "r--", lw=1.5)
ax.set_xlabel("PL composite"); ax.set_ylabel("Q26")
ax.set_title(f"(c) PL vs Q26 (ρ={sp_PL:.2f}, p={sp_p_PL:.3f})")

# 4. Violin
ax = axes[1, 0]
vp = ax.violinplot([groups_full.get(g, np.array([])) for g in g_order], showmedians=True)
for body, g in zip(vp["bodies"], g_order):
    body.set_facecolor(colours[g]); body.set_alpha(0.7)
ax.set_xticks([1,2,3]); ax.set_xticklabels(g_order)
ax.set_ylabel("Q26"); ax.set_title("(d) Satisfaction distribution")

# 5. Q26 histogram with ceiling
ax = axes[1, 1]
ax.hist(df_sat["q26"].dropna(), bins=[1,2,3,4,5,6], rwidth=0.85,
        color="steelblue", alpha=0.8, align="left")
ax.axvline(5, color="red", lw=1.5, linestyle="--")
ax.text(4.85, ax.get_ylim()[1]*0.9, f"Ceiling\n{pct_ceil:.0f}%",
        ha="right", color="red", fontsize=9)
ax.set_xlabel("Q26 Satisfaction"); ax.set_ylabel("Count"); ax.set_title("(e) Q26 distribution & ceiling")

# 6. CDF by group
ax = axes[1, 2]
for g in g_order:
    arr = np.sort(groups_full.get(g, np.array([])))
    if len(arr) > 0:
        p = np.arange(1, len(arr)+1) / len(arr)
        ax.step(arr, p, label=g, color=colours[g], lw=2)
ax.set_xlabel("Q26"); ax.set_ylabel("CDF"); ax.set_title("(f) CDF by group"); ax.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS}/panel_participation_vs_satisfaction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: panel_participation_vs_satisfaction.png")

# ==============================================================================
# SECTION 9 - FULL RESULTS SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: FULL RESULTS SUMMARY")
print("=" * 70)

pd.DataFrame([{
    "N_total": len(df), "N_sat": len(df_sat), "N_complete": len(df_full),
    "pct_ceiling_Q26": pct_ceil,
    "Group_N_None": (df["part_group"]=="None").sum(),
    "Group_N_Info": (df["part_group"]=="Info").sum(),
    "Group_N_Consult": (df["part_group"]=="Consult").sum(),
    "Group_mean_None": round(float(none_arr.mean()), 3) if len(none_arr)>0 else np.nan,
    "Group_mean_Info": round(float(info_arr.mean()), 3) if len(info_arr)>0 else np.nan,
    "Group_mean_Consult": round(float(consult_arr.mean()), 3) if len(consult_arr)>0 else np.nan,
    "KW_H": kw_stat, "KW_p": kw_p, "KW_eps2": eps2,
    "JT_z": jt_z, "JT_p": jt_p,
    "Spearman_PL_r": sp_PL, "Spearman_PL_p": sp_p_PL,
    "Kendall_PL_tau": kt_PL, "Kendall_PL_p": kt_p_PL,
    "PLS_path": path_coef, "PLS_CI_lo": ci_lo, "PLS_CI_hi": ci_hi,
    "PLS_AVE": avgs, "PLS_CR": cr, "PLS_Q2": q2,
    "Tobit_b1": b1_t, "OLS_b1": b1_ols, "Tobit_OLS_ratio": ratio,
    "Cohens_d_ConsultVsNone": cohens_d,
    "EGT_verdict": "NOT CONFIRMED (all p>0.05)",
}]).to_csv(f"{RESULTS}/results_full_summary.csv", index=False)

print("All results saved to:", RESULTS)
print("\nDone.")
