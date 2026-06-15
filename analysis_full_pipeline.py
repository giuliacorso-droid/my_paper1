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
    "4372d09e-Participation_Constructs.xlsx"
)

# ==============================================================================
# SECTION 0 - DATA LOADING AND CLEANING
# ==============================================================================
print("=" * 70)
print("SECTION 0: DATA LOADING AND CLEANING")
print("=" * 70)

raw = pd.read_excel(DATA_PATH, engine="openpyxl", header=0)
print("Raw column names:")
for i, c in enumerate(raw.columns):
    print(f"  [{i:2d}] {repr(c)}")

# IMPORTANT: column names in this Excel file contain literal newline characters.
# Single-keyword matching is NOT sufficient — multiple keywords are required to
# avoid mapping to the wrong column (e.g. "q22" matches Q22\nRaw before Q22\nBinary).
# DO NOT replace this with single-token find_col logic.
def _find_col(df, *keywords):
    """Return first column whose normalised name contains ALL keywords."""
    for c in df.columns:
        cl = str(c).lower().replace("\n", " ")
        if all(kw in cl for kw in keywords):
            return c
    raise KeyError(f"No column matching {keywords!r}. Columns: {list(df.columns)}")

col_q22  = _find_col(raw, "q22", "binary")          # Q22\nBinary\n(0/1)
col_info = _find_col(raw, "info provision", "score") # Info Provision\nSCORE\n(0–1)
col_cons = _find_col(raw, "consultation", "score")   # Consultation\nSCORE\n(0–1)
col_q24  = _find_col(raw, "q24", "ordinal")          # Q24\nOrdinal\n(0/1/2)
col_q25  = _find_col(raw, "q25", "clean")            # Q25\nClean\n(1–5; DK→NaN)
col_flag = _find_col(raw, "flag", "inconsistent")    # FLAG\nInconsistent\n...

print(f"\nMapped columns:")
print(f"  q22   -> {repr(col_q22)}")
print(f"  info  -> {repr(col_info)}")
print(f"  cons  -> {repr(col_cons)}")
print(f"  q24   -> {repr(col_q24)}")
print(f"  q25   -> {repr(col_q25)}")
print(f"  flag  -> {repr(col_flag)}")

work = raw[[col_q22, col_info, col_cons, col_q24, col_q25, col_flag]].copy()
work.columns = ["q22", "info_score", "consult_score", "q24", "q25", "flag"]

for c in ["q22", "info_score", "consult_score", "q24", "q25"]:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work["flag"] = pd.to_numeric(work["flag"], errors="coerce").fillna(0)

work    = work[work["flag"] != 1].reset_index(drop=True)
df      = work[["q22", "info_score", "consult_score", "q24", "q25", "flag"]].copy()
df_sat  = df.dropna(subset=["q25"]).reset_index(drop=True)
df_full = df.dropna(subset=["q22", "info_score", "consult_score", "q24", "q25"]).reset_index(drop=True)

print(f"\nSample sizes after removing flag==1:")
print(f"  df       (all rows)   : N = {len(df)}")
print(f"  df_sat   (q25 present): N = {len(df_sat)}")
print(f"  df_full  (complete)   : N = {len(df_full)}")

# ==============================================================================
# SECTION 1 - DESCRIPTIVE STATISTICS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 70)

from scipy import stats as sp_stats

vars_desc = ["q22", "info_score", "consult_score", "q24", "q25"]
desc_rows = []
for v in vars_desc:
    col_data = df[v].dropna()
    desc_rows.append({
        "variable" : v,
        "N"        : int(col_data.count()),
        "mean"     : col_data.mean(),
        "median"   : col_data.median(),
        "SD"       : col_data.std(),
        "min"      : col_data.min(),
        "max"      : col_data.max(),
        "skew"     : float(sp_stats.skew(col_data)),
        "kurtosis" : float(sp_stats.kurtosis(col_data)),
    })
desc_table = pd.DataFrame(desc_rows).set_index("variable")
print("\nDescriptive statistics:")
print(desc_table.to_string(float_format="{:.3f}".format))
print("\nQ24 value counts:")
print(df["q24"].value_counts().sort_index())
print("\nQ25 value counts:")
print(df["q25"].value_counts().sort_index())

# ==============================================================================
# SECTION 2 - CEILING / FLOOR EFFECTS + TOBIT
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 2: CEILING/FLOOR EFFECTS + TOBIT")
print("=" * 70)

from scipy.optimize import minimize
from numpy.linalg import lstsq as np_lstsq

q24_valid = df["q24"].dropna()
floor_q24_0  = (q24_valid == 0).mean()
floor_q24_01 = (q24_valid <= 1).mean()
print(f"\nFloor effects on Q24:")
print(f"  Proportion at 0      : {floor_q24_0:.3f}")
print(f"  Proportion at 0 or 1 : {floor_q24_01:.3f}")

q25_valid = df_sat["q25"].dropna()
ceil_q25_5  = (q25_valid == 5).mean()
ceil_q25_45 = (q25_valid >= 4).mean()
print(f"\nCeiling effects on Q25:")
print(f"  Proportion at 5      : {ceil_q25_5:.3f}")
print(f"  Proportion at 4 or 5 : {ceil_q25_45:.3f}")

tobit_data = df_full[["q24", "q25"]].dropna().copy()
y_t     = tobit_data["q25"].values
x_t     = tobit_data["q24"].values
X_tobit = np.column_stack([np.ones(len(x_t)), x_t])
L_t, U_t = 1.0, 5.0

def tobit_neg_loglik(params):
    beta      = params[:-1]
    log_sigma = params[-1]
    sigma     = np.exp(log_sigma)
    if sigma <= 0:
        return 1e10
    xb = X_tobit @ beta
    ll = 0.0
    for i in range(len(y_t)):
        if y_t[i] <= L_t:
            p = sp_stats.norm.cdf((L_t - xb[i]) / sigma)
            ll += np.log(max(p, 1e-15))
        elif y_t[i] >= U_t:
            p = 1.0 - sp_stats.norm.cdf((U_t - xb[i]) / sigma)
            ll += np.log(max(p, 1e-15))
        else:
            ll += sp_stats.norm.logpdf(y_t[i], xb[i], sigma)
    return -ll

beta_ols, _, _, _ = np_lstsq(X_tobit, y_t, rcond=None)
sigma_ols = np.std(y_t - X_tobit @ beta_ols)
x0_tobit  = np.append(beta_ols, np.log(max(sigma_ols, 0.01)))

res_tobit            = minimize(tobit_neg_loglik, x0_tobit, method="Nelder-Mead",
                                options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
tobit_beta_intercept = res_tobit.x[0]
tobit_beta_q24       = res_tobit.x[1]
tobit_sigma          = np.exp(res_tobit.x[2])

print(f"\nOLS:   intercept={beta_ols[0]:.3f}, beta_q24={beta_ols[1]:.3f}")
print(f"Tobit: intercept={tobit_beta_intercept:.3f}, beta_q24={tobit_beta_q24:.3f}, sigma={tobit_sigma:.3f}")
print(f"Tobit converged: {res_tobit.success}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(q25_valid, bins=np.arange(0.5, 6.5, 1), density=True,
        color="steelblue", edgecolor="white", alpha=0.8, label="Q25")
xp = np.linspace(1, 5, 200)
ax.plot(xp, sp_stats.norm.pdf(xp, q25_valid.mean(), q25_valid.std()),
        "r-", lw=2, label="Normal fit")
ax.axvline(5, color="orange", ls="--", lw=1.5, label="Ceiling (5)")
ax.set_xlabel("Q25 Satisfaction (1-5)")
ax.set_ylabel("Density")
ax.set_title("Q25 Distribution with Normal Overlay")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "hist_q25_normal.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
cnt24 = q24_valid.value_counts().sort_index()
ax.bar(cnt24.index, cnt24.values, color="teal", edgecolor="white", alpha=0.85)
ax.set_xlabel("Q24 Participation Level")
ax.set_ylabel("Count")
ax.set_title("Q24 Participation Level Distribution")
ax.set_xticks([0, 1, 2])
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "hist_q24.png"), dpi=150)
plt.close(fig)

g0 = df_sat[df_sat["q24"] == 0]["q25"].dropna()
g1 = df_sat[df_sat["q24"] == 1]["q25"].dropna()
g2 = df_sat[df_sat["q24"] == 2]["q25"].dropna()
gdata = [g0.values, g1.values, g2.values]

fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot(gdata,
           tick_labels=["Q24=0\n(None)", "Q24=1\n(Informed)", "Q24=2\n(Consulted)"],
           patch_artist=True,
           boxprops=dict(facecolor="lightblue"),
           medianprops=dict(color="navy", lw=2))
ax.set_ylabel("Q25 Satisfaction (1-5)")
ax.set_title("Satisfaction by Participation Level")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "boxplot_sat_q24.png"), dpi=150)
plt.close(fig)
print("\nFigures saved: hist_q25_normal.png, hist_q24.png, boxplot_sat_q24.png")

# ==============================================================================
# SECTION 3 - NON-PARAMETRIC TESTS
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 3: NON-PARAMETRIC TESTS")
print("=" * 70)

sw_stat, sw_p = sp_stats.shapiro(q25_valid)
print(f"\nShapiro-Wilk on Q25: W={sw_stat:.4f}, p={sw_p:.4f}")

kw_H, kw_p = sp_stats.kruskal(g0, g1, g2)
N_kw = len(g0) + len(g1) + len(g2)
eps_sq = (kw_H - 2) / (N_kw - 3)
print(f"\nKruskal-Wallis: H={kw_H:.4f}, p={kw_p:.4f}, epsilon2={eps_sq:.4f}")

pairs_mw = [(g0, g1, "0 vs 1"), (g0, g2, "0 vs 2"), (g1, g2, "1 vs 2")]
n_comp = 3
print("\nPairwise Mann-Whitney U (Bonferroni corrected):")
for ga, gb, label in pairs_mw:
    U_mw, p_raw = sp_stats.mannwhitneyu(ga, gb, alternative="two-sided")
    p_bonf = min(p_raw * n_comp, 1.0)
    r_rb   = 1 - (2 * U_mw) / (len(ga) * len(gb))
    print(f"  {label}: U={U_mw:.1f}, p_raw={p_raw:.4f}, p_Bonf={p_bonf:.4f}, r_rb={r_rb:.3f}")

pairs_corr = [
    ("q24",           "q25", "Q24 <-> Q25"),
    ("info_score",    "q25", "info_score <-> Q25"),
    ("consult_score", "q25", "consult_score <-> Q25"),
]
print("\nSpearman rho and Kendall tau:")
for va, vb, label in pairs_corr:
    sub = df_full[[va, vb]].dropna()
    rho, p_rho = sp_stats.spearmanr(sub[va], sub[vb])
    tau, p_tau = sp_stats.kendalltau(sub[va], sub[vb])
    print(f"  {label}: rho={rho:.3f} (p={p_rho:.4f}), tau={tau:.3f} (p={p_tau:.4f})")

def jonckheere_terpstra(groups):
    k  = len(groups)
    J  = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xi < xj:
                        J += 1.0
                    elif xi == xj:
                        J += 0.5
    ns   = np.array([len(g) for g in groups])
    N    = ns.sum()
    EJ   = (N ** 2 - np.sum(ns ** 2)) / 4.0
    VarJ = (N ** 2 * (2 * N + 3) - np.sum(ns ** 2 * (2 * ns + 3))) / 72.0
    z    = (J - EJ) / np.sqrt(VarJ)
    p    = 1.0 - sp_stats.norm.cdf(z)
    return J, EJ, VarJ, z, p

J_jt, EJ_jt, VarJ_jt, z_jt, p_jt = jonckheere_terpstra(gdata)
print(f"\nJonckheere-Terpstra: J={J_jt:.1f}, E[J]={EJ_jt:.1f}, z={z_jt:.3f}, p(one-tail)={p_jt:.4f}")

# ==============================================================================
# SECTION 4 - ORDINAL LOGISTIC REGRESSION
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ORDINAL LOGISTIC REGRESSION")
print("=" * 70)

from statsmodels.miscmodels.ordinal_model import OrderedModel

ord_df = df_full[["q22", "info_score", "consult_score", "q24", "q25"]].dropna().copy()
ord_df["sat_cat"] = pd.Categorical(
    ord_df["q25"].astype(int), categories=[1, 2, 3, 4, 5], ordered=True
)

def pseudo_r2_mcfadden(model_result, endog):
    ll_full = model_result.llf
    try:
        null_mod = OrderedModel(endog, np.ones((len(endog), 1)), distr="logit")
        null_res = null_mod.fit(method="bfgs", disp=False)
        ll_null  = null_res.llf
    except Exception:
        return np.nan
    if ll_null and not np.isnan(ll_null) and ll_null != 0:
        return 1 - ll_full / ll_null
    return np.nan

print("\nModel 1: q24 -> satisfaction")
res1 = None
try:
    mod1 = OrderedModel(ord_df["sat_cat"], ord_df[["q24"]], distr="logit")
    res1 = mod1.fit(method="bfgs", disp=False)
    print(res1.summary())
    print(f"McFadden pseudo-R2: {pseudo_r2_mcfadden(res1, ord_df['sat_cat']):.4f}")
except Exception as e:
    print(f"  Model 1 failed: {e}")

print("\nModel 2: q22 + info_score + consult_score -> satisfaction")
res2 = None
try:
    mod2 = OrderedModel(ord_df["sat_cat"], ord_df[["q22", "info_score", "consult_score"]], distr="logit")
    res2 = mod2.fit(method="bfgs", disp=False)
    print(res2.summary())
    print(f"McFadden pseudo-R2: {pseudo_r2_mcfadden(res2, ord_df['sat_cat']):.4f}")
except Exception as e:
    print(f"  Model 2 failed: {e}")

# ==============================================================================
# SECTION 5 - PLS-SEM
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 5: PLS-SEM")
print("=" * 70)

pls_df = df_full[["q22", "info_score", "consult_score", "q25"]].dropna().copy()

def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-12)

pls_df["q22_z"]  = zscore(pls_df["q22"])
pls_df["info_z"] = zscore(pls_df["info_score"])
pls_df["cons_z"] = zscore(pls_df["consult_score"])
pls_df["sat_z"]  = zscore(pls_df["q25"])

indicators = pls_df[["q22_z", "info_z", "cons_z"]].values
sat_z      = pls_df["sat_z"].values
n_pls      = len(pls_df)

weights = np.ones(3) / 3.0
for _ in range(500):
    score       = indicators @ weights
    score       = score / (score.std() + 1e-12)
    inner_proxy = sat_z * np.corrcoef(score, sat_z)[0, 1]
    new_w       = indicators.T @ inner_proxy / n_pls
    new_w       = new_w / (np.linalg.norm(new_w) + 1e-12)
    if np.max(np.abs(new_w - weights)) < 1e-8:
        break
    weights = new_w

participation_score = indicators @ weights
participation_score = (participation_score - participation_score.mean()) / (
    participation_score.std() + 1e-12)

X_inner = np.column_stack([np.ones(n_pls), participation_score])
beta_inner, _, _, _ = np_lstsq(X_inner, sat_z, rcond=None)
y_pred_inner = X_inner @ beta_inner
r2_inner  = 1.0 - np.var(sat_z - y_pred_inner) / (np.var(sat_z) + 1e-12)
path_coef = beta_inner[1]

print(f"\nOuter weights: q22={weights[0]:.3f}, info={weights[1]:.3f}, consult={weights[2]:.3f}")
print(f"Path coefficient (PARTICIPATION -> SATISFACTION): {path_coef:.3f}")
print(f"R2 (inner model): {r2_inner:.3f}")

rng_boot   = np.random.default_rng(42)
boot_paths = []
for _ in range(500):
    idx   = rng_boot.integers(0, n_pls, n_pls)
    ind_b = indicators[idx]
    sat_b = sat_z[idx]
    w_b   = np.ones(3) / 3.0
    for __ in range(200):
        sc_b = ind_b @ w_b
        sc_b = sc_b / (sc_b.std() + 1e-12)
        cor  = np.corrcoef(sc_b, sat_b)[0, 1] if len(sat_b) > 1 else 0.0
        nw_b = ind_b.T @ (sat_b * cor) / len(idx)
        nol  = np.linalg.norm(nw_b)
        if nol < 1e-12:
            break
        nw_b = nw_b / nol
        if np.max(np.abs(nw_b - w_b)) < 1e-6:
            break
        w_b = nw_b
    sc_b = ind_b @ w_b
    sc_b = (sc_b - sc_b.mean()) / (sc_b.std() + 1e-12)
    Xi   = np.column_stack([np.ones(len(idx)), sc_b])
    bi, _, _, _ = np_lstsq(Xi, sat_b, rcond=None)
    boot_paths.append(bi[1])

boot_paths = np.array(boot_paths)
ci_lo, ci_hi = np.percentile(boot_paths, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

d_blind = 7
sse_bf, sso_bf = 0.0, 0.0
for start in range(d_blind):
    omit_idx = list(range(start, n_pls, d_blind))
    keep_idx = [i for i in range(n_pls) if i not in omit_idx]
    if len(keep_idx) < 3:
        continue
    ind_k = indicators[keep_idx]
    sat_k = sat_z[keep_idx]
    w_k   = np.ones(3) / 3.0
    for __ in range(200):
        sc_k = ind_k @ w_k
        sc_k = sc_k / (sc_k.std() + 1e-12)
        cor_k = np.corrcoef(sc_k, sat_k)[0, 1] if len(sat_k) > 1 else 0.0
        nw_k  = ind_k.T @ (sat_k * cor_k) / len(keep_idx)
        nol_k = np.linalg.norm(nw_k)
        if nol_k < 1e-12:
            break
        nw_k = nw_k / nol_k
        if np.max(np.abs(nw_k - w_k)) < 1e-6:
            break
        w_k = nw_k
    sc_ko = indicators[keep_idx] @ w_k
    mu_ko = sc_ko.mean(); sd_ko = sc_ko.std() + 1e-12
    sc_ko = (sc_ko - mu_ko) / sd_ko
    Xi_k  = np.column_stack([np.ones(len(keep_idx)), sc_ko])
    bi_k, _, _, _ = np_lstsq(Xi_k, sat_k, rcond=None)
    sc_o  = (indicators[omit_idx] @ w_k - mu_ko) / sd_ko
    pred_o = np.column_stack([np.ones(len(omit_idx)), sc_o]) @ bi_k
    sse_bf += np.sum((sat_z[omit_idx] - pred_o) ** 2)
    sso_bf += np.sum(sat_z[omit_idx] ** 2)

q2_blind = 1.0 - sse_bf / (sso_bf + 1e-12)
f2_pls   = r2_inner / (1.0 - r2_inner + 1e-12)
print(f"Q2 (blindfolding d=7): {q2_blind:.3f}")
print(f"f2: {f2_pls:.3f}")

# ==============================================================================
# SECTION 6 - EXPECTATION GAP THEORY
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 6: EXPECTATION GAP THEORY")
print("=" * 70)

eg_df = df_sat[["q24", "q25"]].dropna().copy()
eg_df["predicted_higher"] = eg_df["q24"] > 0
eg_df["sat_above_median"] = eg_df["q25"] > eg_df["q25"].median()

table_eg = pd.crosstab(
    eg_df["predicted_higher"].map({True: "Participates", False: "No participation"}),
    eg_df["sat_above_median"].map({True: "Sat>median", False: "Sat<=median"}),
    margins=True,
)
print("\nExpected vs Observed satisfaction direction:")
print(table_eg)

gr_consulted = eg_df[eg_df["q24"] == 2]["q25"]
gr_other     = eg_df[eg_df["q24"] <= 1]["q25"]
U_eg, p_eg   = sp_stats.mannwhitneyu(gr_consulted, gr_other, alternative="greater")
print(f"\nOne-tailed Mann-Whitney (Q24=2 > Q24<=1): U={U_eg:.1f}, p={p_eg:.4f}")

lev_stat, lev_p = sp_stats.levene(
    eg_df[eg_df["q24"] == 0]["q25"].dropna(),
    eg_df[eg_df["q24"] == 1]["q25"].dropna(),
    eg_df[eg_df["q24"] == 2]["q25"].dropna(),
)
print(f"Levene's test: F={lev_stat:.3f}, p={lev_p:.4f}")

g_q24_0 = eg_df[eg_df["q24"] == 0]["q25"].dropna().values
g_q24_2 = eg_df[eg_df["q24"] == 2]["q25"].dropna().values

def cohens_d(a, b):
    na, nb    = len(a), len(b)
    pooled_sd = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    return (np.mean(b) - np.mean(a)) / (pooled_sd + 1e-12)

d_coh   = cohens_d(g_q24_0, g_q24_2)
verdict = "SUPPORTED" if p_eg < 0.05 and d_coh > 0.2 else "NOT SUPPORTED"
print(f"Cohen\'s d (Q24=2 vs Q24=0): {d_coh:.3f}")
print(f"\nExpectation Gap Theory verdict: {verdict}")

# ==============================================================================
# SECTION 7 - XGBOOST + SHAP
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 7: XGBOOST + SHAP")
print("=" * 70)

shap_available      = False
shap_vals_for_panel = None
X_xgb_for_panel     = None
feat_cols_for_panel = None
cv_r2_mean          = np.nan
cv_rmse_mean        = np.nan

try:
    import xgboost as xgb
    import shap
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error

    feat_df = df_full[["q22", "info_score", "consult_score", "q24", "q25"]].dropna().copy()

    def normalise(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-12)

    feat_df["participation_composite"] = (
        0.35 * normalise(feat_df["q22"])
        + 0.20 * normalise(feat_df["q24"])
        + 0.45 * normalise(feat_df["info_score"] + feat_df["consult_score"])
    )

    feature_cols = ["q22", "info_score", "consult_score", "q24", "participation_composite"]
    X_xgb        = feat_df[feature_cols].values
    y_xgb        = feat_df["q25"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_list, cv_rmse_list = [], []
    for train_idx, test_idx in kf.split(X_xgb):
        m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=42, verbosity=0)
        m.fit(X_xgb[train_idx], y_xgb[train_idx])
        pred = m.predict(X_xgb[test_idx])
        cv_r2_list.append(r2_score(y_xgb[test_idx], pred))
        cv_rmse_list.append(np.sqrt(mean_squared_error(y_xgb[test_idx], pred)))

    cv_r2_mean   = np.mean(cv_r2_list)
    cv_rmse_mean = np.mean(cv_rmse_list)
    print(f"\nXGBoost 5-fold CV: R2={cv_r2_mean:.3f}+/-{np.std(cv_r2_list):.3f}, "
          f"RMSE={cv_rmse_mean:.3f}+/-{np.std(cv_rmse_list):.3f}")

    model_full  = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                    random_state=42, verbosity=0)
    model_full.fit(X_xgb, y_xgb)
    explainer   = shap.TreeExplainer(model_full)
    shap_values = explainer.shap_values(X_xgb)

    shap_available      = True
    shap_vals_for_panel = shap_values
    X_xgb_for_panel     = X_xgb
    feat_cols_for_panel = feature_cols

    try:
        shap.summary_plot(shap_values, X_xgb, feature_names=feature_cols,
                          show=False, plot_type="dot")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        print("SHAP beeswarm saved.")
    except Exception as e_bs:
        print(f"SHAP beeswarm failed: {e_bs}")

    try:
        fig_dep, ax_dep = plt.subplots(figsize=(6, 4))
        shap.dependence_plot(3, shap_values, X_xgb, feature_names=feature_cols,
                             ax=ax_dep, show=False)
        fig_dep.tight_layout()
        fig_dep.savefig(os.path.join(RESULTS, "shap_dependence_q24.png"), dpi=150)
        plt.close(fig_dep)
        print("SHAP dependence plot saved.")
    except Exception as e_dep:
        print(f"SHAP dependence plot failed: {e_dep}")

    preds_full = model_full.predict(X_xgb)
    for label, idx in [("best", int(np.argmax(preds_full))),
                        ("worst", int(np.argmin(preds_full)))]:
        try:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx],
                    base_values=explainer.expected_value,
                    data=X_xgb[idx],
                    feature_names=feature_cols,
                ),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"shap_waterfall_{label}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"SHAP waterfall ({label}) saved.")
        except Exception as ew:
            print(f"SHAP waterfall {label} failed: {ew}")

except ImportError as ie:
    print(f"xgboost or shap not installed - skipping Section 7. ({ie})")

# ==============================================================================
# SECTION 8 - COMPREHENSIVE 3x3 PANEL FIGURE (18x15 inches)
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 8: COMPREHENSIVE 3x3 PANEL FIGURE")
print("=" * 70)

fig_panel, axes = plt.subplots(3, 3, figsize=(18, 15))
fig_panel.subplots_adjust(hspace=0.45, wspace=0.38)

def add_bracket(ax, x1, x2, y, p_val, delta=0.05):
    ax.plot([x1, x1, x2, x2], [y, y + delta, y + delta, y], lw=1.2, c="k")
    stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01 else
             "*" if p_val < 0.05 else "ns")
    ax.text((x1 + x2) / 2, y + delta + 0.01, stars, ha="center", va="bottom", fontsize=9)

def draw_box(ax, x, y, w, h, text, color="#AED6F1"):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor="navy", lw=1.5,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold")

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.5))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15,
                label, ha="center", fontsize=7.5, color="darkred")

# (A) Boxplot with significance brackets
ax_A = axes[0, 0]
ax_A.boxplot(gdata,
             tick_labels=["None\n(Q24=0)", "Informed\n(Q24=1)", "Consulted\n(Q24=2)"],
             patch_artist=True,
             boxprops=dict(facecolor="#AED6F1"),
             medianprops=dict(color="navy", lw=2))
ax_A.set_ylabel("Satisfaction (1-5)")
ax_A.set_title("(A) Satisfaction by Participation Level")
y_max_A = max(g.max() for g in gdata if len(g) > 0) + 0.3
_, p01 = sp_stats.mannwhitneyu(g0, g1, alternative="two-sided")
_, p02 = sp_stats.mannwhitneyu(g0, g2, alternative="two-sided")
_, p12 = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
add_bracket(ax_A, 1, 2, y_max_A,        min(p01 * 3, 1.0))
add_bracket(ax_A, 2, 3, y_max_A + 0.22, min(p12 * 3, 1.0))
add_bracket(ax_A, 1, 3, y_max_A + 0.44, min(p02 * 3, 1.0))

# (B) Bar chart mean +/- SE
ax_B = axes[0, 1]
means_B = [g.mean() for g in gdata]
ses_B   = [g.std() / np.sqrt(len(g)) for g in gdata]
clrs_B  = ["#2ECC71", "#F39C12", "#9B59B6"]
bars_B  = ax_B.bar([0, 1, 2], means_B, yerr=ses_B, capsize=5,
                   color=clrs_B, edgecolor="white", width=0.55)
ax_B.set_xticks([0, 1, 2])
ax_B.set_xticklabels(["None\n(Q24=0)", "Informed\n(Q24=1)", "Consulted\n(Q24=2)"])
ax_B.set_ylabel("Mean Satisfaction +/- SE")
ax_B.set_title("(B) Mean Satisfaction by Participation Level")
ax_B.set_ylim(0, 5.8)
for bar, m in zip(bars_B, means_B):
    ax_B.text(bar.get_x() + bar.get_width() / 2, m + 0.12, f"{m:.2f}", ha="center", fontsize=9)

# (C) Scatter with jitter and trend line
ax_C = axes[0, 2]
scatter_df = df_sat[["q24", "q25"]].dropna()
rng_j = np.random.default_rng(7)
jx = scatter_df["q24"].values + rng_j.uniform(-0.12, 0.12, len(scatter_df))
jy = scatter_df["q25"].values + rng_j.uniform(-0.12, 0.12, len(scatter_df))
ax_C.scatter(jx, jy, alpha=0.45, color="steelblue", s=25)
m_sc, b_sc = np.polyfit(scatter_df["q24"], scatter_df["q25"], 1)
xl = np.linspace(-0.2, 2.2, 50)
ax_C.plot(xl, m_sc * xl + b_sc, "r-", lw=2, label=f"slope={m_sc:.2f}")
ax_C.set_xlabel("Q24 Participation Level")
ax_C.set_ylabel("Q25 Satisfaction")
ax_C.set_title("(C) Q24 vs Satisfaction (jittered)")
ax_C.set_xticks([0, 1, 2])
ax_C.legend(fontsize=8)

# (D) PLS-SEM path diagram using matplotlib patches
ax_D = axes[1, 0]
ax_D.set_xlim(0, 10)
ax_D.set_ylim(0, 7)
ax_D.axis("off")
ax_D.set_title("(D) PLS-SEM Path Diagram")
draw_box(ax_D, 2.0, 5.5, 2.5, 0.9, "Q22\n(Participation\nlevel)")
draw_box(ax_D, 2.0, 3.5, 2.5, 0.9, "Info Score\n(Q23 info)")
draw_box(ax_D, 2.0, 1.5, 2.5, 0.9, "Consult Score\n(Q23 consult)")
draw_box(ax_D, 5.7, 3.5, 2.2, 1.0, "PARTICIPATION\n(Latent)", color="#F9E79F")
draw_box(ax_D, 8.8, 3.5, 2.2, 1.0, "SATISFACTION\n(Latent)", color="#ABEBC6")
draw_arrow(ax_D, 3.25, 5.5, 4.6, 4.1, f"w={weights[0]:.2f}")
draw_arrow(ax_D, 3.25, 3.5, 4.6, 3.5, f"w={weights[1]:.2f}")
draw_arrow(ax_D, 3.25, 1.5, 4.6, 2.9, f"w={weights[2]:.2f}")
draw_arrow(ax_D, 6.8,  3.5, 7.7, 3.5, f"beta={path_coef:.2f}")

# (E) SHAP beeswarm recreated manually if SHAP not available
ax_E = axes[1, 1]
ax_E.set_title("(E) SHAP Feature Importance")
if shap_available and shap_vals_for_panel is not None:
    mean_abs = np.abs(shap_vals_for_panel).mean(axis=0)
    sidx     = np.argsort(mean_abs)
    clrs_E   = ["#E74C3C", "#E67E22", "#3498DB", "#2ECC71", "#9B59B6"]
    ax_E.barh(range(len(feat_cols_for_panel)), mean_abs[sidx],
              color=[clrs_E[i % len(clrs_E)] for i in range(len(sidx))])
    ax_E.set_yticks(range(len(feat_cols_for_panel)))
    ax_E.set_yticklabels([feat_cols_for_panel[i] for i in sidx], fontsize=8)
    ax_E.set_xlabel("Mean |SHAP value|")
else:
    feat_names_e = ["q22", "info_score", "consult_score", "q24", "composite"]
    corrs_e = []
    for fn in ["q22", "info_score", "consult_score", "q24"]:
        sub_e = df_full[[fn, "q25"]].dropna()
        corrs_e.append(abs(sp_stats.spearmanr(sub_e[fn], sub_e["q25"])[0]))
    comp_e = (df_full["q22"].fillna(0) * 0.35
              + df_full["q24"].fillna(0) * 0.20
              + (df_full["info_score"].fillna(0) + df_full["consult_score"].fillna(0)) * 0.45)
    corrs_e.append(abs(sp_stats.spearmanr(comp_e, df_full["q25"].fillna(df_full["q25"].mean()))[0]))
    sidx_e = np.argsort(corrs_e)
    ax_E.barh(range(5), [corrs_e[i] for i in sidx_e], color="#3498DB", alpha=0.8)
    ax_E.set_yticks(range(5))
    ax_E.set_yticklabels([feat_names_e[i] for i in sidx_e], fontsize=8)
    ax_E.set_xlabel("|Spearman rho| (importance proxy)")

# (F) Tobit vs OLS coefficient comparison bar chart
ax_F = axes[1, 2]
ax_F.set_title("(F) Tobit vs OLS: beta (Q24->Q25)")
beta_labels_F = ["OLS", "Tobit"]
beta_vals_F   = [beta_ols[1], tobit_beta_q24]
clrs_F        = ["#2980B9", "#E74C3C"]
bars_F = ax_F.bar(beta_labels_F, beta_vals_F, color=clrs_F, width=0.45, edgecolor="white")
ax_F.axhline(0, color="black", lw=0.8, ls="--")
ax_F.set_ylabel("Coefficient beta")
for bar, val in zip(bars_F, beta_vals_F):
    off = 0.008 if val >= 0 else -0.012
    ax_F.text(bar.get_x() + bar.get_width() / 2, val + off, f"{val:.3f}",
              ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

# (G) Ceiling/floor histogram Q25 annotated
ax_G = axes[2, 0]
ax_G.hist(q25_valid, bins=np.arange(0.5, 6.5, 1),
          color="mediumseagreen", edgecolor="white", alpha=0.85)
ax_G.axvline(5, color="red",  ls="--", lw=2, label=f"Ceiling 5: {ceil_q25_5:.1%}")
ax_G.axvline(1, color="blue", ls="--", lw=1.5, label="Floor 1")
ax_G.set_xlabel("Q25 Satisfaction")
ax_G.set_ylabel("Count")
ax_G.set_title("(G) Ceiling/Floor: Q25 Distribution")
ax_G.legend(fontsize=8)
ylim_G = ax_G.get_ylim()[1]
ax_G.text(4.9, ylim_G * 0.88, f"{ceil_q25_5:.1%} at ceiling", ha="right", fontsize=8, color="red")
ax_G.text(1.1, ylim_G * 0.88, f"{(q25_valid==1).mean():.1%} at floor", ha="left", fontsize=8, color="blue")

# (H) Expectation gap verdict text panel
ax_H = axes[2, 1]
ax_H.axis("off")
ax_H.set_title("(H) Expectation Gap Verdict")
summary_text = (
    f"Expectation Gap Theory Test\n"
    f"{'=' * 34}\n"
    f"Q24=2 vs Q24<=1 (one-tailed MW):\n"
    f"  U = {U_eg:.0f},  p = {p_eg:.4f}\n\n"
    f"Cohen\'s d (Q24=2 vs Q24=0):\n"
    f"  d = {d_coh:.3f}\n\n"
    f"Levene variance test:\n"
    f"  F = {lev_stat:.3f},  p = {lev_p:.4f}\n\n"
    f"Verdict: {verdict}\n\n"
    f"{'Consulted participants report' if verdict == 'SUPPORTED' else 'No significant gap:'}\n"
    f"{'higher satisfaction (gap theory ok).' if verdict == 'SUPPORTED' else 'groups similar in satisfaction.'}"
)
ax_H.text(0.05, 0.95, summary_text, transform=ax_H.transAxes,
          va="top", ha="left", fontsize=8.5, family="monospace",
          bbox=dict(boxstyle="round", facecolor="#FEF9E7", edgecolor="#F39C12", lw=1.5))

# (I) Jonckheere trend visualization
ax_I = axes[2, 2]
ax_I.set_title("(I) Jonckheere-Terpstra Trend")
medians_I = [np.median(g) for g in gdata]
means_I   = [np.mean(g)   for g in gdata]
ax_I.plot([0, 1, 2], means_I,   "o-",  color="#E74C3C", lw=2, ms=8, label="Mean")
ax_I.plot([0, 1, 2], medians_I, "s--", color="#2980B9", lw=2, ms=8, label="Median")
ax_I.fill_between([0, 1, 2], means_I, alpha=0.12, color="#E74C3C")
ax_I.set_xticks([0, 1, 2])
ax_I.set_xticklabels(["None\n(Q24=0)", "Informed\n(Q24=1)", "Consulted\n(Q24=2)"])
ax_I.set_ylabel("Satisfaction (1-5)")
ax_I.legend(fontsize=8)
ax_I.text(0.97, 0.06, f"JT: z={z_jt:.2f}, p={p_jt:.4f}",
          transform=ax_I.transAxes, ha="right", fontsize=8.5, color="darkred",
          bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange", lw=1))

fig_panel.suptitle(
    "Comprehensive Analysis Panel: Public Participation & Satisfaction",
    fontsize=14, fontweight="bold", y=1.01,
)
panel_path = os.path.join(RESULTS, "panel_3x3_comprehensive.png")
fig_panel.savefig(panel_path, dpi=150, bbox_inches="tight")
plt.close(fig_panel)
print(f"Panel figure saved: {panel_path}")

# ==============================================================================
# SECTION 9 - SUMMARY AND SAVE
# ==============================================================================
print("\n" + "=" * 70)
print("SECTION 9: SUMMARY AND SAVE")
print("=" * 70)

summary_rows = [
    {"Analysis": "Descriptive",    "Test/Model": "N complete cases",
     "Statistic": str(len(df_full)), "p-value": "-", "Effect size": "-"},
    {"Analysis": "Descriptive",    "Test/Model": "Mean Q25",
     "Statistic": f"{df_sat['q25'].mean():.3f}", "p-value": "-", "Effect size": "-"},
    {"Analysis": "Ceiling/Floor",  "Test/Model": "Ceiling Q25 at 5",
     "Statistic": f"{ceil_q25_5:.3f}", "p-value": "-", "Effect size": "-"},
    {"Analysis": "Ceiling/Floor",  "Test/Model": "Floor Q24 at 0",
     "Statistic": f"{floor_q24_0:.3f}", "p-value": "-", "Effect size": "-"},
    {"Analysis": "Tobit",          "Test/Model": "Tobit beta (Q24->Q25)",
     "Statistic": f"{tobit_beta_q24:.3f}", "p-value": "-", "Effect size": "-"},
    {"Analysis": "Tobit",          "Test/Model": "OLS beta (Q24->Q25)",
     "Statistic": f"{beta_ols[1]:.3f}", "p-value": "-", "Effect size": "-"},
    {"Analysis": "Non-parametric", "Test/Model": "Shapiro-Wilk Q25",
     "Statistic": f"W={sw_stat:.4f}", "p-value": f"{sw_p:.4f}", "Effect size": "-"},
    {"Analysis": "Non-parametric", "Test/Model": "Kruskal-Wallis H",
     "Statistic": f"H={kw_H:.4f}", "p-value": f"{kw_p:.4f}",
     "Effect size": f"epsilon2={eps_sq:.3f}"},
    {"Analysis": "Non-parametric", "Test/Model": "Jonckheere-Terpstra z",
     "Statistic": f"z={z_jt:.3f}", "p-value": f"{p_jt:.4f}", "Effect size": "-"},
    {"Analysis": "PLS-SEM",        "Test/Model": "Path beta (PART->SAT)",
     "Statistic": f"{path_coef:.3f}",
     "p-value": f"95%CI[{ci_lo:.3f},{ci_hi:.3f}]",
     "Effect size": f"R2={r2_inner:.3f},f2={f2_pls:.3f},Q2={q2_blind:.3f}"},
    {"Analysis": "Expectation Gap","Test/Model": "MW U (Q24=2 vs <=1)",
     "Statistic": f"U={U_eg:.0f}", "p-value": f"{p_eg:.4f}",
     "Effect size": f"d={d_coh:.3f}"},
    {"Analysis": "Expectation Gap","Test/Model": "Verdict",
     "Statistic": verdict, "p-value": "-", "Effect size": "-"},
]
if shap_available:
    summary_rows.append({
        "Analysis": "XGBoost", "Test/Model": "5-fold CV R2",
        "Statistic": f"{cv_r2_mean:.3f}", "p-value": "-",
        "Effect size": f"RMSE={cv_rmse_mean:.3f}",
    })

summary_df = pd.DataFrame(summary_rows)
csv_path   = os.path.join(RESULTS, "results_full_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSummary CSV saved: {csv_path}")
print("\nConsolidated Summary Table:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("ALL SECTIONS COMPLETE")
print(f"Results saved to: {RESULTS}")
print("=" * 70)
