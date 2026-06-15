import matplotlib; matplotlib.use("Agg")

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings('ignore')

# ── Results directory ────────────────────────────────────────────────────────
RESULTS_DIR = "/home/user/my_paper1/results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── XGBoost / SHAP ──────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("WARNING: xgboost not available; Section 7 will be skipped.")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("WARNING: shap not available; SHAP plots will be skipped.")

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
DATA_PATH = (
    "/root/.claude/uploads/7e325fe5-5704-55cc-9ede-a4a3bdde6097/"
    "4372d09e-Participation_Constructs.xlsx"
)

print("=" * 70)
print("SECTION 0: DATA LOADING AND CLEANING")
print("=" * 70)

raw = pd.read_excel(DATA_PATH, engine="openpyxl")

print("All column names (before mapping):")
for i, c in enumerate(raw.columns):
    print(f"  [{i}] {repr(c)}")

# Map columns by partial name match (case-insensitive)
def _find_col(df, *keywords):
    """Return first column whose lower-cased text contains ALL keywords."""
    for c in df.columns:
        cl = c.lower().replace("\n", " ")
        if all(kw in cl for kw in keywords):
            return c
    raise KeyError(f"No column matching {keywords!r}")

col_q22   = _find_col(raw, "q22", "binary")
col_info  = _find_col(raw, "info provision", "score")
col_cons  = _find_col(raw, "consultation", "score")
col_q24   = _find_col(raw, "q24", "ordinal")
col_q25   = _find_col(raw, "q25", "clean")
col_flag  = _find_col(raw, "flag", "inconsistent")

print(f"\nMapped columns (partial-name match):")
print(f"  q22         -> {repr(col_q22)}")
print(f"  info_score  -> {repr(col_info)}")
print(f"  cons_score  -> {repr(col_cons)}")
print(f"  q24         -> {repr(col_q24)}")
print(f"  q25/sat     -> {repr(col_q25)}")
print(f"  flag        -> {repr(col_flag)}")

# Explicit column mapping — column names contain literal newlines from Excel
raw = raw.rename(columns={
    col_q22:  "q22",
    col_info: "info_score",
    col_cons: "consult_score",
    col_q24:  "q24",
    col_q25:  "satisfaction",
    col_flag: "flag",
})

# Remove internally inconsistent records and cast to numeric
df = raw[raw["flag"].fillna(0) != 1].reset_index(drop=True)
for c in ["q22", "info_score", "consult_score", "q24", "satisfaction"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Sub-dataframes
df_sat = df.dropna(subset=["satisfaction"]).reset_index(drop=True)
df_full = df.dropna(subset=["q22", "info_score", "consult_score", "q24", "satisfaction"]).reset_index(drop=True)

print(f"\nSample sizes:")
print(f"  df       (all cleaned rows)          : n = {len(df)}")
print(f"  df_sat   (satisfaction non-NaN)       : n = {len(df_sat)}")
print(f"  df_full  (all 5 variables non-NaN)    : n = {len(df_full)}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 70)

vars_desc = {
    "q22": df["q22"],
    "info_score": df["info_score"],
    "consult_score": df["consult_score"],
    "q24": df["q24"],
    "satisfaction (Q25)": df["satisfaction"],
}

desc_rows = []
for vname, s in vars_desc.items():
    s_clean = s.dropna()
    desc_rows.append({
        "Variable": vname,
        "N": int(s_clean.count()),
        "Mean": round(s_clean.mean(), 4),
        "Median": round(s_clean.median(), 4),
        "SD": round(s_clean.std(), 4),
        "Min": round(s_clean.min(), 4),
        "Max": round(s_clean.max(), 4),
        "Skew": round(s_clean.skew(), 4),
        "Kurtosis": round(s_clean.kurtosis(), 4),
    })

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))

print("\nValue counts — Q24:")
print(df["q24"].value_counts().sort_index())
print("\nValue counts — Satisfaction (Q25):")
print(df["satisfaction"].value_counts().sort_index())

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: CEILING AND FLOOR EFFECT TESTS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: CEILING AND FLOOR EFFECT TESTS")
print("=" * 70)

q24_clean = df["q24"].dropna()
q25_clean = df["satisfaction"].dropna()

# Floor on Q24
prop_q24_0 = (q24_clean == 0).mean()
prop_q24_01 = ((q24_clean == 0) | (q24_clean == 1)).mean()
print(f"\nFloor effect on Q24:")
print(f"  Proportion at 0:   {prop_q24_0:.3f}")
print(f"  Proportion at 0+1: {prop_q24_01:.3f}")

# Ceiling on Q25
prop_q25_5 = (q25_clean == 5).mean()
prop_q25_45 = (q25_clean >= 4).mean()
print(f"\nCeiling effect on Q25:")
print(f"  Proportion at 5:   {prop_q25_5:.3f}")
print(f"  Proportion at 4+5: {prop_q25_45:.3f}")

# Kurtosis test
kurt_q25 = q25_clean.kurtosis()
print(f"\nKurtosis of Q25: {kurt_q25:.4f}")
if kurt_q25 > 1.0:
    print("  → Leptokurtic: possible ceiling/floor concentration")
elif kurt_q25 < -1.0:
    print("  → Platykurtic: responses spread across scale")
else:
    print("  → Near-mesokurtic: approximately normal distribution shape")

# ── Tobit regression (two-limit, lower=1, upper=5) ──────────────────────────
def tobit_loglik(params, y, X, lower=1.0, upper=5.0):
    """Two-limit Tobit log-likelihood (negative, for minimization)."""
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return 1e10
    mu = X @ beta
    ll = 0.0
    for i in range(len(y)):
        if y[i] <= lower:
            z = (lower - mu[i]) / sigma
            p = stats.norm.cdf(z)
            ll += np.log(max(p, 1e-300))
        elif y[i] >= upper:
            z = (upper - mu[i]) / sigma
            p = 1 - stats.norm.cdf(z)
            ll += np.log(max(p, 1e-300))
        else:
            ll += stats.norm.logpdf(y[i], mu[i], sigma)
    return -ll

df_tobit = df_full[["q24", "satisfaction"]].dropna().reset_index(drop=True)
y_tobit = df_tobit["satisfaction"].values.astype(float)
X_tobit = sm.add_constant(df_tobit["q24"].values.astype(float))

# OLS for comparison
ols_res = sm.OLS(y_tobit, X_tobit).fit()
ols_beta_q24 = ols_res.params[1]
print(f"\nOLS β for Q24→Q25: {ols_beta_q24:.4f} (p={ols_res.pvalues[1]:.4f})")

# Tobit optimization
init_params = np.array([ols_res.params[0], ols_res.params[1], np.log(y_tobit.std())])
tobit_result = minimize(
    tobit_loglik,
    init_params,
    args=(y_tobit, X_tobit, 1.0, 5.0),
    method="Nelder-Mead",
    options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
)
tobit_beta_q24 = tobit_result.x[1]
tobit_sigma = np.exp(tobit_result.x[-1])
print(f"Tobit β for Q24→Q25: {tobit_beta_q24:.4f}  (σ={tobit_sigma:.4f})")
if abs(ols_beta_q24) > 1e-10:
    print(f"Attenuation ratio (Tobit/OLS): {tobit_beta_q24/ols_beta_q24:.4f}")
else:
    print("OLS β ≈ 0; attenuation ratio undefined")

# ── Section 2 figures ─────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

# Histogram Q25 with normal overlay
ax = axes2[0]
ax.hist(q25_clean, bins=np.arange(0.5, 6.5, 1), density=True, alpha=0.7,
        color="steelblue", edgecolor="white")
xr = np.linspace(1, 5, 200)
ax.plot(xr, stats.norm.pdf(xr, q25_clean.mean(), q25_clean.std()),
        "r-", lw=2, label="Normal overlay")
ax.axvline(5, color="red", linestyle="--", alpha=0.5, label="Ceiling")
ax.set_xlabel("Satisfaction (Q25)")
ax.set_ylabel("Density")
ax.set_title("Q25 Distribution with Normal Overlay")
ax.legend()

# Histogram Q24
ax = axes2[1]
ax.hist(q24_clean, bins=[-0.5, 0.5, 1.5, 2.5], align="mid",
        edgecolor="white", color="coral", alpha=0.8)
ax.set_xlabel("Participation Level (Q24)")
ax.set_ylabel("Count")
ax.set_title("Q24 Distribution")
ax.set_xticks([0, 1, 2])

# Boxplot by group
ax = axes2[2]
groups_box = [df_full[df_full["q24"] == g]["satisfaction"].dropna().values
              for g in [0, 1, 2]]
bp = ax.boxplot(groups_box, patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Q24=0", "Q24=1", "Q24=2"])
ax.set_ylabel("Satisfaction (Q25)")
ax.set_title("Satisfaction by Q24 Group")

fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, "fig2_ceiling_floor.png"), dpi=150)
plt.close(fig2)
print("Saved fig2_ceiling_floor.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: NON-PARAMETRIC TESTS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: NON-PARAMETRIC TESTS")
print("=" * 70)

# Shapiro-Wilk on Q25
sw_stat, sw_p = stats.shapiro(q25_clean)
print(f"\nShapiro-Wilk on Q25: W={sw_stat:.4f}, p={sw_p:.4f}")
if sw_p < 0.05:
    print("  → Q25 is NOT normally distributed")
else:
    print("  → Q25 does not significantly deviate from normality")

# Kruskal-Wallis
sat_by_group = {g: df_full[df_full["q24"] == g]["satisfaction"].dropna().values
                for g in [0, 1, 2]}
valid_groups = [(g, v) for g, v in sat_by_group.items() if len(v) > 0]

if len(valid_groups) >= 2:
    kw_stat, kw_p = stats.kruskal(*[v for _, v in valid_groups])
    n_total = sum(len(v) for _, v in valid_groups)
    k_groups = len(valid_groups)
    epsilon_sq = (kw_stat - k_groups + 1) / (n_total - k_groups)
    epsilon_sq = max(0.0, epsilon_sq)
    print(f"\nKruskal-Wallis H={kw_stat:.4f}, p={kw_p:.4f}, ε²={epsilon_sq:.4f}")
    if epsilon_sq < 0.01:
        eff_label = "negligible"
    elif epsilon_sq < 0.06:
        eff_label = "small"
    elif epsilon_sq < 0.14:
        eff_label = "medium"
    else:
        eff_label = "large"
    print(f"  → Effect size: {eff_label}")
else:
    kw_stat, kw_p, epsilon_sq = np.nan, np.nan, np.nan
    print("Not enough groups for Kruskal-Wallis")

# Pairwise Mann-Whitney U with Bonferroni correction
group_pairs = [(0, 1), (0, 2), (1, 2)]
n_comparisons = len(group_pairs)
print("\nPairwise Mann-Whitney U (Bonferroni corrected):")
mw_results = []
for g1, g2 in group_pairs:
    v1 = sat_by_group.get(g1, np.array([]))
    v2 = sat_by_group.get(g2, np.array([]))
    if len(v1) > 0 and len(v2) > 0:
        u_stat, u_p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        p_bonf = min(u_p * n_comparisons, 1.0)
        n1, n2 = len(v1), len(v2)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        print(f"  Q24={g1} vs Q24={g2}: U={u_stat:.1f}, p_raw={u_p:.4f}, "
              f"p_bonf={p_bonf:.4f}, r_rb={r_rb:.4f}")
        mw_results.append({"pair": f"{g1}vs{g2}", "U": u_stat,
                            "p_raw": u_p, "p_bonf": p_bonf, "r_rb": r_rb})
    else:
        print(f"  Q24={g1} vs Q24={g2}: insufficient data")

# Spearman and Kendall correlations
print("\nCorrelation analysis:")
sp_r_q24, sp_p_q24, kd_t_q24, kd_p_q24 = np.nan, np.nan, np.nan, np.nan
for var_name, col in [("Q24", "q24"), ("info_score", "info_score"),
                       ("consult_score", "consult_score")]:
    sub = df_full[[col, "satisfaction"]].dropna().reset_index(drop=True)
    if len(sub) > 3:
        sp_r, sp_p = stats.spearmanr(sub[col], sub["satisfaction"])
        kd_t, kd_p = stats.kendalltau(sub[col], sub["satisfaction"])
        print(f"  {var_name} ↔ Q25: Spearman ρ={sp_r:.4f} (p={sp_p:.4f}), "
              f"Kendall τ={kd_t:.4f} (p={kd_p:.4f})")
        if var_name == "Q24":
            sp_r_q24, sp_p_q24 = sp_r, sp_p
            kd_t_q24, kd_p_q24 = kd_t, kd_p
    else:
        print(f"  {var_name} ↔ Q25: insufficient data")

# Jonckheere-Terpstra trend test (manual implementation)
# J = sum over all ordered pairs (i<j) of U(group_i < group_j).
# Large J indicates that later groups tend to have LARGER values (increasing trend).
def jonckheere_terpstra(groups_data):
    """
    Manual Jonckheere-Terpstra trend test.
    groups_data: list of arrays in ordered group sequence (g0, g1, ...).
    J = sum_{i<j} #{(a,b): a in g_i, b in g_j, a < b}
    Large J -> increasing trend. Returns J, z, p (one-tailed upper).
    """
    k = len(groups_data)
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            # scipy mannwhitneyu returns U1 = #{(a,b): a in g_i, b in g_j, a > b}
            # For increasing-trend JT we need #{a < b} = n_i*n_j - U1
            ni = len(groups_data[i])
            nj = len(groups_data[j])
            u1, _ = stats.mannwhitneyu(groups_data[i], groups_data[j],
                                        alternative="two-sided")
            J += (ni * nj - u1)  # #{a_i < a_j}
    n = [len(g) for g in groups_data]
    N = sum(n)
    E_J = (N**2 - sum(ni**2 for ni in n)) / 4
    term1 = N**2 * (2 * N + 3)
    term2 = sum(ni**2 * (2 * ni + 3) for ni in n)
    Var_J = (term1 - term2) / 72
    if Var_J <= 0:
        return J, np.nan, np.nan
    z = (J - E_J) / np.sqrt(Var_J)
    p = stats.norm.sf(z)  # one-tailed
    return J, z, p

jt_groups = [sat_by_group.get(g, np.array([])) for g in [0, 1, 2]]
jt_groups_valid = [g for g in jt_groups if len(g) > 0]
if len(jt_groups_valid) >= 2:
    J_stat, J_z, J_p = jonckheere_terpstra(jt_groups_valid)
    print(f"\nJonckheere-Terpstra trend test: J={J_stat:.1f}, z={J_z:.4f}, "
          f"p={J_p:.4f} (one-tailed)")
    if J_p < 0.05:
        print("  → Significant positive trend across Q24 levels")
    else:
        print("  → No significant monotone trend")
else:
    J_stat, J_z, J_p = np.nan, np.nan, np.nan
    print("\nJonckheere-Terpstra: insufficient data")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: ORDINAL LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: ORDINAL LOGISTIC REGRESSION")
print("=" * 70)

df_ord = (df_full[["q22", "info_score", "consult_score", "q24", "satisfaction"]]
          .dropna().reset_index(drop=True))
df_ord["sat_cat"] = df_ord["satisfaction"].astype(int).astype("category")

ordinal_results = {}

# Model 1: q24 → satisfaction
try:
    mod1 = OrderedModel(df_ord["sat_cat"], df_ord[["q24"]], distr="logit")
    res1 = mod1.fit(method="bfgs", disp=False)
    print("\nModel 1 (Q24 → Satisfaction):")
    print(res1.summary())
    llf1 = res1.llf
    ll0_1 = getattr(res1, "llnull", np.nan)
    if not np.isnan(ll0_1) and ll0_1 != 0:
        mcfadden1 = 1 - llf1 / ll0_1
        print(f"  McFadden pseudo-R²: {mcfadden1:.4f}")
    ordinal_results["model1"] = res1
except Exception as e:
    print(f"Model 1 failed: {e}")
    ordinal_results["model1"] = None

# Model 2: q22 + info_score + consult_score → satisfaction
try:
    mod2 = OrderedModel(df_ord["sat_cat"],
                        df_ord[["q22", "info_score", "consult_score"]],
                        distr="logit")
    res2 = mod2.fit(method="bfgs", disp=False)
    print("\nModel 2 (q22 + info_score + consult_score → Satisfaction):")
    print(res2.summary())
    llf2 = res2.llf
    ll0_2 = getattr(res2, "llnull", np.nan)
    if not np.isnan(ll0_2) and ll0_2 != 0:
        mcfadden2 = 1 - llf2 / ll0_2
        print(f"  McFadden pseudo-R²: {mcfadden2:.4f}")
    ordinal_results["model2"] = res2
except Exception as e:
    print(f"Model 2 failed: {e}")
    ordinal_results["model2"] = None

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: PLS-SEM PATH MODEL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: PLS-SEM PATH MODEL")
print("=" * 70)


def standardize(x):
    s = x.std()
    if s == 0:
        return x - x.mean()
    return (x - x.mean()) / s


def pls_mode_a(X_df, y_series, n_iter=500, tol=1e-6):
    """
    PLS mode-A outer weights via power method.
    Returns outer_weights, latent_scores.
    """
    X = X_df.values.astype(float)
    p = X.shape[1]
    w = np.ones(p) / np.sqrt(p)
    for _ in range(n_iter):
        scores = X @ w
        std_s = scores.std()
        if std_s > 0:
            scores = (scores - scores.mean()) / std_s
        new_w = np.array([np.corrcoef(X[:, j], scores)[0, 1] for j in range(p)])
        norm = np.linalg.norm(new_w)
        if norm > 0:
            new_w = new_w / norm
        if np.linalg.norm(new_w - w) < tol:
            break
        w = new_w
    latent = X @ w
    std_l = latent.std()
    if std_l > 0:
        latent = (latent - latent.mean()) / std_l
    return w, latent


df_pls = (df_full[["q22", "info_score", "consult_score", "satisfaction"]]
          .dropna().reset_index(drop=True))
X_ind = df_pls[["q22", "info_score", "consult_score"]].apply(standardize)
y_sat = standardize(df_pls["satisfaction"])

outer_weights, latent_scores = pls_mode_a(X_ind, y_sat)
print(f"\nOuter weights (q22, info_score, consult_score): {np.round(outer_weights, 4)}")

# Inner model
X_inner = sm.add_constant(latent_scores)
inner_ols = sm.OLS(y_sat, X_inner).fit()
path_coef = inner_ols.params.iloc[1]
r2_inner = inner_ols.rsquared
print(f"Path coefficient (PARTICIPATION → SATISFACTION): {path_coef:.4f}")
print(f"R² (inner model): {r2_inner:.4f}")

# Bootstrap CI (500 resamples)
np.random.seed(42)
n_boot = 500
boot_paths = []
for _ in range(n_boot):
    idx = np.random.choice(len(df_pls), size=len(df_pls), replace=True)
    X_b = X_ind.iloc[idx].reset_index(drop=True)
    y_b = y_sat.iloc[idx].reset_index(drop=True)
    try:
        _, lat_b = pls_mode_a(X_b, y_b)
        X_ib = sm.add_constant(lat_b)
        ols_b = sm.OLS(y_b, X_ib).fit()
        boot_paths.append(ols_b.params.iloc[1])
    except Exception:
        pass

boot_paths = np.array(boot_paths)
ci_low = np.percentile(boot_paths, 2.5)
ci_high = np.percentile(boot_paths, 97.5)
print(f"Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# Q² via blindfolding (d=7)
d = 7
n_obs = len(df_pls)
sse_blind = 0.0
sso_blind = 0.0
y_mean_pls = y_sat.mean()
for start in range(d):
    blind_idx = list(range(start, n_obs, d))
    train_idx = [i for i in range(n_obs) if i not in blind_idx]
    if len(train_idx) < 5:
        continue
    X_train = X_ind.iloc[train_idx].reset_index(drop=True)
    y_train = y_sat.iloc[train_idx].reset_index(drop=True)
    try:
        w_bl, lat_train = pls_mode_a(X_train, y_train)
        X_itr = sm.add_constant(lat_train)
        ols_blind = sm.OLS(y_train, X_itr).fit()
        X_test = X_ind.iloc[blind_idx].reset_index(drop=True)
        y_test = y_sat.iloc[blind_idx].values
        lat_test = X_test.values @ w_bl
        std_lt = lat_test.std()
        if std_lt > 0:
            lat_test = (lat_test - lat_test.mean()) / std_lt
        X_itest = sm.add_constant(lat_test)
        y_pred_blind = X_itest @ ols_blind.params
        sse_blind += np.sum((y_test - y_pred_blind) ** 2)
        sso_blind += np.sum((y_test - y_mean_pls) ** 2)
    except Exception:
        pass

q2 = 1 - sse_blind / sso_blind if sso_blind > 0 else np.nan
print(f"Q² (blindfolding, d=7): {q2:.4f}")

f2 = (r2_inner) / (1 - r2_inner) if r2_inner < 1 else np.nan
print(f"f² effect size: {f2:.4f}")
if not np.isnan(f2):
    if f2 < 0.02:
        f2_label = "negligible"
    elif f2 < 0.15:
        f2_label = "small"
    elif f2 < 0.35:
        f2_label = "medium"
    else:
        f2_label = "large"
    print(f"  → {f2_label} effect")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: EXPECTATION GAP THEORY ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: EXPECTATION GAP THEORY ASSESSMENT")
print("=" * 70)

print("\nMean satisfaction by Q24 group:")
for g in [0, 1, 2]:
    v = sat_by_group.get(g, np.array([]))
    if len(v) > 0:
        print(f"  Q24={g}: M={v.mean():.3f}, n={len(v)}")
    else:
        print(f"  Q24={g}: no data")

m0 = sat_by_group[0].mean() if len(sat_by_group[0]) > 0 else np.nan
m1 = sat_by_group[1].mean() if len(sat_by_group[1]) > 0 else np.nan
m2 = sat_by_group[2].mean() if len(sat_by_group[2]) > 0 else np.nan

pred_direction = "Q24=2 < Q24=1 < Q24=0"
print(f"\nExpectation Gap Theory predicted direction: {pred_direction}")
print(f"Observed: Q24=0: {m0:.3f}, Q24=1: {m1:.3f}, Q24=2: {m2:.3f}")

obs_pattern = []
if not (np.isnan(m2) or np.isnan(m1)):
    obs_pattern.append("Q24=2 < Q24=1" if m2 < m1 else "Q24=2 >= Q24=1")
if not (np.isnan(m2) or np.isnan(m0)):
    obs_pattern.append("Q24=2 < Q24=0" if m2 < m0 else "Q24=2 >= Q24=0")
print("Observed pattern:", ", ".join(obs_pattern))

# One-tailed Mann-Whitney: Q24=2 LOWER than Q24=1 or Q24=0
eg_results = {}
for g_comp in [0, 1]:
    v_comp = sat_by_group.get(g_comp, np.array([]))
    v2_eg = sat_by_group.get(2, np.array([]))
    if len(v2_eg) > 0 and len(v_comp) > 0:
        u_eg, p_eg = stats.mannwhitneyu(v2_eg, v_comp, alternative="less")
        eg_results[g_comp] = (u_eg, p_eg)
        print(f"  Mann-Whitney (Q24=2 < Q24={g_comp}): U={u_eg:.1f}, "
              f"p={p_eg:.4f} (one-tailed)")
    else:
        eg_results[g_comp] = (np.nan, np.nan)

# Levene's test
levene_vals = [v for v in sat_by_group.values() if len(v) > 0]
if len(levene_vals) >= 2:
    lev_stat, lev_p = stats.levene(*levene_vals)
    print(f"\nLevene's test for variance homogeneity: W={lev_stat:.4f}, p={lev_p:.4f}")
    if lev_p < 0.05:
        print("  → Variances are NOT homogeneous")
    else:
        print("  → Variances are homogeneous")
else:
    lev_stat, lev_p = np.nan, np.nan

# Cohen's d between Q24=0 and Q24=2
v0 = sat_by_group.get(0, np.array([]))
v2_coh = sat_by_group.get(2, np.array([]))
if len(v0) > 1 and len(v2_coh) > 1:
    pooled_sd = np.sqrt(
        ((len(v0) - 1) * v0.std()**2 + (len(v2_coh) - 1) * v2_coh.std()**2)
        / (len(v0) + len(v2_coh) - 2)
    )
    cohens_d = (v0.mean() - v2_coh.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    print(f"\nCohen's d (Q24=0 vs Q24=2): {cohens_d:.4f}")
else:
    cohens_d = np.nan

# Verdict
any_p = [eg_results[g][1] for g in [0, 1] if not np.isnan(eg_results.get(g, (np.nan, np.nan))[1])]
n_sig = sum(p < 0.05 for p in any_p)
if n_sig == 2:
    eg_verdict = "CONFIRMED"
elif n_sig == 1:
    eg_verdict = "PARTIALLY CONFIRMED"
else:
    eg_verdict = "DISCONFIRMED"
print(f"\nExpectation Gap Theory verdict: {eg_verdict}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: XGBOOST + SHAP
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: XGBOOST + SHAP")
print("=" * 70)

xgb_r2_cv = np.nan
xgb_rmse_cv = np.nan
shap_values_arr = None
X_shap = None
feature_cols_shap = ["q22", "info_score", "consult_score", "q24", "participation_composite"]

if XGBOOST_OK:
    df_xgb = (df_full[["q22", "info_score", "consult_score", "q24", "satisfaction"]]
              .dropna().reset_index(drop=True))

    def norm_col(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mn) / (mx - mn)

    pc = (0.35 * norm_col(df_xgb["q22"])
          + 0.20 * norm_col(df_xgb["q24"])
          + 0.45 * norm_col(df_xgb["info_score"] + df_xgb["consult_score"]))
    df_xgb["participation_composite"] = pc

    X_shap = df_xgb[feature_cols_shap].values
    y_xgb = df_xgb["satisfaction"].values

    model_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores_cv = []
    rmse_scores_cv = []
    for train_i, test_i in kf.split(X_shap):
        model_xgb.fit(X_shap[train_i], y_xgb[train_i])
        preds = model_xgb.predict(X_shap[test_i])
        r2_scores_cv.append(r2_score(y_xgb[test_i], preds))
        rmse_scores_cv.append(np.sqrt(mean_squared_error(y_xgb[test_i], preds)))

    xgb_r2_cv = np.mean(r2_scores_cv)
    xgb_rmse_cv = np.mean(rmse_scores_cv)
    print(f"XGBoost 5-fold CV: R²={xgb_r2_cv:.4f}, RMSE={xgb_rmse_cv:.4f}")

    # Fit on full data for SHAP
    model_xgb.fit(X_shap, y_xgb)

    if SHAP_OK:
        explainer = shap.TreeExplainer(model_xgb)
        shap_values_arr = explainer.shap_values(X_shap)

        # Beeswarm plot
        try:
            shap.summary_plot(shap_values_arr, X_shap,
                              feature_names=feature_cols_shap, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "fig7_shap_beeswarm.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved fig7_shap_beeswarm.png")
        except Exception as e:
            print(f"SHAP beeswarm plot failed: {e}")
            plt.close("all")

        # Dependence plot for q24
        try:
            q24_idx = feature_cols_shap.index("q24")
            fig_dep, ax_dep = plt.subplots(figsize=(6, 4))
            shap.dependence_plot(q24_idx, shap_values_arr, X_shap,
                                 feature_names=feature_cols_shap, ax=ax_dep, show=False)
            fig_dep.tight_layout()
            fig_dep.savefig(os.path.join(RESULTS_DIR, "fig7_shap_dependence_q24.png"),
                            dpi=150, bbox_inches="tight")
            plt.close(fig_dep)
            print("Saved fig7_shap_dependence_q24.png")
        except Exception as e:
            print(f"SHAP dependence plot failed: {e}")
            plt.close("all")

        # Waterfall for best and worst predictions
        try:
            preds_all = model_xgb.predict(X_shap)
            best_idx = int(np.argmin(np.abs(preds_all - y_xgb)))
            worst_idx = int(np.argmax(np.abs(preds_all - y_xgb)))
            base_val = explainer.expected_value
            if hasattr(base_val, "__len__"):
                base_val = base_val[0]
            for label, idx in [("best", best_idx), ("worst", worst_idx)]:
                try:
                    exp_obj = shap.Explanation(
                        values=shap_values_arr[idx],
                        base_values=float(base_val),
                        data=X_shap[idx],
                        feature_names=feature_cols_shap,
                    )
                    shap.plots.waterfall(exp_obj, show=False)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(RESULTS_DIR, f"fig7_shap_waterfall_{label}.png"),
                        dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"Saved fig7_shap_waterfall_{label}.png")
                except Exception as e:
                    print(f"SHAP waterfall ({label}) failed: {e}")
                    plt.close("all")
        except Exception as e:
            print(f"SHAP waterfall setup failed: {e}")
    else:
        print("SHAP not available; skipping SHAP plots.")
else:
    print("XGBoost not available; Section 7 skipped.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: COMPREHENSIVE RESULTS FIGURE (3×3 panel)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: COMPREHENSIVE RESULTS FIGURE")
print("=" * 70)

fig8 = plt.figure(figsize=(18, 15))
gs = GridSpec(3, 3, figure=fig8, hspace=0.48, wspace=0.38)

colors_abc = ["#4878CF", "#6ACC65", "#D65F5F"]
groups_A_vals = [sat_by_group.get(g, np.array([])) for g in [0, 1, 2]]
labels_A = [f"Q24={g}" for g in [0, 1, 2]]

# ── (A) Boxplot satisfaction by Q24 group with significance brackets ────────
ax_A = fig8.add_subplot(gs[0, 0])
valid_A = [(lbl, v) for lbl, v in zip(labels_A, groups_A_vals) if len(v) > 0]
lbl_A_valid = [x[0] for x in valid_A]
val_A_valid = [x[1] for x in valid_A]
bp = ax_A.boxplot(val_A_valid, patch_artist=True)
ax_A.set_xticks(range(1, len(lbl_A_valid) + 1))
ax_A.set_xticklabels(lbl_A_valid)
for patch, color in zip(bp["boxes"], colors_abc):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_A.set_ylabel("Satisfaction (Q25)")
ax_A.set_title("(A) Satisfaction by Q24 Group")

# Significance brackets
if len(val_A_valid) >= 2:
    y_max_A = max(v.max() for v in val_A_valid)
    y_bracket = y_max_A + 0.3
    for pos_pair in [(1, 2), (1, 3), (2, 3)]:
        x1p, x2p = pos_pair
        if x2p <= len(val_A_valid):
            v1_br = val_A_valid[x1p - 1]
            v2_br = val_A_valid[x2p - 1]
            _, p_br = stats.mannwhitneyu(v1_br, v2_br, alternative="two-sided")
            p_br_b = min(p_br * 3, 1.0)
            sig_lbl = ("***" if p_br_b < 0.001 else
                       ("**" if p_br_b < 0.01 else
                        ("*" if p_br_b < 0.05 else "ns")))
            ax_A.annotate("", xy=(x2p, y_bracket), xytext=(x1p, y_bracket),
                          arrowprops=dict(arrowstyle="-", color="black"))
            ax_A.text((x1p + x2p) / 2, y_bracket + 0.05, sig_lbl,
                      ha="center", fontsize=9)
            y_bracket += 0.35

# ── (B) Bar chart mean±SE by Q24 group ────────────────────────────────────
ax_B = fig8.add_subplot(gs[0, 1])
means_B = [v.mean() for v in val_A_valid]
ses_B = [v.std() / np.sqrt(len(v)) if len(v) > 0 else 0 for v in val_A_valid]
x_B = np.arange(len(lbl_A_valid))
ax_B.bar(x_B, means_B, yerr=ses_B, capsize=5,
         color=colors_abc[:len(means_B)], alpha=0.8)
ax_B.set_xticks(x_B)
ax_B.set_xticklabels(lbl_A_valid)
ax_B.set_ylabel("Mean Satisfaction (±SE)")
ax_B.set_title("(B) Mean ± SE by Q24 Group")
if means_B:
    ax_B.set_ylim(0, max(m + s for m, s in zip(means_B, ses_B)) + 0.5)

# ── (C) Scatter Q24 vs satisfaction with jitter and trend line ─────────────
ax_C = fig8.add_subplot(gs[0, 2])
np.random.seed(0)
x_jit = df_full["q24"].values.astype(float) + np.random.uniform(-0.15, 0.15, len(df_full))
y_jit = df_full["satisfaction"].values.astype(float)
valid_C = ~(np.isnan(x_jit) | np.isnan(y_jit))
ax_C.scatter(x_jit[valid_C], y_jit[valid_C], alpha=0.4, s=30, color="steelblue")
if valid_C.sum() > 2:
    m_C, b_C, _, _, _ = stats.linregress(
        df_full["q24"].values.astype(float)[valid_C], y_jit[valid_C])
    xfit_C = np.linspace(df_full["q24"].min(), df_full["q24"].max(), 50)
    ax_C.plot(xfit_C, m_C * xfit_C + b_C, "r-", lw=2, label=f"Trend (β={m_C:.3f})")
    ax_C.legend(fontsize=8)
ax_C.set_xlabel("Q24 (with jitter)")
ax_C.set_ylabel("Satisfaction (Q25)")
ax_C.set_title("(C) Q24 vs Satisfaction")
ax_C.set_xticks([0, 1, 2])

# ── (D) PLS-SEM path diagram schematic ────────────────────────────────────
ax_D = fig8.add_subplot(gs[1, 0])
ax_D.set_xlim(0, 10)
ax_D.set_ylim(0, 6)
ax_D.axis("off")
ax_D.set_title("(D) PLS-SEM Path Diagram")

ind_labels_D = ["q22", "info_score", "consult_score"]
ind_ys_D = [5.0, 3.0, 1.0]
ind_x_D = 0.3
for lbl, iy in zip(ind_labels_D, ind_ys_D):
    rect = mpatches.FancyBboxPatch((ind_x_D, iy - 0.4), 1.8, 0.8,
                                    boxstyle="round,pad=0.1",
                                    fc="#AED6F1", ec="steelblue", lw=1.5)
    ax_D.add_patch(rect)
    ax_D.text(ind_x_D + 0.9, iy, lbl, ha="center", va="center", fontsize=8)

part_x_D, part_y_D = 4.5, 3.0
ellipse_part = mpatches.Ellipse((part_x_D, part_y_D), 2.2, 1.5,
                                 fc="#A9DFBF", ec="#27AE60", lw=2)
ax_D.add_patch(ellipse_part)
ax_D.text(part_x_D, part_y_D, "PARTICIP.", ha="center", va="center",
          fontsize=8, fontweight="bold")

sat_x_D, sat_y_D = 8.5, 3.0
ellipse_sat_D = mpatches.Ellipse((sat_x_D, sat_y_D), 2.0, 1.2,
                                  fc="#F9E79F", ec="#D4AC0D", lw=2)
ax_D.add_patch(ellipse_sat_D)
ax_D.text(sat_x_D, sat_y_D, "SATISF.", ha="center", va="center",
          fontsize=8, fontweight="bold")

for ow, iy in zip(outer_weights, ind_ys_D):
    target_y = part_y_D + (iy - part_y_D) * 0.35
    ax_D.annotate("", xy=(part_x_D - 1.1, target_y),
                  xytext=(ind_x_D + 1.8, iy),
                  arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5))
    ax_D.text(ind_x_D + 2.3, (iy + target_y) / 2,
              f"{ow:.2f}", fontsize=7, color="steelblue")

ax_D.annotate("", xy=(sat_x_D - 1.0, sat_y_D), xytext=(part_x_D + 1.1, part_y_D),
              arrowprops=dict(arrowstyle="->", color="#27AE60", lw=2.5))
ax_D.text(6.5, sat_y_D + 0.55, f"β={path_coef:.3f}", fontsize=9,
          color="#27AE60", fontweight="bold")
ax_D.text(6.5, sat_y_D - 0.55, f"R²={r2_inner:.3f}", fontsize=8, color="gray")

# ── (E) SHAP summary (manual bar chart as fallback) ─────────────────────
ax_E = fig8.add_subplot(gs[1, 1])
shap_plotted_E = False
if XGBOOST_OK and SHAP_OK and shap_values_arr is not None:
    try:
        mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)
        ax_E.barh(np.arange(len(sorted_idx)), mean_abs_shap[sorted_idx],
                  color="#E74C3C", alpha=0.8)
        ax_E.set_yticks(np.arange(len(sorted_idx)))
        ax_E.set_yticklabels([feature_cols_shap[i] for i in sorted_idx], fontsize=8)
        ax_E.set_xlabel("Mean |SHAP value|")
        ax_E.set_title("(E) SHAP Feature Importance")
        shap_plotted_E = True
    except Exception as e:
        print(f"SHAP panel E failed: {e}")

if not shap_plotted_E:
    ax_E.text(0.5, 0.5, "SHAP not available\n(install xgboost and shap)",
              ha="center", va="center", transform=ax_E.transAxes, fontsize=10)
    ax_E.set_title("(E) SHAP Feature Importance")
    ax_E.axis("off")

# ── (F) Tobit vs OLS coefficient comparison ───────────────────────────────
ax_F = fig8.add_subplot(gs[1, 2])
coef_labels_F = ["OLS β", "Tobit β"]
coef_vals_F = [ols_beta_q24, tobit_beta_q24]
colors_F = ["#3498DB", "#E67E22"]
bars_F = ax_F.bar(coef_labels_F, coef_vals_F, color=colors_F, alpha=0.85, width=0.5)
ax_F.axhline(0, color="black", lw=0.8)
for bar, val in zip(bars_F, coef_vals_F):
    offset = 0.005 if val >= 0 else -0.015
    va_pos = "bottom" if val >= 0 else "top"
    ax_F.text(bar.get_x() + bar.get_width() / 2, val + offset,
              f"{val:.4f}", ha="center", va=va_pos, fontsize=10)
ax_F.set_ylabel("Coefficient (Q24 → Q25)")
ax_F.set_title("(F) Tobit vs OLS Coefficients")

# ── (G) Ceiling/floor histogram Q25 annotated ────────────────────────────
ax_G = fig8.add_subplot(gs[2, 0])
counts_G = [int((q25_clean == v).sum()) for v in [1, 2, 3, 4, 5]]
bar_colors_G = ["#85C1E9", "#85C1E9", "#85C1E9", "#85C1E9", "#85C1E9"]
ax_G.bar([1, 2, 3, 4, 5], counts_G, color=bar_colors_G, edgecolor="white", alpha=0.85)
# Highlight ceiling and floor
ax_G.bar([5], [counts_G[4]], color="#E74C3C", alpha=0.85,
         label=f"Ceiling n={counts_G[4]} ({prop_q25_5:.1%})")
ax_G.bar([1], [counts_G[0]], color="#1ABC9C", alpha=0.85,
         label=f"Floor n={counts_G[0]}")
ax_G.set_xlabel("Satisfaction (Q25)")
ax_G.set_ylabel("Count")
ax_G.set_title(f"(G) Q25 Distribution\nCeiling={prop_q25_5:.1%} at 5")
ax_G.legend(fontsize=7)

# ── (H) Expectation gap theory verdict text panel ─────────────────────────
ax_H = fig8.add_subplot(gs[2, 1])
ax_H.axis("off")
ax_H.set_title("(H) Expectation Gap Theory")
verdict_color_map = {
    "CONFIRMED": "#27AE60",
    "PARTIALLY CONFIRMED": "#F39C12",
    "DISCONFIRMED": "#E74C3C",
}
vc = verdict_color_map.get(eg_verdict, "gray")
cohens_d_str = f"{cohens_d:.3f}" if not np.isnan(cohens_d) else "n/a"
eg_text_lines = (
    "Expectation Gap Theory\n\n"
    "Prediction: Higher participation\n"
    "→ Lower satisfaction\n"
    "(raised expectations not met)\n\n"
    f"Observed means:\n"
    f"  Q24=0: {m0:.3f}\n"
    f"  Q24=1: {m1:.3f}\n"
    f"  Q24=2: {m2:.3f}\n\n"
    f"Cohen's d (0 vs 2): {cohens_d_str}\n\n"
    f"VERDICT: {eg_verdict}"
)
ax_H.text(0.05, 0.95, eg_text_lines, transform=ax_H.transAxes,
          fontsize=8.5, va="top", ha="left", family="monospace",
          bbox=dict(boxstyle="round", facecolor=vc, alpha=0.15))
ax_H.text(0.5, 0.05, eg_verdict, transform=ax_H.transAxes, fontsize=14,
          ha="center", va="bottom", color=vc, fontweight="bold")

# ── (I) Jonckheere trend visualization ───────────────────────────────────
ax_I = fig8.add_subplot(gs[2, 2])
group_means_I = [sat_by_group.get(g, np.array([])).mean()
                 if len(sat_by_group.get(g, [])) > 0 else np.nan
                 for g in [0, 1, 2]]
group_ses_I = [sat_by_group.get(g, np.array([])).std() / np.sqrt(len(sat_by_group.get(g, [])))
               if len(sat_by_group.get(g, [])) > 1 else 0
               for g in [0, 1, 2]]
x_I_full = [0, 1, 2]
valid_I_mask = [not np.isnan(m) for m in group_means_I]
x_I_v = [x for x, v in zip(x_I_full, valid_I_mask) if v]
m_I_v = [m for m, v in zip(group_means_I, valid_I_mask) if v]
se_I_v = [s for s, v in zip(group_ses_I, valid_I_mask) if v]

ax_I.errorbar(x_I_v, m_I_v, yerr=se_I_v, fmt="o-", color="#8E44AD",
              capsize=5, lw=2, markersize=8)
if len(x_I_v) > 1:
    ax_I.fill_between(x_I_v,
                      [m - s for m, s in zip(m_I_v, se_I_v)],
                      [m + s for m, s in zip(m_I_v, se_I_v)],
                      alpha=0.2, color="#8E44AD")
ax_I.set_xticks([0, 1, 2])
ax_I.set_xticklabels(["Q24=0\n(None)", "Q24=1\n(Info)", "Q24=2\n(Consult)"])
ax_I.set_ylabel("Mean Satisfaction")
jt_pstr = f"{J_p:.4f}" if not np.isnan(J_p) else "n/a"
jt_zstr = f"{J_z:.3f}" if not np.isnan(J_z) else "n/a"
jt_jstr = f"{J_stat:.0f}" if not np.isnan(J_stat) else "n/a"
ax_I.set_title(f"(I) Jonckheere Trend\nJ={jt_jstr}, z={jt_zstr}, p={jt_pstr}")

fig8.suptitle("Participation & Satisfaction: Comprehensive Analysis",
              fontsize=14, fontweight="bold", y=0.99)
fig8.savefig(os.path.join(RESULTS_DIR, "fig8_comprehensive_results.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig8)
print("Saved fig8_comprehensive_results.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: SUMMARY AND SAVE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9: SUMMARY AND SAVE")
print("=" * 70)


def safe_round(val, ndigits=4):
    try:
        if np.isnan(val):
            return "n/a"
        return round(float(val), ndigits)
    except Exception:
        return str(val)


summary_rows = [
    # Descriptive
    {"Category": "Descriptive", "Metric": "n (df)", "Value": len(df), "Notes": "After removing flag==1"},
    {"Category": "Descriptive", "Metric": "n (df_sat)", "Value": len(df_sat), "Notes": "Satisfaction non-NaN"},
    {"Category": "Descriptive", "Metric": "n (df_full)", "Value": len(df_full), "Notes": "All 5 vars non-NaN"},
    {"Category": "Descriptive", "Metric": "Mean Q25", "Value": safe_round(q25_clean.mean()), "Notes": ""},
    {"Category": "Descriptive", "Metric": "SD Q25", "Value": safe_round(q25_clean.std()), "Notes": ""},
    # Ceiling/Floor
    {"Category": "Ceiling/Floor", "Metric": "Prop Q24=0 (floor)", "Value": safe_round(prop_q24_0), "Notes": ""},
    {"Category": "Ceiling/Floor", "Metric": "Prop Q24=0+1", "Value": safe_round(prop_q24_01), "Notes": ""},
    {"Category": "Ceiling/Floor", "Metric": "Prop Q25=5 (ceiling)", "Value": safe_round(prop_q25_5), "Notes": ""},
    {"Category": "Ceiling/Floor", "Metric": "Prop Q25>=4", "Value": safe_round(prop_q25_45), "Notes": ""},
    {"Category": "Ceiling/Floor", "Metric": "OLS β (Q24→Q25)", "Value": safe_round(ols_beta_q24), "Notes": ""},
    {"Category": "Ceiling/Floor", "Metric": "Tobit β (Q24→Q25)", "Value": safe_round(tobit_beta_q24), "Notes": ""},
    # Non-parametric
    {"Category": "Non-Parametric", "Metric": "Shapiro-Wilk W", "Value": safe_round(sw_stat), "Notes": "Q25"},
    {"Category": "Non-Parametric", "Metric": "Shapiro-Wilk p", "Value": safe_round(sw_p), "Notes": "Q25"},
    {"Category": "Non-Parametric", "Metric": "Kruskal-Wallis H", "Value": safe_round(kw_stat), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "Kruskal-Wallis p", "Value": safe_round(kw_p), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "Epsilon-squared", "Value": safe_round(epsilon_sq), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "Spearman rho (Q24-Q25)", "Value": safe_round(sp_r_q24), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "Spearman p (Q24-Q25)", "Value": safe_round(sp_p_q24), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "Kendall tau (Q24-Q25)", "Value": safe_round(kd_t_q24), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "JT J stat", "Value": safe_round(J_stat, 1), "Notes": "Jonckheere-Terpstra"},
    {"Category": "Non-Parametric", "Metric": "JT z", "Value": safe_round(J_z), "Notes": ""},
    {"Category": "Non-Parametric", "Metric": "JT p (one-tailed)", "Value": safe_round(J_p), "Notes": ""},
    # PLS-SEM
    {"Category": "PLS-SEM", "Metric": "Path coef (PART→SAT)", "Value": safe_round(path_coef), "Notes": ""},
    {"Category": "PLS-SEM", "Metric": "R2 (inner)", "Value": safe_round(r2_inner), "Notes": ""},
    {"Category": "PLS-SEM", "Metric": "Q2 (blindfolding d=7)", "Value": safe_round(q2), "Notes": ""},
    {"Category": "PLS-SEM", "Metric": "f2", "Value": safe_round(f2), "Notes": ""},
    {"Category": "PLS-SEM", "Metric": "Bootstrap CI low", "Value": safe_round(ci_low), "Notes": "95%"},
    {"Category": "PLS-SEM", "Metric": "Bootstrap CI high", "Value": safe_round(ci_high), "Notes": "95%"},
    {"Category": "PLS-SEM", "Metric": "OW q22", "Value": safe_round(outer_weights[0]), "Notes": "Outer weight"},
    {"Category": "PLS-SEM", "Metric": "OW info_score", "Value": safe_round(outer_weights[1]), "Notes": "Outer weight"},
    {"Category": "PLS-SEM", "Metric": "OW consult_score", "Value": safe_round(outer_weights[2]), "Notes": "Outer weight"},
    # Expectation Gap
    {"Category": "ExpGap", "Metric": "Cohens d (Q24=0 vs 2)", "Value": safe_round(cohens_d), "Notes": ""},
    {"Category": "ExpGap", "Metric": "Verdict", "Value": eg_verdict, "Notes": ""},
    {"Category": "ExpGap", "Metric": "Levene W", "Value": safe_round(lev_stat), "Notes": ""},
    {"Category": "ExpGap", "Metric": "Levene p", "Value": safe_round(lev_p), "Notes": ""},
    # XGBoost
    {"Category": "XGBoost", "Metric": "CV R2 (5-fold)", "Value": safe_round(xgb_r2_cv), "Notes": ""},
    {"Category": "XGBoost", "Metric": "CV RMSE (5-fold)", "Value": safe_round(xgb_rmse_cv), "Notes": ""},
]

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, "results_full_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved summary to {csv_path}")
print("\n=== PIPELINE COMPLETE ===")
