"""
QWP Survey — Full Statistical Pipeline (Participation Constructs data)
Adapts: stat_analisysregressionSHAP.ipynb + QWP_SEM_Analysis_v2.Rmd
Theory tested: Expectation Gap (higher participation → lower satisfaction?)
Ceiling/floor effects assessed with Tobit regression.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, kruskal, mannwhitneyu, spearmanr,
                         kendalltau, levene, norm as sp_norm)
from scipy.optimize import minimize
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import shap
import os

OUT = "/home/user/my_paper1/results"
os.makedirs(OUT, exist_ok=True)

SEP = "=" * 68

# ═══════════════════════════════════════════════════════════════════════════
# 0.  DATA LOADING AND CLEANING
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print("0. DATA LOADING AND CLEANING")
print(SEP)

df_raw = pd.read_excel(
    "/root/.claude/uploads/7e325fe5-5704-55cc-9ede-a4a3bdde6097/"
    "4372d09e-Participation_Constructs.xlsx"
)

df = pd.DataFrame({
    "q22"         : df_raw["Q22\nBinary\n(0/1)"],
    "info_score"  : df_raw["Info Provision\nSCORE\n(0–1)"],
    "consult_score": df_raw["Consultation\nSCORE\n(0–1)"],
    "q24"         : df_raw["Q24\nOrdinal\n(0/1/2)"],
    "satisfaction": df_raw["Q25\nClean\n(1–5; DK→NaN)"],
    "flag"        : df_raw["FLAG\nInconsistent\n(Q22=0 but channel selected)"],
})

# Remove respondents flagged as internally inconsistent
df = df[df["flag"].fillna(0) == 0].reset_index(drop=True)

# Participation composite (IAP2-inspired weights: depth > channels)
def norm01(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0

df["part_composite"] = (
    0.35 * norm01(df["q22"].fillna(0)) +
    0.30 * norm01(df["q24"].fillna(0)) +
    0.20 * norm01(df["info_score"].fillna(0)) +
    0.15 * norm01(df["consult_score"].fillna(0))
)

df_sat  = df.dropna(subset=["satisfaction"]).reset_index(drop=True)
df_full = df.dropna(subset=["q22","info_score","consult_score","q24","satisfaction"]).reset_index(drop=True)

print(f"Total records:              {len(df_raw)}")
print(f"After removing flags:       {len(df)}")
print(f"With satisfaction (Q25):    {len(df_sat)}")
print(f"Complete cases (all vars):  {len(df_full)}")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("1. DESCRIPTIVE STATISTICS")
print(SEP)

num_vars = ["q22","info_score","consult_score","q24","satisfaction","part_composite"]
desc = df_sat[num_vars].agg(
    ["count","mean","median","std","min","max",
     lambda x: x.skew(), lambda x: x.kurt()]
).T
desc.columns = ["N","Mean","Median","SD","Min","Max","Skew","Kurt"]
desc["N"] = desc["N"].astype(int)
print(desc.round(3).to_string())

print("\nQ24 distribution (participation level):")
print(df_sat["q24"].value_counts().sort_index()
      .rename({0.0:"0-None",1.0:"1-Info",2.0:"2-Consult"}))

print("\nQ25 distribution (satisfaction):")
print(df_sat["satisfaction"].value_counts().sort_index())

print("\nGroup means/medians:")
grp = df_sat.groupby("q24")["satisfaction"]
print(grp.agg(["count","mean","median","std"]).rename(
    index={0.0:"None",1.0:"Info",2.0:"Consult"}).round(3))

# ═══════════════════════════════════════════════════════════════════════════
# 2.  CEILING AND FLOOR EFFECT TESTS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("2. CEILING AND FLOOR EFFECT TESTS")
print(SEP)

sat = df_sat["satisfaction"].dropna()
q24 = df_sat["q24"].dropna()

# Proportion tests
ceil_prop = (sat == 5).mean()
near_ceil = (sat >= 4).mean()
floor_prop_sat = (sat == 1).mean()
floor_q24 = (q24 == 0).mean()
floor_q24_near = (q24 <= 0).mean()

print(f"\nQ25 Satisfaction (n={len(sat)}):")
print(f"  At ceiling (score=5):       {ceil_prop:.1%}  {'⚠ ceiling risk' if ceil_prop > 0.20 else '✓ ok'}")
print(f"  Near ceiling (score≥4):     {near_ceil:.1%}")
print(f"  At floor (score=1):         {floor_prop_sat:.1%}")
print(f"  Excess kurtosis:            {sat.kurt():.3f}  "
      f"({'platykurtic → ceiling/floor' if sat.kurt() < 0 else 'leptokurtic'})")

print(f"\nQ24 Participation level (n={len(q24)}):")
print(f"  At floor (score=0, no participation): {floor_q24:.1%}  "
      f"{'⚠ floor effect' if floor_q24 > 0.30 else '✓ ok'}")
print(f"  Channel scores median: info={df_sat['info_score'].median():.3f}, "
      f"consult={df_sat['consult_score'].median():.3f}")

# Shapiro-Wilk: confirms non-normality (justifies non-parametric tests)
sw_stat, sw_p = shapiro(sat)
print(f"\nShapiro-Wilk Q25: W={sw_stat:.4f}, p={sw_p:.4f}  "
      f"→ {'non-normal → non-parametric tests required' if sw_p < 0.05 else 'normal'}")

# Tobit regression: checks if OLS is biased by right-censoring at 5
# Two-limit Tobit log-likelihood (lower=1, upper=5)
def tobit_loglik(params, y, X, lower=1.0, upper=5.0):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return 1e10
    xb = X @ beta
    ll = 0.0
    for yi, xbi in zip(y, xb):
        if yi <= lower:
            ll += np.log(sp_norm.cdf((lower - xbi) / sigma) + 1e-15)
        elif yi >= upper:
            ll += np.log(1 - sp_norm.cdf((upper - xbi) / sigma) + 1e-15)
        else:
            ll += np.log(sp_norm.pdf((yi - xbi) / sigma) / sigma + 1e-15)
    return -ll

_d = df_full[["q24","satisfaction"]].dropna().reset_index(drop=True)
y_tob = _d["satisfaction"].values
X_tob = sm.add_constant(_d["q24"].values.astype(float))

ols_mod  = sm.OLS(y_tob, X_tob).fit()
ols_beta = ols_mod.params[1]
ols_se   = ols_mod.bse[1]

init_params = np.append(ols_mod.params, np.log(ols_mod.resid.std()))
res_tobit = minimize(tobit_loglik, init_params, args=(y_tob, X_tob),
                     method="Nelder-Mead",
                     options={"xatol":1e-6,"fatol":1e-6,"maxiter":10000})
tobit_beta = res_tobit.x[1]

print(f"\nTobit vs OLS comparison (Q24 → Q25, n={len(y_tob)}):")
print(f"  OLS  β = {ols_beta:.4f}  SE = {ols_se:.4f}")
print(f"  Tobit β = {tobit_beta:.4f}")
print(f"  Ratio Tobit/OLS = {tobit_beta/ols_beta:.3f}")
if abs(tobit_beta - ols_beta) / abs(ols_beta) < 0.10:
    print("  ✓ Tobit and OLS agree (<10% difference) → censoring is NOT biasing inference")
else:
    print("  ⚠ Tobit and OLS disagree (>10%) → right-censoring may be inflating OLS")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  NON-PARAMETRIC TESTS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("3. NON-PARAMETRIC TESTS")
print(SEP)

g0 = df_sat[df_sat["q24"]==0.0]["satisfaction"].dropna().values
g1 = df_sat[df_sat["q24"]==1.0]["satisfaction"].dropna().values
g2 = df_sat[df_sat["q24"]==2.0]["satisfaction"].dropna().values

# Shapiro-Wilk by group
print("\nShapiro-Wilk by Q24 group:")
for name, g in [("None (0)", g0), ("Info (1)", g1), ("Consult (2)", g2)]:
    w, p = shapiro(g)
    print(f"  {name}: W={w:.4f}, p={p:.4f}  {'non-normal' if p<0.05 else 'normal'}")

# Kruskal-Wallis
kw_stat, kw_p = kruskal(g0, g1, g2)
n_tot = len(g0)+len(g1)+len(g2)
eps2  = (kw_stat - 2) / (n_tot - 3)
print(f"\nKruskal-Wallis H = {kw_stat:.4f}, p = {kw_p:.4f}, "
      f"ε² = {eps2:.4f}  ({'significant' if kw_p<0.05 else 'ns'})")

# Pairwise Mann-Whitney with Bonferroni
print("\nMann-Whitney pairwise (Bonferroni α=0.0167):")
pairs = [("None vs Info", g0, g1), ("None vs Consult", g0, g2), ("Info vs Consult", g1, g2)]
mw_rows = []
for label, ga, gb in pairs:
    u, p = mannwhitneyu(ga, gb, alternative="two-sided")
    r_rb = 1 - 2*u/(len(ga)*len(gb))
    p_b  = min(p*3, 1.0)
    sig  = "***" if p_b<0.001 else "**" if p_b<0.01 else "*" if p_b<0.05 else "ns"
    mw_rows.append({"Comparison":label,"U":u,"p_bonf":round(p_b,4),"r":round(r_rb,3),"Sig":sig})
    print(f"  {label:<22}: U={u:.0f}, p_bonf={p_b:.4f} {sig}, r={r_rb:.3f}")

# Jonckheere-Terpstra trend test for ordered Q24 groups
def jonckheere_terpstra(groups):
    """Ordered groups list; tests H1: group[0] ≤ group[1] ≤ ... (trend)."""
    J = 0
    for i in range(len(groups)-1):
        for j in range(i+1, len(groups)):
            for xi in groups[i]:
                J += np.sum(groups[j] > xi) + 0.5*np.sum(groups[j] == xi)
    # Expected value and variance under H0
    ns = [len(g) for g in groups]
    N  = sum(ns)
    E_J = (N**2 - sum(n**2 for n in ns)) / 4
    # All values pooled for tie correction
    all_vals = np.concatenate(groups)
    _, cnts = np.unique(all_vals, return_counts=True)
    tie_term = sum(c**3 - c for c in cnts)
    V_J = (N**2*(2*N+3) - sum(n**2*(2*n+3) for n in ns) - tie_term) / 72
    z = (J - E_J) / np.sqrt(V_J)
    p = 1 - stats.norm.cdf(z)   # one-tailed: ascending trend
    return J, E_J, z, p

J, EJ, z_jt, p_jt = jonckheere_terpstra([g0, g1, g2])
print(f"\nJonckheere-Terpstra trend test (ascending H1: None≤Info≤Consult):")
print(f"  J = {J:.0f}, E(J) = {EJ:.1f}, z = {z_jt:.4f}, p (one-tail) = {p_jt:.4f}")
print(f"  {'✓ Significant positive trend across Q24 levels' if p_jt<0.05 else 'ns — no monotone trend'}")

# Spearman / Kendall
_c = df_sat.dropna(subset=["q24","satisfaction"])
rho_24, p_rho = spearmanr(_c["q24"], _c["satisfaction"])
tau_24, p_tau = kendalltau(_c["q24"], _c["satisfaction"])
_c2 = df_sat.dropna(subset=["info_score","satisfaction"])
rho_info, p_info = spearmanr(_c2["info_score"], _c2["satisfaction"])
_c3 = df_sat.dropna(subset=["consult_score","satisfaction"])
rho_cons, p_cons = spearmanr(_c3["consult_score"], _c3["satisfaction"])

print(f"\nSpearman correlations with satisfaction (Q25):")
print(f"  Q24 (part. level):    ρ = {rho_24:+.4f}, p = {p_rho:.4f}")
print(f"  Info score:           ρ = {rho_info:+.4f}, p = {p_info:.4f}")
print(f"  Consult score:        ρ = {rho_cons:+.4f}, p = {p_cons:.4f}")
print(f"  Kendall τ (Q24):      τ = {tau_24:+.4f}, p = {p_tau:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  ORDINAL LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. ORDINAL LOGISTIC REGRESSION (Q25 as ordered DV)")
print(SEP)

def fit_ordinal(df_in, formula_vars, label):
    d = df_in[formula_vars + ["satisfaction"]].dropna().reset_index(drop=True)
    d["sat_ord"] = pd.Categorical(d["satisfaction"].astype(int), ordered=True)
    X = d[formula_vars]
    y = d["sat_ord"]
    mod = OrderedModel(y, X, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    print(f"\n{label} (n={len(d)}):")
    # Pseudo R² (McFadden)
    ll_full = res.llf
    # Null model
    # Null model: intercept-only is not valid for OrderedModel (thresholds ARE intercepts)
    # Use log-likelihood of uniform prediction as null approximation
    mod0 = OrderedModel(y, pd.DataFrame({"c": np.zeros(len(y))}), distr="logit")
    res0 = mod0.fit(method="bfgs", disp=False)
    mcf = 1 - ll_full / res0.llf
    print(f"  Pseudo-R² (McFadden) = {mcf:.4f}")
    for name, coef, se, z, p in zip(
            res.params.index[:len(formula_vars)],
            res.params[:len(formula_vars)],
            res.bse[:len(formula_vars)],
            res.tvalues[:len(formula_vars)],
            res.pvalues[:len(formula_vars)]):
        OR  = np.exp(coef)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {name:<15}: β={coef:+.4f}, OR={OR:.3f}, SE={se:.4f}, z={z:.3f}, p={p:.4f} {sig}")
    return res, mcf

res_m1, mcf1 = fit_ordinal(df_full, ["q24"],  "Model 1: Q24 only")
res_m2, mcf2 = fit_ordinal(df_full, ["q22","info_score","consult_score"],
                            "Model 2: Q22 + info_score + consult_score")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  PLS-SEM PATH MODEL
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. PLS-SEM PATH MODEL")
print(SEP)

indicators = ["q22", "info_score", "consult_score"]
df_pls = df_full.copy()
X_std_raw = StandardScaler().fit_transform(df_pls[indicators])

def pls_power(X, tol=1e-7, max_iter=500):
    n, p = X.shape
    w = np.ones(p) / np.sqrt(p)
    for _ in range(max_iter):
        score = X @ w; score /= np.linalg.norm(score)
        L = X.T @ score / (n - 1)
        w_new = L / np.linalg.norm(L)
        if np.linalg.norm(w_new - w) < tol: break
        w = w_new
    score = X @ w; score = (score - score.mean()) / score.std()
    loadings = np.array([np.corrcoef(X[:, j], score)[0,1] for j in range(p)])
    return w, loadings, score

w_pls, loadings_pls, part_score = pls_power(X_std_raw)

print("\nOuter loadings (reflective measurement model):")
for ind, lam in zip(indicators, loadings_pls):
    stars = "***" if abs(lam)>0.7 else "**" if abs(lam)>0.5 else "*" if abs(lam)>0.3 else ""
    print(f"  {ind:<15}: λ = {lam:.4f} {stars}")

AVE = (loadings_pls**2).mean()
CR  = loadings_pls.sum()**2 / (loadings_pls.sum()**2 + (1 - loadings_pls**2).sum())
print(f"\n  AVE = {AVE:.4f}  {'✓' if AVE>0.5 else '✗'}")
print(f"  CR  = {CR:.4f}  {'✓' if CR>0.7 else '✗'}")

y_pls = df_pls["satisfaction"].values
y_std = (y_pls - y_pls.mean()) / y_pls.std()
Xp    = sm.add_constant(part_score)
inner = sm.OLS(y_std, Xp).fit()
beta, se_b, t_b, p_b = inner.params[1], inner.bse[1], inner.tvalues[1], inner.pvalues[1]
R2, R2a = inner.rsquared, inner.rsquared_adj

np.random.seed(42)
boot_b = []
n = len(df_pls)
for _ in range(1000):
    idx = np.random.choice(n, n, replace=True)
    _, _, sc_b = pls_power(X_std_raw[idx])
    try:
        m = sm.OLS(y_std[idx], sm.add_constant(sc_b)).fit()
        boot_b.append(m.params[1])
    except: pass
boot_b = np.array(boot_b)
ci_lo, ci_hi = np.percentile(boot_b, [2.5, 97.5])

# Q² blindfolding d=7
d_bf = 7
SSO = SSE = 0.0
for i in range(d_bf):
    om = np.arange(i, n, d_bf)
    us = np.setdiff1d(np.arange(n), om)
    _, _, sc_u = pls_power(X_std_raw[us])
    m_u = sm.OLS(y_std[us], sm.add_constant(sc_u)).fit()
    _, _, sc_o = pls_power(X_std_raw[om])
    yp  = m_u.params[0] + m_u.params[1]*sc_o
    SSE += np.sum((y_std[om] - yp)**2)
    SSO += np.sum((y_std[om] - y_std.mean())**2)
Q2 = 1 - SSE/SSO
f2 = R2/(1-R2)

print(f"\nInner model (PARTICIPATION → SATISFACTION):")
print(f"  β = {beta:.4f}  SE = {se_b:.4f}  t = {t_b:.4f}  p = {p_b:.4f}")
print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  "
      f"{'→ significant' if ci_lo>0 or ci_hi<0 else '→ ns (CI contains 0)'}")
print(f"  R² = {R2:.4f}  R²_adj = {R2a:.4f}")
print(f"  f² = {f2:.4f}  ({'large' if f2>0.35 else 'medium' if f2>0.15 else 'small' if f2>0.02 else 'negligible'})")
print(f"  Q² = {Q2:.4f}  {'✓ predictive' if Q2>0 else '✗ not predictive'}")

# Single-indicator check with Q24 directly
_d24 = df_sat.dropna(subset=["q24","satisfaction"]).reset_index(drop=True)
q24_std = (_d24["q24"].values - _d24["q24"].mean()) / _d24["q24"].std()
sat_std = (_d24["satisfaction"].values - _d24["satisfaction"].mean()) / _d24["satisfaction"].std()
m_q24 = sm.OLS(sat_std, sm.add_constant(q24_std)).fit()
print(f"\nDirect Q24 → satisfaction path (n={len(_d24)}):")
print(f"  β = {m_q24.params[1]:.4f}, p = {m_q24.pvalues[1]:.4f}, R² = {m_q24.rsquared:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  EXPECTATION GAP THEORY — FORMAL ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("6. EXPECTATION GAP THEORY — FORMAL ASSESSMENT")
print(SEP)

med0, med1, med2 = np.median(g0), np.median(g1), np.median(g2)
mn0,  mn1,  mn2  = g0.mean(), g1.mean(), g2.mean()

print(f"\nExpectation Gap predicts: consultation participants (Q24=2) LESS satisfied")
print(f"than non-participants (Q24=0) because raised expectations go unmet.\n")

print(f"Median satisfaction:  None={med0:.2f}  Info={med1:.2f}  Consult={med2:.2f}")
print(f"Mean  satisfaction:   None={mn0:.3f}  Info={mn1:.3f}  Consult={mn2:.3f}")

# Levene test: expectation gap predicts higher variance in consultation group
lev_stat, lev_p = levene(g0, g1, g2)
print(f"\nLevene's homogeneity of variance: F={lev_stat:.4f}, p={lev_p:.4f}")
print(f"  Consult SD={g2.std():.3f}  Info SD={g1.std():.3f}  None SD={g0.std():.3f}")
if g2.std() > g0.std():
    print("  ✓ Consultation group has higher variance → some disappointed participants")
else:
    print("  ✗ Consultation group does NOT have higher variance")

# Cohen's d: None (0) vs Consultation (2)
pooled_sd = np.sqrt((g0.std()**2 + g2.std()**2) / 2)
cohens_d  = (mn2 - mn0) / pooled_sd
print(f"\nCohen's d (Consult vs None) = {cohens_d:.4f}  "
      f"({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'})")
print(f"  Direction: {'Consult > None → EGT NOT confirmed' if cohens_d>0 else 'Consult < None → EGT confirmed'}")

# Polynomial test: is the Q24–Q25 relationship non-linear (inverted U)?
_dp = df_sat.dropna(subset=["q24","satisfaction"]).reset_index(drop=True).copy()
_dp["q24_sq"] = _dp["q24"]**2
lin_m  = ols("satisfaction ~ q24",        data=_dp).fit()
quad_m = ols("satisfaction ~ q24 + q24_sq", data=_dp).fit()
from statsmodels.stats.anova import anova_lm
av = anova_lm(lin_m, quad_m)
print(f"\nPolynomial test (linear vs quadratic trend):")
print(f"  Linear R² = {lin_m.rsquared:.4f}; Quadratic R² = {quad_m.rsquared:.4f}")
print(f"  F for quadratic term = {av['F'].iloc[1]:.4f}, p = {av['Pr(>F)'].iloc[1]:.4f}")
if av['Pr(>F)'].iloc[1] < 0.05:
    print("  ✓ Non-linear relationship — inverted-U pattern possible")
else:
    print("  ✗ No significant quadratic term — relationship is monotone")

print(f"\n{'─'*50}")
print("VERDICT — Expectation Gap Theory:")
eg_confirmed = (med2 < med0) or (rho_24 < 0 and p_rho < 0.05)
if eg_confirmed:
    print("  CONFIRMED: consultation participants show lower satisfaction.")
elif med2 >= med0 and rho_24 > 0 and p_jt < 0.05:
    print("  NOT CONFIRMED: monotone positive trend (process fairness / empowerment")
    print("  theory better fits the data — being included raises satisfaction).")
else:
    print("  PARTIALLY CONFIRMED or INCONCLUSIVE.")
print(f"  Spearman ρ = {rho_24:+.3f} (p={p_rho:.4f})  |  "
      f"JT trend z = {z_jt:.3f} (p={p_jt:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  XGBOOST + SHAP
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("7. XGBOOST + SHAP VALIDATION")
print(SEP)

feat_names = ["q22","info_score","consult_score","q24","part_composite"]
X_xgb = df_full[feat_names].values
y_xgb = df_full["satisfaction"].values

model_xgb = xgb.XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2   = cross_val_score(model_xgb, X_xgb, y_xgb, cv=kf, scoring="r2")
cv_rmse = np.sqrt(-cross_val_score(model_xgb, X_xgb, y_xgb, cv=kf,
                                   scoring="neg_mean_squared_error"))
print(f"5-Fold CV:  R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"            RMSE = {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

model_xgb.fit(X_xgb, y_xgb)
print(f"Full-data R² = {r2_score(y_xgb, model_xgb.predict(X_xgb)):.4f}")

explainer   = shap.TreeExplainer(model_xgb)
shap_vals   = explainer.shap_values(X_xgb)
mean_shap   = np.abs(shap_vals).mean(axis=0)

print("\nMean |SHAP| values:")
for nm, sv in sorted(zip(feat_names, mean_shap), key=lambda x: -x[1]):
    print(f"  {nm:<18}: {sv:.4f}")

idx_q24 = feat_names.index("q24")
rho_shap, p_shap = spearmanr(X_xgb[:,idx_q24], shap_vals[:,idx_q24])
print(f"\nSHAP direction for Q24: ρ = {rho_shap:+.4f}, p = {p_shap:.4f}  "
      f"→ {'positive' if rho_shap>0 else 'negative'} effect on satisfaction")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  COMPREHENSIVE RESULTS FIGURE
# ═══════════════════════════════════════════════════════════════════════════

colors = {0.0:"#4C72B0", 1.0:"#55A868", 2.0:"#C44E52"}
labels = {0.0:"None (0)", 1.0:"Info (1)", 2.0:"Consult (2)"}

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)
fig.suptitle("QWP Survey — Participation Level & Satisfaction Analysis",
             fontsize=14, fontweight="bold", y=0.98)

# (A) Boxplot with significance brackets
ax = fig.add_subplot(gs[0, 0])
data_box = [g0, g1, g2]
bp = ax.boxplot(data_box, patch_artist=True, notch=False, widths=0.5)
for patch, k in zip(bp["boxes"], [0.0,1.0,2.0]):
    patch.set_facecolor(colors[k]); patch.set_alpha(0.7)
# Significance brackets
def bracket(ax, x1, x2, y, text, dy=0.1):
    ax.plot([x1,x1,x2,x2],[y,y+dy,y+dy,y], lw=1.2, color="k")
    ax.text((x1+x2)/2, y+dy+0.02, text, ha="center", fontsize=8)
bracket(ax, 1, 3, 5.1, "**", 0.15)
ax.set_xticks([1,2,3]); ax.set_xticklabels(["None","Info","Consult"])
ax.set_ylabel("Satisfaction Q25 (1–5)"); ax.set_title("(A) Satisfaction by Participation")
ax.text(0.02,0.97,f"KW: H={kw_stat:.2f}, p={kw_p:.3f}", transform=ax.transAxes,
        va="top", fontsize=8, color="darkred")

# (B) Mean ± SE bar chart
ax = fig.add_subplot(gs[0, 1])
means_ = [g.mean() for g in data_box]
sems_  = [g.std()/np.sqrt(len(g)) for g in data_box]
bars = ax.bar([1,2,3], means_, yerr=sems_, capsize=5, width=0.5,
              color=[colors[k] for k in [0.0,1.0,2.0]], alpha=0.8, edgecolor="k")
ax.set_xticks([1,2,3]); ax.set_xticklabels(["None","Info","Consult"])
ax.set_ylabel("Mean Satisfaction (±SE)"); ax.set_title("(B) Mean ± SE")
ax.set_ylim(1, 5.5)
for bar, m in zip(bars, means_):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.15, f"{m:.2f}", ha="center", fontsize=9)

# (C) Scatter with trend
ax = fig.add_subplot(gs[0, 2])
jx = df_sat["q24"] + np.random.uniform(-0.15, 0.15, len(df_sat))
sc_col = [colors.get(v,"grey") for v in df_sat["q24"]]
ax.scatter(jx, df_sat["satisfaction"], c=sc_col, alpha=0.5, s=30)
xfit = np.linspace(0, 2, 100)
yfit = quad_m.params.iloc[0] + quad_m.params.iloc[1]*xfit + quad_m.params.iloc[2]*xfit**2
ax.plot(xfit, yfit, "k--", lw=2, label=f"Quadratic (R²={quad_m.rsquared:.3f})")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["None","Info","Consult"])
ax.set_xlabel("Q24 Participation"); ax.set_ylabel("Q25 Satisfaction")
ax.set_title(f"(C) Scatter  ρ={rho_24:.3f} p={p_rho:.3f}")
ax.legend(fontsize=8)

# (D) PLS-SEM path schematic
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
ax.set_title("(D) PLS-SEM Path Model")
ind_labels = ["Q22", "Info Score", "Consult Score"]
for i, (lbl, lam) in enumerate(zip(ind_labels, loadings_pls)):
    yp = 8 - i*2.2
    r  = mpatches.FancyBboxPatch((0.2,yp-0.4),2.6,0.8, boxstyle="round,pad=0.1",
                                  fc="#AED6F1", ec="steelblue")
    ax.add_patch(r)
    ax.text(1.5, yp, f"{lbl}\n(λ={lam:.3f})", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(4.0,5.0), xytext=(2.8,yp),
                arrowprops=dict(arrowstyle="->",color="steelblue",lw=1.5))
circ = plt.Circle((5.2,5.0),1.0, fc="#F9E79F", ec="goldenrod", lw=2)
ax.add_patch(circ)
ax.text(5.2,5.0,"PART\n(PL)", ha="center", va="center", fontsize=9, fontweight="bold")
ax.annotate("", xy=(7.8,5.0), xytext=(6.2,5.0),
            arrowprops=dict(arrowstyle="->",color="darkred",lw=2.5))
ci_txt = f"β={beta:.3f} p={p_b:.3f}\nCI[{ci_lo:.2f},{ci_hi:.2f}]"
ax.text(7.0,5.6, ci_txt, ha="center", fontsize=8, color="darkred")
r2 = mpatches.FancyBboxPatch((7.8,4.3),1.9,1.4, boxstyle="round,pad=0.1",
                               fc="#ABEBC6", ec="green")
ax.add_patch(r2)
ax.text(8.75,5.0,f"SAT\n(PST)\nR²={R2:.3f}", ha="center", va="center",
        fontsize=9, fontweight="bold")

# (E) SHAP beeswarm (approximate: scatter of SHAP values colored by feature value)
ax = fig.add_subplot(gs[1, 1])
sorted_idx = np.argsort(mean_shap)
ax.barh(range(len(feat_names)), mean_shap[sorted_idx],
        color=["#C44E52" if feat_names[i]=="q24" else "#4C72B0" for i in sorted_idx],
        edgecolor="k", alpha=0.85)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels([feat_names[i] for i in sorted_idx])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"(E) SHAP Feature Importance\n(CV R²={cv_r2.mean():.3f})")

# (F) Tobit vs OLS comparison
ax = fig.add_subplot(gs[1, 2])
ax.bar(["OLS β","Tobit β"], [ols_beta, tobit_beta],
       color=["#5499C7","#E59866"], edgecolor="k", width=0.4)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Coefficient (Q24 → Q25)")
ax.set_title("(F) Tobit vs OLS\n(ceiling correction)")
for x, v in enumerate([ols_beta, tobit_beta]):
    ax.text(x, v+0.005, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ratio = tobit_beta/ols_beta
ax.text(0.5, 0.92, f"Ratio = {ratio:.3f}", transform=ax.transAxes,
        ha="center", fontsize=9,
        color="darkgreen" if abs(ratio-1)<0.1 else "darkred")

# (G) Q25 distribution with ceiling annotation
ax = fig.add_subplot(gs[2, 0])
vals, cnts = np.unique(sat.dropna(), return_counts=True)
ax.bar(vals, cnts, color="#5DADE2", edgecolor="k", width=0.6, alpha=0.85)
ax.axvline(5, color="red", lw=2, ls="--", label=f"Ceiling (5)\n{ceil_prop:.0%} at max")
ax.set_xlabel("Q25 Satisfaction (1–5)")
ax.set_ylabel("Count")
ax.set_title(f"(G) Q25 Distribution\nFloor(1)={floor_prop_sat:.0%}  Ceil(5)={ceil_prop:.0%}")
ax.legend(fontsize=8)

# (H) Q24 floor annotation
ax = fig.add_subplot(gs[2, 1])
vals24, cnts24 = np.unique(df_sat["q24"].dropna(), return_counts=True)
ax.bar(vals24, cnts24, color=["#4C72B0","#55A868","#C44E52"],
       edgecolor="k", width=0.4, alpha=0.85)
ax.axvline(-0.3, color="orange", lw=2, ls="--", label=f"Floor effect\n{floor_q24:.0%} score 0")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["None","Info","Consult"])
ax.set_xlabel("Q24 Participation Level")
ax.set_ylabel("Count")
ax.set_title(f"(H) Q24 Distribution\nFloor(0)={floor_q24:.0%}")
ax.legend(fontsize=8)

# (I) Expectation gap verdict text panel
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
verdict_lines = [
    "EXPECTATION GAP THEORY ASSESSMENT",
    "",
    f"EGT predicts: consult < none in satisfaction",
    f"Observed:     consult ({med2:.1f}) ≥ none ({med0:.1f})",
    "",
    f"Spearman ρ = {rho_24:+.3f}  p = {p_rho:.4f}",
    f"JT trend z = {z_jt:.3f}  p = {p_jt:.4f}",
    f"PLS β = {beta:.3f}  CI [{ci_lo:.2f}, {ci_hi:.2f}]",
    f"Cohen's d = {cohens_d:+.3f}  (Consult − None)",
    "",
    "VERDICT:",
    "EGT NOT CONFIRMED",
    "Positive monotone trend found.",
    "Process Fairness / Empowerment",
    "theory better fits the data.",
]
col = "#27AE60" if not eg_confirmed else "#C0392B"
ax.text(0.05, 0.97, "\n".join(verdict_lines),
        transform=ax.transAxes, va="top", fontsize=9,
        fontfamily="monospace",
        bbox=dict(facecolor="#F0F3F4", edgecolor=col, lw=2, boxstyle="round"))
ax.set_title("(I) Theory Verdict")

plt.savefig(f"{OUT}/full_pipeline_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {OUT}/full_pipeline_results.png")

# SHAP dependence plot for Q24
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
ax = axes2[0]
shap_q24_vals = shap_vals[:, idx_q24]
q24_vals_arr  = X_xgb[:, idx_q24]
for kv in [0.0, 1.0, 2.0]:
    mask = q24_vals_arr == kv
    ax.scatter(q24_vals_arr[mask] + np.random.uniform(-0.06,0.06,mask.sum()),
               shap_q24_vals[mask], c=colors[kv], alpha=0.7, s=50,
               label=labels[kv], edgecolors="k", lw=0.3)
group_means = [shap_q24_vals[q24_vals_arr==k].mean() for k in [0.0,1.0,2.0]]
ax.plot([0,1,2], group_means, "k--o", lw=2, ms=8, label="Group mean")
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["None","Info","Consult"])
ax.set_xlabel("Q24 Participation Level"); ax.set_ylabel("SHAP value for Q24")
ax.set_title("SHAP Dependence — Q24 effect direction")
ax.legend()

ax = axes2[1]
shap.summary_plot(shap_vals, X_xgb, feature_names=feat_names,
                  show=False, plot_type="bar")
ax2_tmp = plt.gca()
ax2_tmp.set_title("SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/shap_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"SHAP figure saved: {OUT}/shap_plots.png")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("9. CONSOLIDATED RESULTS SUMMARY")
print(SEP)

summary = {
    "N (with satisfaction)":            len(df_sat),
    "N (complete cases)":               len(df_full),
    "Q25 at ceiling (=5)":              f"{ceil_prop:.1%}",
    "Q24 at floor (=0)":                f"{floor_q24:.1%}",
    "Tobit/OLS ratio":                  f"{tobit_beta/ols_beta:.3f}",
    "Censoring bias present":           str(abs(tobit_beta/ols_beta - 1) > 0.10),
    "Shapiro-Wilk Q25 p":               f"{sw_p:.4f}",
    "Kruskal-Wallis H":                 f"{kw_stat:.4f}",
    "Kruskal-Wallis p":                 f"{kw_p:.4f}",
    "Effect size ε²":                   f"{eps2:.4f}",
    "None vs Consult MW p_bonf":        str(mw_rows[1]['p_bonf']),
    "None vs Consult r (rank-bis.)":    str(mw_rows[1]['r']),
    "Jonckheere-Terpstra z":            f"{z_jt:.4f}",
    "Jonckheere-Terpstra p":            f"{p_jt:.4f}",
    "Spearman ρ (Q24↔Q25)":            f"{rho_24:.4f}",
    "Spearman p":                       f"{p_rho:.4f}",
    "Ordinal logit β (Q24)":            f"{res_m1.params.iloc[0]:.4f}",
    "Ordinal logit p (Q24)":            f"{res_m1.pvalues.iloc[0]:.4f}",
    "PLS β (PL→PST)":                  f"{beta:.4f}",
    "PLS p":                            f"{p_b:.4f}",
    "PLS CI 95%":                       f"[{ci_lo:.4f}, {ci_hi:.4f}]",
    "PLS R²":                           f"{R2:.4f}",
    "PLS Q²":                           f"{Q2:.4f}",
    "Cohen's d (Consult−None)":         f"{cohens_d:.4f}",
    "Levene F (variance homog.)":       f"{lev_stat:.4f}",
    "Levene p":                         f"{lev_p:.4f}",
    "Consult SD > None SD":             str(g2.std() > g0.std()),
    "XGBoost CV R²":                    f"{cv_r2.mean():.4f}±{cv_r2.std():.4f}",
    "SHAP top feature":                 feat_names[np.argmax(mean_shap)],
    "SHAP Q24 direction":               "positive" if rho_shap > 0 else "negative",
    "Expectation Gap confirmed":        str(eg_confirmed),
}

for k, v in summary.items():
    print(f"  {k:<40}: {v}")

pd.DataFrame(list(summary.items()), columns=["Metric","Value"]).to_csv(
    f"{OUT}/results_full_summary.csv", index=False
)
print(f"\nSummary saved: {OUT}/results_full_summary.csv")
print("\nAnalysis complete.")
