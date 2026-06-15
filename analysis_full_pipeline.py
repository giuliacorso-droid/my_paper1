import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import os
import warnings
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
os.makedirs('/home/user/my_paper1/results', exist_ok=True)

RESULTS_DIR = '/home/user/my_paper1/results'
DATA_PATH = '/root/.claude/uploads/7e325fe5-5704-55cc-9ede-a4a3bdde6097/4372d09e-Participation_Constructs.xlsx'

# ============================================================
# SECTION 0: DATA LOADING AND CLEANING
# ============================================================
print("=" * 60)
print("SECTION 0: DATA LOADING AND CLEANING")
print("=" * 60)

raw = pd.read_excel(DATA_PATH, engine='openpyxl')
print("Shape:", raw.shape)

# Explicit mapping — column names contain literal newlines from Excel
# Using multi-keyword match to be robust to minor header variations
def _find_col(df, *keywords):
    """Return first column whose lower-cased name contains ALL keywords."""
    for c in df.columns:
        cl = str(c).lower().replace("\n", " ")
        if all(kw in cl for kw in keywords):
            return c
    raise KeyError(f"No column matching {keywords!r}. Columns: {list(df.columns)}")

col_q22  = _find_col(raw, "q22", "binary")
col_info = _find_col(raw, "info provision", "score")
col_cons = _find_col(raw, "consultation", "score")
col_q24  = _find_col(raw, "q24", "ordinal")
col_sat  = _find_col(raw, "q25", "clean")
col_flag = _find_col(raw, "flag", "inconsistent")

print(f"Mapped: q22={col_q22!r}, info={col_info!r}, consult={col_cons!r}")
print(f"        q24={col_q24!r}, sat={col_sat!r}, flag={col_flag!r}")

df_raw = raw.rename(columns={
    col_q22: 'q22', col_info: 'info_score', col_cons: 'consult_score',
    col_q24: 'q24', col_sat: 'satisfaction', col_flag: 'flag',
})

needed = ['q22', 'info_score', 'consult_score', 'q24', 'satisfaction']

# Remove internally inconsistent records
df = df_raw[df_raw['flag'].fillna(0) != 1].reset_index(drop=True)

for c in needed:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df_sat  = df.dropna(subset=['satisfaction']).reset_index(drop=True)
df_full = df.dropna(subset=needed).reset_index(drop=True)

print(f"\nSample sizes:")
print(f"  Full cleaned df: n={len(df)}")
print(f"  With satisfaction: n={len(df_sat)}")
print(f"  Complete cases: n={len(df_full)}")

# ============================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 60)

desc_rows = []
for c in needed:
    s = df[c].dropna()
    desc_rows.append({
        'Variable': c,
        'N': len(s),
        'Mean': round(s.mean(), 3),
        'Median': round(s.median(), 3),
        'SD': round(s.std(), 3),
        'Min': round(s.min(), 3),
        'Max': round(s.max(), 3),
        'Skew': round(stats.skew(s), 3),
        'Kurtosis': round(stats.kurtosis(s), 3)
    })

desc_df = pd.DataFrame(desc_rows)
print("\nDescriptive Statistics:")
print(desc_df.to_string(index=False))

print("\nQ24 value counts:")
print(df['q24'].value_counts().sort_index())
print("\nSatisfaction (Q25) value counts:")
print(df['satisfaction'].value_counts().sort_index())

# ============================================================
# SECTION 2: CEILING AND FLOOR EFFECT TESTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: CEILING AND FLOOR EFFECT TESTS")
print("=" * 60)

q24_valid = df['q24'].dropna()
q25_valid = df_sat['satisfaction']

floor_q24_0     = (q24_valid == 0).mean()
floor_q24_01    = (q24_valid <= 1).mean()
ceiling_q25_5   = (q25_valid == 5).mean()
ceiling_q25_45  = (q25_valid >= 4).mean()

print(f"\nFloor effects (Q24):")
print(f"  Proportion at 0:   {floor_q24_0:.3f}")
print(f"  Proportion at 0+1: {floor_q24_01:.3f}")
print(f"\nCeiling effects (Q25):")
print(f"  Proportion at 5:   {ceiling_q25_5:.3f}")
print(f"  Proportion at 4+5: {ceiling_q25_45:.3f}")

kurt_q25 = stats.kurtosis(q25_valid)
print(f"\nKurtosis Q25: {kurt_q25:.3f}")
if abs(kurt_q25) > 1:
    print("  -> Non-normal distribution indicated (|kurtosis| > 1)")
else:
    print("  -> Roughly normal kurtosis")

# Tobit regression (two-limit, lower=1, upper=5)
def tobit_loglik(params, y, X, lo=1, hi=5):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    xb = X @ beta
    ll = 0.0
    for yi, xbi in zip(y, xb):
        if yi <= lo:
            ll += np.log(stats.norm.cdf((lo - xbi) / sigma) + 1e-15)
        elif yi >= hi:
            ll += np.log(1 - stats.norm.cdf((hi - xbi) / sigma) + 1e-15)
        else:
            ll += stats.norm.logpdf(yi, xbi, sigma)
    return -ll

tobit_data = df_full[['q24', 'satisfaction']].dropna().reset_index(drop=True)
y_t = tobit_data['satisfaction'].values
X_t = np.column_stack([np.ones(len(y_t)), tobit_data['q24'].values])

ols_res = np.linalg.lstsq(X_t, y_t, rcond=None)
beta_ols = ols_res[0]
print(f"\nOLS: intercept={beta_ols[0]:.3f}, beta_q24={beta_ols[1]:.3f}")

init_params = np.append(beta_ols, np.log(np.std(y_t)))
result = minimize(tobit_loglik, init_params, args=(y_t, X_t, 1, 5),
                  method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
beta_tobit = result.x[:-1]
print(f"Tobit: intercept={beta_tobit[0]:.3f}, beta_q24={beta_tobit[1]:.3f}")
print(f"  OLS beta={beta_ols[1]:.3f} vs Tobit beta={beta_tobit[1]:.3f}")
if abs(beta_tobit[1]) > abs(beta_ols[1]):
    print("  -> Tobit > OLS: ceiling/floor censoring attenuated OLS estimate")
else:
    print("  -> Tobit ~= OLS: censoring not severe")

# Figures
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Q25 histogram with normal overlay
ax = axes[0]
ax.hist(q25_valid, bins=np.arange(0.5, 6.5, 1), density=True, alpha=0.7, color='steelblue', edgecolor='white')
xn = np.linspace(1, 5, 200)
ax.plot(xn, stats.norm.pdf(xn, q25_valid.mean(), q25_valid.std()), 'r-', lw=2, label='Normal overlay')
ax.axvline(5, color='orange', ls='--', label=f'Ceiling=5 ({ceiling_q25_5:.1%})')
ax.set_xlabel('Satisfaction (Q25)'); ax.set_ylabel('Density')
ax.set_title('Q25 Distribution with Normal Overlay'); ax.legend()

# Q24 histogram
ax = axes[1]
ax.hist(q24_valid, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color='teal', edgecolor='white')
ax.set_xlabel('Q24 Participation Level'); ax.set_ylabel('Count')
ax.set_title('Q24 Distribution'); ax.set_xticks([0, 1, 2])

# Boxplot by group
ax = axes[2]
groups = [df_full[df_full['q24'] == g]['satisfaction'].values for g in [0, 1, 2]]
ax.boxplot(groups, patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
ax.set_xlabel('Q24 Group'); ax.set_ylabel('Satisfaction')
ax.set_title('Satisfaction by Q24 Group')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'section2_ceiling_floor.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: section2_ceiling_floor.png")

# ============================================================
# SECTION 3: NON-PARAMETRIC TESTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: NON-PARAMETRIC TESTS")
print("=" * 60)

# Shapiro-Wilk
sw_stat, sw_p = stats.shapiro(q25_valid)
print(f"\nShapiro-Wilk on Q25: W={sw_stat:.4f}, p={sw_p:.4f}")
print(f"  -> {'Non-normal' if sw_p < 0.05 else 'Cannot reject normality'}")

# Kruskal-Wallis
groups_kw = [df_full[df_full['q24'] == g]['satisfaction'].dropna().values for g in [0, 1, 2]]
H, p_kw = stats.kruskal(*groups_kw)
n_total = sum(len(g) for g in groups_kw)
eps_sq = (H - len(groups_kw) + 1) / (n_total - len(groups_kw))
print(f"\nKruskal-Wallis: H={H:.3f}, p={p_kw:.4f}, e2={eps_sq:.3f}")

# Pairwise Mann-Whitney with Bonferroni
pairs = [(0, 1), (0, 2), (1, 2)]
n_pairs = len(pairs)
print("\nPairwise Mann-Whitney U (Bonferroni corrected):")
for g1, g2 in pairs:
    a, b = groups_kw[g1], groups_kw[g2]
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    p_adj = min(p * n_pairs, 1.0)
    n1, n2 = len(a), len(b)
    rb = 1 - 2*U / (n1 * n2)
    print(f"  Q24={g1} vs Q24={g2}: U={U:.1f}, p={p:.4f}, p_adj={p_adj:.4f}, r_b={rb:.3f}")

# Correlations
print("\nSpearman and Kendall correlations with Satisfaction:")
for var in ['q24', 'info_score', 'consult_score']:
    tmp = df_full[[var, 'satisfaction']].dropna()
    rho, p_rho = stats.spearmanr(tmp[var], tmp['satisfaction'])
    tau, p_tau = stats.kendalltau(tmp[var], tmp['satisfaction'])
    print(f"  {var}: Spearman rho={rho:.3f} (p={p_rho:.4f}), Kendall tau={tau:.3f} (p={p_tau:.4f})")

# Jonckheere-Terpstra (manual)
def jonckheere_terpstra(groups_list):
    k = len(groups_list)
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            U_ij, _ = stats.mannwhitneyu(groups_list[i], groups_list[j], alternative='less')
            J += U_ij
    n = [len(g) for g in groups_list]
    N = sum(n)
    mu_J = (N**2 - sum(ni**2 for ni in n)) / 4
    var_J = (N**2 * (2*N + 3) - sum(ni**2 * (2*ni + 3) for ni in n)) / 72
    z_J = (J - mu_J) / np.sqrt(var_J)
    p_J = 1 - stats.norm.cdf(z_J)
    return J, z_J, p_J

J_stat, z_J, p_J = jonckheere_terpstra(groups_kw)
print(f"\nJonckheere-Terpstra trend test:")
print(f"  J={J_stat:.1f}, Z={z_J:.3f}, p={p_J:.4f}")
print(f"  -> {'Significant' if p_J < 0.05 else 'Non-significant'} ordered trend")

# ============================================================
# SECTION 4: ORDINAL LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: ORDINAL LOGISTIC REGRESSION")
print("=" * 60)

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d4 = df_full[['q24', 'q22', 'info_score', 'consult_score', 'satisfaction']].dropna().reset_index(drop=True)
    d4['satisfaction_cat'] = d4['satisfaction'].astype(int).astype(str)

    print("\nModel 1: q24 -> satisfaction")
    mod1 = OrderedModel(d4['satisfaction_cat'], d4[['q24']], distr='logit')
    res1 = mod1.fit(method='bfgs', disp=False)
    print(res1.summary())
    pseudo_r2_1 = 1 - res1.llf / res1.llnull
    print(f"  Pseudo-R2: {pseudo_r2_1:.4f}")

    print("\nModel 2: q22 + info_score + consult_score -> satisfaction")
    mod2 = OrderedModel(d4['satisfaction_cat'], d4[['q22', 'info_score', 'consult_score']], distr='logit')
    res2 = mod2.fit(method='bfgs', disp=False)
    print(res2.summary())
    pseudo_r2_2 = 1 - res2.llf / res2.llnull
    print(f"  Pseudo-R2: {pseudo_r2_2:.4f}")

except Exception as e:
    print(f"  Ordinal model error: {e}")
    pseudo_r2_1 = pseudo_r2_2 = np.nan
    res1 = res2 = None

# ============================================================
# SECTION 5: PLS-SEM PATH MODEL
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: PLS-SEM PATH MODEL")
print("=" * 60)

d5 = df_full[['q22', 'info_score', 'consult_score', 'satisfaction']].dropna().reset_index(drop=True)

def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-15)

X_pls = d5[['q22', 'info_score', 'consult_score']].values
Y_pls = d5['satisfaction'].values

X_std = np.column_stack([standardize(X_pls[:, i]) for i in range(X_pls.shape[1])])
Y_std = standardize(Y_pls)

# Mode-A outer weights (power method)
w = np.ones(X_std.shape[1]) / np.sqrt(X_std.shape[1])
for _ in range(100):
    score = X_std @ w
    score = (score - score.mean()) / (score.std() + 1e-15)
    w_new = X_std.T @ score / len(score)
    w_new /= (np.linalg.norm(w_new) + 1e-15)
    if np.allclose(w, w_new, atol=1e-6):
        break
    w = w_new

participation_lv = X_std @ w
participation_lv = (participation_lv - participation_lv.mean()) / (participation_lv.std() + 1e-15)

# Inner model
Xin = np.column_stack([np.ones(len(participation_lv)), participation_lv])
beta_inner = np.linalg.lstsq(Xin, Y_std, rcond=None)[0]
y_pred_inner = Xin @ beta_inner
resid = Y_std - y_pred_inner
ss_res = (resid**2).sum()
ss_tot = ((Y_std - Y_std.mean())**2).sum()
R2_pls = 1 - ss_res / ss_tot
print(f"\nPLS-SEM inner path: beta={beta_inner[1]:.3f}, R2={R2_pls:.3f}")
print(f"Outer weights: q22={w[0]:.3f}, info_score={w[1]:.3f}, consult_score={w[2]:.3f}")

# Bootstrap CI (500 resamples)
np.random.seed(42)
boot_betas = []
n5 = len(d5)
for _ in range(500):
    idx = np.random.choice(n5, n5, replace=True)
    Xb = X_std[idx]
    Yb = Y_std[idx]
    wb = np.ones(Xb.shape[1]) / np.sqrt(Xb.shape[1])
    for __ in range(100):
        sb = Xb @ wb
        sb = (sb - sb.mean()) / (sb.std() + 1e-15)
        wb_new = Xb.T @ sb / len(sb)
        wb_new /= (np.linalg.norm(wb_new) + 1e-15)
        if np.allclose(wb, wb_new, atol=1e-6):
            break
        wb = wb_new
    lv_b = Xb @ wb
    lv_b = (lv_b - lv_b.mean()) / (lv_b.std() + 1e-15)
    Xin_b = np.column_stack([np.ones(len(lv_b)), lv_b])
    Yb_pred = np.linalg.lstsq(Xin_b, Yb, rcond=None)[0]
    boot_betas.append(Yb_pred[1])

boot_betas = np.array(boot_betas)
ci_low, ci_high = np.percentile(boot_betas, [2.5, 97.5])
print(f"Bootstrap 95% CI for path coefficient: [{ci_low:.3f}, {ci_high:.3f}]")

# Q2 (blindfolding, d=7)
d_blind = 7
q2_ss_res, q2_ss_tot = 0, 0
for start in range(d_blind):
    idx_out = list(range(start, n5, d_blind))
    idx_in  = [i for i in range(n5) if i not in idx_out]
    X_in_, Y_in_ = X_std[idx_in], Y_std[idx_in]
    X_out_, Y_out_ = X_std[idx_out], Y_std[idx_out]
    wb = np.ones(X_in_.shape[1]) / np.sqrt(X_in_.shape[1])
    for __ in range(100):
        sb = X_in_ @ wb
        sb = (sb - sb.mean()) / (sb.std() + 1e-15)
        wb_new = X_in_.T @ sb / len(sb)
        wb_new /= (np.linalg.norm(wb_new) + 1e-15)
        if np.allclose(wb, wb_new, atol=1e-6):
            break
        wb = wb_new
    lv_in = X_in_ @ wb
    lv_in = (lv_in - lv_in.mean()) / (lv_in.std() + 1e-15)
    Xin_in = np.column_stack([np.ones(len(lv_in)), lv_in])
    beta_in = np.linalg.lstsq(Xin_in, Y_in_, rcond=None)[0]
    lv_out = X_out_ @ wb
    mean_lv_in = (X_in_ @ wb).mean()
    std_lv_in  = (X_in_ @ wb).std() + 1e-15
    lv_out = (lv_out - mean_lv_in) / std_lv_in
    y_hat_out = beta_in[0] + beta_in[1] * lv_out
    q2_ss_res += ((Y_out_ - y_hat_out)**2).sum()
    q2_ss_tot += ((Y_out_ - Y_std.mean())**2).sum()

Q2_pls = 1 - q2_ss_res / q2_ss_tot
f2_pls = R2_pls / (1 - R2_pls + 1e-15)
print(f"Q2 = {Q2_pls:.3f}, f2 = {f2_pls:.3f}")

# ============================================================
# SECTION 6: EXPECTATION GAP THEORY
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: EXPECTATION GAP THEORY ASSESSMENT")
print("=" * 60)

g0 = df_full[df_full['q24'] == 0]['satisfaction'].dropna().values
g1 = df_full[df_full['q24'] == 1]['satisfaction'].dropna().values
g2 = df_full[df_full['q24'] == 2]['satisfaction'].dropna().values

print(f"\nMean satisfaction by Q24:")
print(f"  Q24=0 (None):    M={g0.mean():.3f}, n={len(g0)}")
print(f"  Q24=1 (Info):    M={g1.mean():.3f}, n={len(g1)}")
print(f"  Q24=2 (Consult): M={g2.mean():.3f}, n={len(g2)}")

if len(g2) > 0 and len(g0) > 0:
    U_20, p_20 = stats.mannwhitneyu(g2, g0, alternative='less')
    print(f"\nOne-tailed MW (Q24=2 < Q24=0): U={U_20:.1f}, p={p_20:.4f}")
if len(g2) > 0 and len(g1) > 0:
    U_21, p_21 = stats.mannwhitneyu(g2, g1, alternative='less')
    print(f"One-tailed MW (Q24=2 < Q24=1): U={U_21:.1f}, p={p_21:.4f}")

# Levene's test
all_groups = [g for g in [g0, g1, g2] if len(g) > 1]
lev_stat, lev_p = stats.levene(*all_groups)
print(f"\nLevene's test: W={lev_stat:.3f}, p={lev_p:.4f}")

# Cohen's d between Q24=0 and Q24=2
if len(g0) > 1 and len(g2) > 1:
    pooled_sd = np.sqrt(((len(g0)-1)*g0.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g0)+len(g2)-2))
    cohens_d = (g0.mean() - g2.mean()) / (pooled_sd + 1e-15)
    print(f"Cohen's d (Q24=0 vs Q24=2): d={cohens_d:.3f}")

# Verdict
gap_confirmed = False
gap_partial   = False
if len(g2) > 0 and len(g0) > 0:
    if g2.mean() < g0.mean():
        gap_partial = True
        if p_20 < 0.05:
            gap_confirmed = True

if gap_confirmed:
    verdict = "CONFIRMED: Higher participation associated with lower satisfaction (expectation gap)"
elif gap_partial:
    verdict = "PARTIALLY CONFIRMED: Direction matches but not statistically significant"
else:
    verdict = "DISCONFIRMED: Higher participation not associated with lower satisfaction"
print(f"\nExpectation Gap Verdict: {verdict}")

# ============================================================
# SECTION 7: XGBOOST + SHAP
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: XGBOOST + SHAP")
print("=" * 60)

try:
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    d7 = df_full[['q22', 'info_score', 'consult_score', 'q24', 'satisfaction']].dropna().reset_index(drop=True)

    scaler = MinMaxScaler()
    d7_norm = pd.DataFrame(scaler.fit_transform(d7[['q22', 'q24', 'info_score', 'consult_score']]),
                           columns=['q22_n', 'q24_n', 'info_n', 'cons_n'])
    d7['participation_composite'] = (0.35 * d7_norm['q22_n'] +
                                     0.20 * d7_norm['q24_n'] +
                                     0.45 * (d7_norm['info_n'] + d7_norm['cons_n']))

    features = ['q22', 'info_score', 'consult_score', 'q24', 'participation_composite']
    X7 = d7[features].values
    y7 = d7['satisfaction'].values

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                   random_state=42, verbosity=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2   = cross_val_score(xgb_model, X7, y7, cv=kf, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(xgb_model, X7, y7, cv=kf, scoring='neg_mean_squared_error'))
    print(f"\nXGBoost 5-fold CV: R2={cv_r2.mean():.3f}+-{cv_r2.std():.3f}, RMSE={cv_rmse.mean():.3f}+-{cv_rmse.std():.3f}")

    xgb_model.fit(X7, y7)

    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X7)

        try:
            fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
            feature_importance = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(feature_importance)
            ax_shap.barh(np.array(features)[sorted_idx], feature_importance[sorted_idx], color='steelblue')
            ax_shap.set_xlabel('Mean |SHAP value|')
            ax_shap.set_title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'section7_shap_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: section7_shap_importance.png")
        except Exception as e:
            print(f"  SHAP beeswarm save error: {e}")

        try:
            q24_idx = features.index('q24')
            fig_dep, ax_dep = plt.subplots(figsize=(7, 5))
            ax_dep.scatter(X7[:, q24_idx], shap_values[:, q24_idx], alpha=0.6, c='teal')
            ax_dep.set_xlabel('q24'); ax_dep.set_ylabel('SHAP value')
            ax_dep.set_title('SHAP Dependence: q24')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'section7_shap_dependence_q24.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: section7_shap_dependence_q24.png")
        except Exception as e:
            print(f"  SHAP dependence save error: {e}")

    except ImportError:
        print("  SHAP not available, skipping SHAP analysis")

except ImportError:
    print("  XGBoost not available, skipping section 7")
    features = ['q22', 'info_score', 'consult_score', 'q24', 'participation_composite']
    d7 = df_full.copy()
    d7['participation_composite'] = np.nan
    xgb_model = None
    cv_r2 = np.array([np.nan])
    cv_rmse = np.array([np.nan])

# ============================================================
# SECTION 8: COMPREHENSIVE RESULTS FIGURE
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: COMPREHENSIVE RESULTS FIGURE")
print("=" * 60)

fig8, axes8 = plt.subplots(3, 3, figsize=(18, 15))
fig8.suptitle('Participation-Satisfaction Study: Comprehensive Results', fontsize=14, fontweight='bold')

# (A) Boxplot satisfaction by Q24 group
ax = axes8[0, 0]
bp = ax.boxplot([g0, g1, g2],
                patch_artist=True, notch=False)
colors = ['#4C72B0', '#DD8452', '#55A868']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Satisfaction (Q25)')
ax.set_title('(A) Satisfaction by Q24 Group')
# Significance brackets
y_max = max(np.concatenate([g0, g1, g2])) if len(g0) and len(g1) and len(g2) else 5
h = 0.2
pairs_sig = [(1, 2, p_kw)]
for x1, x2, pv in [(1, 3, p_kw)]:
    y = y_max + h
    ax.plot([x1, x1, x2, x2], [y, y+h*0.5, y+h*0.5, y], 'k-', lw=1)
    sig_str = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
    ax.text((x1+x2)/2, y+h*0.5, sig_str, ha='center', va='bottom', fontsize=10)

# (B) Bar chart mean+-SE
ax = axes8[0, 1]
means = [g0.mean() if len(g0) else 0, g1.mean() if len(g1) else 0, g2.mean() if len(g2) else 0]
sems  = [g0.std()/np.sqrt(len(g0)) if len(g0) else 0,
         g1.std()/np.sqrt(len(g1)) if len(g1) else 0,
         g2.std()/np.sqrt(len(g2)) if len(g2) else 0]
ax.bar([0, 1, 2], means, yerr=sems, capsize=5, color=colors, edgecolor='black')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Q24=0', 'Q24=1', 'Q24=2'])
ax.set_ylabel('Mean Satisfaction'); ax.set_title('(B) Mean+-SE Satisfaction by Group')
ax.set_ylim(0, 6)

# (C) Scatter Q24 vs satisfaction with jitter
ax = axes8[0, 2]
x_jit = df_full['q24'] + np.random.uniform(-0.1, 0.1, len(df_full))
ax.scatter(x_jit, df_full['satisfaction'], alpha=0.4, color='steelblue', s=30)
z = np.polyfit(df_full['q24'].dropna(), df_full['satisfaction'].dropna(), 1)
p_trend = np.poly1d(z)
xline = np.linspace(-0.1, 2.1, 100)
ax.plot(xline, p_trend(xline), 'r-', lw=2, label=f'Trend (slope={z[0]:.2f})')
ax.set_xlabel('Q24 (with jitter)'); ax.set_ylabel('Satisfaction')
ax.set_title('(C) Q24 vs Satisfaction'); ax.legend()

# (D) PLS-SEM path diagram
ax = axes8[1, 0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(D) PLS-SEM Path Diagram')
# Boxes
def draw_box(ax, x, y, bw, bh, text, color='lightblue'):
    rect = mpatches.FancyBboxPatch((x-bw/2, y-bh/2), bw, bh,
                                    boxstyle='round,pad=0.1', facecolor=color, edgecolor='navy', lw=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

draw_box(ax, 2, 7, 2.5, 0.8, 'q22', 'lightyellow')
draw_box(ax, 2, 5, 2.5, 0.8, 'info_score', 'lightyellow')
draw_box(ax, 2, 3, 2.5, 0.8, 'consult_score', 'lightyellow')
draw_box(ax, 5, 5, 2.5, 1.2, 'PARTICIPATION\n(Latent)', 'lightblue')
draw_box(ax, 8.5, 5, 2.5, 1.2, 'SATISFACTION\n(Latent)', 'lightgreen')
# Arrows
for y_ind, wt in zip([7, 5, 3], w):
    ax.annotate('', xy=(3.75, 5), xytext=(3.25, y_ind),
                arrowprops=dict(arrowstyle='->', color='navy'))
    ax.text(3.5, (5+y_ind)/2, f'w={wt:.2f}', fontsize=7, color='navy')
ax.annotate('', xy=(7.25, 5), xytext=(6.25, 5),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
ax.text(6.75, 5.3, f'beta={beta_inner[1]:.2f}', fontsize=9, color='darkred', fontweight='bold')
ax.text(6.75, 4.6, f'R2={R2_pls:.2f}', fontsize=8, color='gray')

# (E) SHAP summary or placeholder
ax = axes8[1, 1]
try:
    shap_img_path = os.path.join(RESULTS_DIR, 'section7_shap_importance.png')
    if os.path.exists(shap_img_path):
        img = plt.imread(shap_img_path)
        ax.imshow(img); ax.axis('off')
        ax.set_title('(E) SHAP Feature Importance')
    else:
        raise FileNotFoundError
except Exception:
    ax.axis('off')
    ax.set_title('(E) SHAP Summary')
    ax.text(0.5, 0.5, 'XGBoost/SHAP\nnot available\nor figures not saved',
            ha='center', va='center', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

# (F) Tobit vs OLS coefficient comparison
ax = axes8[1, 2]
ax.bar(['OLS beta', 'Tobit beta'], [beta_ols[1], beta_tobit[1]], color=['steelblue', 'darkorange'])
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Coefficient (Q24 -> Q25)')
ax.set_title('(F) Tobit vs OLS Coefficients')
for i, (lbl, val) in enumerate(zip(['OLS', 'Tobit'], [beta_ols[1], beta_tobit[1]])):
    ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# (G) Q25 histogram annotated
ax = axes8[2, 0]
ax.hist(q25_valid, bins=np.arange(0.5, 6.5, 1), color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(5, color='red', ls='--', lw=2, label=f'Ceiling: {ceiling_q25_5:.1%} at 5')
ax.axvline(1, color='orange', ls='--', lw=2, label='Floor: 1')
ax.set_xlabel('Satisfaction'); ax.set_ylabel('Count')
ax.set_title('(G) Q25 Distribution -- Ceiling/Floor')
ax.legend(fontsize=8)

# (H) Expectation gap verdict text
ax = axes8[2, 1]
ax.axis('off')
ax.set_title('(H) Expectation Gap Theory')
lines = [
    f"Mean Sat Q24=0: {g0.mean():.2f}" if len(g0) else "Q24=0: n/a",
    f"Mean Sat Q24=1: {g1.mean():.2f}" if len(g1) else "Q24=1: n/a",
    f"Mean Sat Q24=2: {g2.mean():.2f}" if len(g2) else "Q24=2: n/a",
    "",
    f"KW H={H:.2f}, p={p_kw:.3f}",
    f"Levene p={lev_p:.3f}",
    "",
    "VERDICT:",
    verdict[:40],
    verdict[40:] if len(verdict) > 40 else ""
]
ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# (I) Jonckheere trend visualization
ax = axes8[2, 2]
q24_vals = [0, 1, 2]
mean_sat = [g.mean() if len(g) else np.nan for g in [g0, g1, g2]]
ax.plot(q24_vals, mean_sat, 'o-', color='steelblue', ms=8, lw=2)
ax.fill_between(q24_vals,
                [m - s for m, s in zip(mean_sat, sems)],
                [m + s for m, s in zip(mean_sat, sems)],
                alpha=0.3, color='steelblue')
ax.set_xlabel('Q24 Participation Level'); ax.set_ylabel('Mean Satisfaction')
ax.set_title(f'(I) Jonckheere Trend: Z={z_J:.2f}, p={p_J:.3f}')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['None', 'Info', 'Consult'])
sig_str = '(sig.)' if p_J < 0.05 else '(n.s.)'
ax.text(0.98, 0.05, f'Trend {sig_str}', transform=ax.transAxes, ha='right', fontsize=10,
        color='red' if p_J < 0.05 else 'gray')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: comprehensive_results.png")

# ============================================================
# SECTION 9: SUMMARY AND SAVE
# ============================================================
print("\n" + "=" * 60)
print("SECTION 9: SUMMARY AND SAVE")
print("=" * 60)

summary_rows = [
    {'Test': 'Sample size (full cleaned)',        'Statistic': f'n={len(df)}',           'Detail': ''},
    {'Test': 'Sample size (with satisfaction)',   'Statistic': f'n={len(df_sat)}',        'Detail': ''},
    {'Test': 'Sample size (complete cases)',      'Statistic': f'n={len(df_full)}',       'Detail': ''},
    {'Test': 'Q25 Shapiro-Wilk',                 'Statistic': f'W={sw_stat:.4f}',        'Detail': f'p={sw_p:.4f}'},
    {'Test': 'Kruskal-Wallis',                   'Statistic': f'H={H:.3f}',              'Detail': f'p={p_kw:.4f}, e2={eps_sq:.3f}'},
    {'Test': 'Jonckheere-Terpstra',              'Statistic': f'J={J_stat:.1f}, Z={z_J:.3f}', 'Detail': f'p={p_J:.4f}'},
    {'Test': 'Ceiling Q25 (at 5)',               'Statistic': f'{ceiling_q25_5:.1%}',    'Detail': f'4+5: {ceiling_q25_45:.1%}'},
    {'Test': 'Floor Q24 (at 0)',                 'Statistic': f'{floor_q24_0:.1%}',      'Detail': f'0+1: {floor_q24_01:.1%}'},
    {'Test': 'OLS beta (Q24->Q25)',              'Statistic': f'{beta_ols[1]:.3f}',      'Detail': ''},
    {'Test': 'Tobit beta (Q24->Q25)',            'Statistic': f'{beta_tobit[1]:.3f}',    'Detail': ''},
    {'Test': 'PLS path beta (PART->SAT)',        'Statistic': f'{beta_inner[1]:.3f}',    'Detail': f'R2={R2_pls:.3f}, Q2={Q2_pls:.3f}'},
    {'Test': 'Expectation Gap Verdict',          'Statistic': verdict[:60],              'Detail': ''},
    {'Test': 'XGBoost CV R2',                   'Statistic': f'{cv_r2.mean():.3f}',     'Detail': f'RMSE={cv_rmse.mean():.3f}'},
]

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(RESULTS_DIR, 'results_full_summary.csv'), index=False)
desc_df.to_csv(os.path.join(RESULTS_DIR, 'descriptive_statistics.csv'), index=False)

print(f"\nAll results saved to: {RESULTS_DIR}")
print("\nAnalysis complete.")
