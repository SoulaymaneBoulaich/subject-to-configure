# -*- coding: utf-8 -*-
"""
============================================================
 PRE-ML STUDY -- PART 3
 Model Fitting, Cross-Validation, Regularization,
 Residual Diagnostics, Best-Fit Line Analysis
============================================================
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

SAVE_DIR = os.path.dirname(__file__)
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# ── Load arrays saved by Part 2 ───────────────────────────
X_train_s = np.load(os.path.join(SAVE_DIR, "X_train_s.npy"))
X_val_s   = np.load(os.path.join(SAVE_DIR, "X_val_s.npy"))
X_test_s  = np.load(os.path.join(SAVE_DIR, "X_test_s.npy"))
y_train   = np.load(os.path.join(SAVE_DIR, "y_train.npy"))
y_val     = np.load(os.path.join(SAVE_DIR, "y_val.npy"))
y_test    = np.load(os.path.join(SAVE_DIR, "y_test.npy"))
FINAL_FEATURES = pd.read_csv(os.path.join(SAVE_DIR, "final_features.csv"),
                              header=None)[0].tolist()

from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler

TARGET = "mental_wellness_index"

def metrics(y_true, y_pred, label=""):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    n, p = len(y_true), X_train_s.shape[1]
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return dict(label=label, MSE=round(mse,4), RMSE=round(rmse,4),
                MAE=round(mae,4), R2=round(r2,4), Adj_R2=round(adj_r2,4))

# ─── 12. OLS -- CLOSED-FORM SOLUTION ──────────────────────
print("="*65)
print("  STEP 12 -- OLS LINEAR REGRESSION (Closed-Form & sklearn)")
print("="*65)

print("""
  Mathematical derivation of OLS:
  ─────────────────────────────────────────────────────────
  Model  :  ŷ = Xbeta + eps
  Loss   :  L(beta) = ‖y - Xbeta‖^2  =  (y - Xbeta)ᵀ(y - Xbeta)

  ∂L/∂beta = -2Xᵀ(y - Xbeta) = 0
  ⟹  XᵀXbeta = Xᵀy            (Normal Equations)
  ⟹  betâ = (XᵀX)⁻^1 Xᵀy      (if XᵀX is invertible)

  Assumptions (Gauss-Markov):
    ① E[eps] = 0  (zero-mean errors)
    ② Cov(eps) = sigma^2I  (homoscedasticity, no autocorrelation)
    ③ rank(X) = p  (no perfect multicollinearity)
    -> betâ_OLS is BLUE (Best Linear Unbiased Estimator)
  ─────────────────────────────────────────────────────────
""")

ols = LinearRegression()
ols.fit(X_train_s, y_train)

y_pred_train = ols.predict(X_train_s)
y_pred_val   = ols.predict(X_val_s)
y_pred_test  = ols.predict(X_test_s)

results = []
results.append(metrics(y_train, y_pred_train, "OLS-Train"))
results.append(metrics(y_val,   y_pred_val,   "OLS-Val"))
results.append(metrics(y_test,  y_pred_test,  "OLS-Test"))

print("  OLS Results:")
for r in results[-3:]:
    print(f"    {r['label']:15s} RMSE={r['RMSE']:.4f}  R^2={r['R2']:.4f}  Adj-R^2={r['Adj_R2']:.4f}")

# Coefficient table
coef_df = pd.DataFrame({"feature": FINAL_FEATURES,
                         "coefficient": ols.coef_}).sort_values("coefficient", key=abs, ascending=False)
print(f"\n  Intercept : {ols.intercept_:.4f}")
print("  Top-10 coefficients:")
print(coef_df.head(10).to_string(index=False))
coef_df.to_csv(os.path.join(SAVE_DIR, "ols_coefficients.csv"), index=False)

# ─── 13. K-FOLD CROSS-VALIDATION (k=10) ──────────────────
print("\n" + "="*65)
print("  STEP 13 -- 10-FOLD CROSS-VALIDATION")
print("="*65)

print("""
  Cross-Validation Rationale:
  ─────────────────────────────────────────────────────────
  Single train/val split is high-variance -- CV gives a more
  reliable estimate of generalisation error.

  k-Fold CV:
    For k = 1..K:
      Fit model on D \\ Dₖ
      Evaluate on Dₖ
  CV score = (1/K) Σ metric(Dₖ)

  We use K=10 (standard); stratified not needed (regression).
  ─────────────────────────────────────────────────────────
""")

X_full = np.vstack([X_train_s, X_val_s])
y_full = np.concatenate([y_train, y_val])

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_models = {
    "OLS"       : LinearRegression(),
    "Ridge(alpha=1)": Ridge(alpha=1.0),
    "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
}

cv_results = {}
for name, model in cv_models.items():
    scores_r2   = cross_val_score(model, X_full, y_full, cv=kf, scoring="r2")
    scores_rmse = np.sqrt(-cross_val_score(model, X_full, y_full, cv=kf,
                                           scoring="neg_mean_squared_error"))
    cv_results[name] = {"R2_mean": scores_r2.mean(), "R2_std": scores_r2.std(),
                        "RMSE_mean": scores_rmse.mean(), "RMSE_std": scores_rmse.std()}
    print(f"  {name:20s}  R^2={scores_r2.mean():.4f}±{scores_r2.std():.4f}"
          f"  RMSE={scores_rmse.mean():.4f}±{scores_rmse.std():.4f}")

# Box-plot of CV R^2 distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("10-Fold Cross-Validation Results", fontsize=13, fontweight="bold")

cv_r2_all, cv_rmse_all, cv_labels = [], [], []
for name, model in cv_models.items():
    s_r2  = cross_val_score(model, X_full, y_full, cv=kf, scoring="r2")
    s_rmse= np.sqrt(-cross_val_score(model, X_full, y_full, cv=kf,
                                     scoring="neg_mean_squared_error"))
    cv_r2_all.append(s_r2); cv_rmse_all.append(s_rmse); cv_labels.append(name)

axes[0].boxplot(cv_r2_all,  labels=cv_labels, patch_artist=True); axes[0].set_title("CV R^2")
axes[1].boxplot(cv_rmse_all, labels=cv_labels, patch_artist=True); axes[1].set_title("CV RMSE")
for ax in axes: ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig06_cv_boxplots.png")); plt.close()
print("  >> fig06_cv_boxplots.png saved")

# ─── 14. RIDGE REGRESSION -- L2 REGULARISATION ────────────
print("\n" + "="*65)
print("  STEP 14 -- RIDGE REGRESSION  (L2 penalty)")
print("="*65)

print("""
  Ridge objective:
  ─────────────────────────────────────────────────────────
  L_ridge(beta) = ‖y - Xbeta‖^2 + alpha‖beta‖^2

  Closed-form:  betâ_ridge = (XᵀX + alphaI)⁻^1 Xᵀy

  Effect:
    - alphaI adds to the diagonal of XᵀX -> always invertible
    - Shrinks beta toward 0 but never exactly 0
    - Bias↑ Variance↓ trade-off
    - Optimal alpha found via RidgeCV (GCV / leave-one-out)
  ─────────────────────────────────────────────────────────
""")

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

alphas_ridge = np.logspace(-3, 4, 100)
ridge_cv = RidgeCV(alphas=alphas_ridge, cv=10, scoring="r2")
ridge_cv.fit(X_full, y_full)
print(f"  Best Ridge alpha : {ridge_cv.alpha_:.5f}")

ridge_best = Ridge(alpha=ridge_cv.alpha_)
ridge_best.fit(X_train_s, y_train)
results.append(metrics(y_test, ridge_best.predict(X_test_s), f"Ridge(alpha={ridge_cv.alpha_:.4f})-Test"))

# ─── 15. LASSO REGRESSION -- L1 REGULARISATION ────────────
print("\n" + "="*65)
print("  STEP 15 -- LASSO REGRESSION  (L1 penalty)")
print("="*65)

print("""
  Lasso objective:
  ─────────────────────────────────────────────────────────
  L_lasso(beta) = ‖y - Xbeta‖^2 + alpha‖beta‖₁

  No closed-form -> solved via coordinate descent or LARS.

  Effect:
    - Can set some betaⱼ = 0 exactly -> built-in feature selection
    - Geometry: L1 ball has corners -> solution hits axes
    - Optimal alpha: LassoCV with k-fold
  ─────────────────────────────────────────────────────────
""")

lasso_cv = LassoCV(cv=10, max_iter=20000, random_state=42)
lasso_cv.fit(X_full, y_full)
print(f"  Best Lasso alpha : {lasso_cv.alpha_:.6f}")

lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=20000)
lasso_best.fit(X_train_s, y_train)
n_zero = (lasso_best.coef_ == 0).sum()
print(f"  Features zeroed out by Lasso : {n_zero} / {len(FINAL_FEATURES)}")
results.append(metrics(y_test, lasso_best.predict(X_test_s), f"Lasso(alpha={lasso_cv.alpha_:.5f})-Test"))

# ─── 16. ELASTIC NET -- L1+L2 ─────────────────────────────
print("\n" + "="*65)
print("  STEP 16 -- ELASTIC NET  (L1 + L2 penalty)")
print("="*65)

print("""
  ElasticNet objective:
  ─────────────────────────────────────────────────────────
  L_EN = ‖y - Xbeta‖^2 + alpha·rho·‖beta‖₁ + alpha·(1-rho)/2·‖beta‖^2

  Parameters:
    alpha   -> overall regularisation strength
    rho (l1_ratio) -> balance between L1 and L2
    rho=1 -> Lasso ;  rho=0 -> Ridge ;  0<rho<1 -> ElasticNet

  Best of both worlds: sparsity + grouping effect
  ─────────────────────────────────────────────────────────
""")

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
en_cv = ElasticNetCV(l1_ratio=l1_ratios, cv=10, max_iter=20000, random_state=42)
en_cv.fit(X_full, y_full)
print(f"  Best EN alpha={en_cv.alpha_:.6f}  l1_ratio={en_cv.l1_ratio_:.2f}")

en_best = ElasticNet(alpha=en_cv.alpha_, l1_ratio=en_cv.l1_ratio_, max_iter=20000)
en_best.fit(X_train_s, y_train)
results.append(metrics(y_test, en_best.predict(X_test_s),
                        f"ElasticNet(alpha={en_cv.alpha_:.5f},rho={en_cv.l1_ratio_})-Test"))

# ─── 17. REGULARIZATION PATH ─────────────────────────────
print("\n" + "="*65)
print("  STEP 17 -- REGULARIZATION PATH (Ridge & Lasso)")
print("="*65)

alphas_path = np.logspace(-3, 3, 120)
ridge_coefs, lasso_coefs = [], []
for a in alphas_path:
    r = Ridge(alpha=a).fit(X_train_s, y_train)
    ridge_coefs.append(r.coef_)
    l = Lasso(alpha=a, max_iter=20000).fit(X_train_s, y_train)
    lasso_coefs.append(l.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Regularization Paths", fontsize=14, fontweight="bold")

for i in range(ridge_coefs.shape[1]):
    axes[0].plot(np.log10(alphas_path), ridge_coefs[:, i], linewidth=0.9)
axes[0].axvline(np.log10(ridge_cv.alpha_), color="k", linestyle="--", linewidth=1.5,
                label=f"Best alpha={ridge_cv.alpha_:.3f}")
axes[0].set_title("Ridge -- L2 Shrinkage Path")
axes[0].set_xlabel("log₁0(alpha)"); axes[0].set_ylabel("Coefficient value")
axes[0].legend()

for i in range(lasso_coefs.shape[1]):
    axes[1].plot(np.log10(alphas_path), lasso_coefs[:, i], linewidth=0.9)
axes[1].axvline(np.log10(lasso_cv.alpha_), color="k", linestyle="--", linewidth=1.5,
                label=f"Best alpha={lasso_cv.alpha_:.4f}")
axes[1].set_title("Lasso -- L1 Sparsity Path")
axes[1].set_xlabel("log₁0(alpha)"); axes[1].set_ylabel("Coefficient value")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig07_regularization_paths.png")); plt.close()
print("  >> fig07_regularization_paths.png saved")

# ─── 18. BEST FITTING LINE ANALYSIS ──────────────────────
print("\n" + "="*65)
print("  STEP 18 -- BEST FITTING LINE (OLS vs Ridge vs Lasso)")
print("="*65)

y_preds = {
    "OLS"  : ols.predict(X_test_s),
    "Ridge": ridge_best.predict(X_test_s),
    "Lasso": lasso_best.predict(X_test_s),
    "EN"   : en_best.predict(X_test_s),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Best-Fit Line: Predicted vs Actual", fontsize=14, fontweight="bold")
colors = {"OLS": "#2ecc71", "Ridge": "#3498db", "Lasso": "#e74c3c", "EN": "#9b59b6"}

for ax, (name, yp) in zip(axes.flatten(), y_preds.items()):
    ax.scatter(y_test, yp, alpha=0.25, s=12, color=colors[name])
    lo, hi = min(y_test.min(), yp.min()), max(y_test.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect fit")
    r2 = r2_score(y_test, yp)
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    ax.set_title(f"{name}  |  R^2={r2:.4f}  RMSE={rmse:.4f}", fontsize=10)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig08_best_fit_line.png")); plt.close()
print("  >> fig08_best_fit_line.png saved")

# ─── 19. RESIDUAL DIAGNOSTICS ─────────────────────────────
print("\n" + "="*65)
print("  STEP 19 -- RESIDUAL DIAGNOSTICS (OLS)")
print("="*65)

resid = y_test - y_preds["OLS"]
print("""
  Residual diagnostics verify Gauss-Markov assumptions:
  ① E[eps]=0         -> mean of residuals ≈ 0
  ② Homoscedasticity -> residuals vs ŷ show no pattern
  ③ Normality      -> Q-Q plot / Shapiro-Wilk test
  ④ No autocorrelation -> Durbin-Watson statistic
""")

print(f"  Residual mean : {resid.mean():.6f}  (should ≈ 0)")
print(f"  Residual std  : {resid.std():.4f}")

# Breusch-Pagan test proxy via correlation of |resid| with ŷ
bp_corr, bp_p = stats.pearsonr(np.abs(resid), y_preds["OLS"])
print(f"  Breusch-Pagan proxy  corr(|eps|, ŷ)={bp_corr:.4f}  p={bp_p:.4e}")
print(f"  -> {'Homoscedastic OK' if bp_p > 0.05 else 'Heteroscedastic FAIL -- consider WLS'}")

sw_stat, sw_p = stats.shapiro(resid[:5000])
print(f"  Shapiro-Wilk on residuals  p={sw_p:.4e}")
print(f"  -> {'Normal residuals OK' if sw_p > 0.05 else 'Non-normal residuals FAIL'}")

# Durbin-Watson
dw = np.sum(np.diff(resid)**2) / np.sum(resid**2)
print(f"  Durbin-Watson statistic : {dw:.4f}  (ideal ≈ 2)")

fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)
fig.suptitle("OLS Residual Diagnostics", fontsize=14, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_preds["OLS"], resid, alpha=0.2, s=8, color="#4C72B0")
ax1.axhline(0, color="red", linewidth=1.5)
ax1.set_xlabel("Fitted ŷ"); ax1.set_ylabel("Residual eps")
ax1.set_title("Residuals vs Fitted")

ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(resid, dist="norm", plot=ax2)
ax2.set_title("Normal Q-Q Plot")

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(resid, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
ax3.set_title("Residual Histogram"); ax3.set_xlabel("eps")

ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_preds["OLS"], np.sqrt(np.abs(resid)), alpha=0.2, s=8, color="#DD8452")
ax4.set_xlabel("Fitted ŷ"); ax4.set_ylabel("√|eps|")
ax4.set_title("Scale-Location (Homoscedasticity)")

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(resid[:300], color="#4C72B0", linewidth=0.7)
ax5.axhline(0, color="red", linewidth=1); ax5.set_title("Residual Series (first 300)")

ax6 = fig.add_subplot(gs[1, 2])
from statsmodels.graphics.tsaplots import plot_acf
try:
    plot_acf(resid, lags=30, ax=ax6, zero=False)
    ax6.set_title("ACF of Residuals")
except:
    ax6.bar(range(1, 31), [np.corrcoef(resid[:-i], resid[i:])[0,1] for i in range(1, 31)])
    ax6.set_title("ACF (manual)")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig09_residual_diagnostics.png")); plt.close()
print("  >> fig09_residual_diagnostics.png saved")

# ─── 20. FINAL MODEL COMPARISON TABLE ────────────────────
print("\n" + "="*65)
print("  STEP 20 -- FINAL MODEL COMPARISON")
print("="*65)

final_df = pd.DataFrame(results)
print(final_df.to_string(index=False))
final_df.to_csv(os.path.join(SAVE_DIR, "final_model_comparison.csv"), index=False)
print("\n  >> final_model_comparison.csv saved")

# Pick winner
best_row = final_df[final_df["label"].str.contains("Test")].sort_values("R2", ascending=False).iloc[0]
print(f"\n  * BEST MODEL on Test set: {best_row['label']}")
print(f"    RMSE = {best_row['RMSE']}  R^2 = {best_row['R2']}")

print("\n" + "="*65)
print("  PRE-ML STUDY COMPLETE")
print("="*65)
print("""
  Summary of all steps performed:
   1.  Data loading & cleaning (stress_level normalisation)
   2.  Target selection with mathematical justification
   3.  Pearson + Spearman correlation with target
   4.  Normality tests (Shapiro-Wilk, KS, D'Agostino)
   5.  Feature distributions, skewness & kurtosis
   6.  IQR outlier detection
   7.  Full Pearson correlation heatmap
   8.  VIF -- multicollinearity quantification
   9.  Scatter plots (top features vs target)
  10.  Feature engineering (interaction & ratio terms)
  11.  Final feature set selection
  12.  Stratified 70/15/15 train-val-test split
  13.  StandardScaler (fit on train, transform others)
  14.  OLS closed-form derivation & fit
  15.  10-Fold Cross-Validation across 4 models
  16.  Ridge regression (L2) with RidgeCV alpha selection
  17.  Lasso regression (L1) with LassoCV alpha selection
  18.  ElasticNet (L1+L2) with ElasticNetCV selection
  19.  Regularisation paths (bias-variance trade-off)
  20.  Best-fit line analysis (predicted vs actual)
  21.  Residual diagnostics (BP test, DW, ACF, QQ)
  22.  Final model comparison table
""")
