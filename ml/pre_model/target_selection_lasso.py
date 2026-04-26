# -*- coding: utf-8 -*-
"""
================================================================
  TARGET SELECTION VIA LASSO SCREENING
  ---------------------------------------------------------------
  Idea:
    Treat every continuous column as a potential target y.
    For each candidate y:
      1. Fit LassoCV (10-fold) to predict y from all other cols
      2. Record best alpha, CV-R2, RMSE, selected features
      3. Show regularization path
      4. Rank candidates by predictability (R2)

  Why Lasso for target selection?
    - L1 penalty forces irrelevant features to exactly 0
    - The number of selected features reflects how "structured"
      the relationship is
    - A high R2 with few selected features = clean, learnable y
    - A high R2 with many features = complex but predictable y
    - Low R2 everywhere = y is essentially noise / unexplained
================================================================
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.linear_model  import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# ── 1. LOAD & CLEAN DATA ──────────────────────────────────
def load_clean():
    df_raw = pd.read_csv(os.path.join(SAVE_DIR, "encoded_data.csv"), low_memory=False)
    def enc(v):
        m = {"Low": 2, "Medium": 5, "High": 8}
        if str(v) in m: return m[str(v)]
        try: return float(v)
        except: return np.nan
    df_raw["stress_level"] = df_raw["stress_level"].apply(enc)
    df = df_raw.dropna().copy()
    df = df[df["gender"].isin([0, 1])].copy()
    df.reset_index(drop=True, inplace=True)
    return df

df = load_clean()
print(f"Dataset shape: {df.shape}")

# ── 2. CANDIDATE TARGET COLUMNS ───────────────────────────
# We consider every column that is:
#   - numeric (float or int)
#   - not a binary dummy (std > 0.3 to avoid pure 0/1 flags)
#   - has enough unique values (>10)
#   - not near-constant

candidates = []
for col in df.columns:
    s = df[col]
    if s.dtype not in [float, "float64", "int64"]: continue
    if s.std() < 0.3:    continue        # near-binary / constant
    if s.nunique() < 10: continue        # too few levels
    candidates.append(col)

# Also force-add binary targets of interest for comparison
extra_binary = ["mental_health_status", "support_systems_access",
                "online_support_usage", "gender"]
binary_targets = [c for c in extra_binary if c in df.columns]

print(f"\nContinuous candidate targets ({len(candidates)}):")
for c in candidates: print(f"  - {c}")
print(f"\nBinary targets also evaluated ({len(binary_targets)}):")
for c in binary_targets: print(f"  - {c}")

all_targets = candidates + [c for c in binary_targets if c not in candidates]

# ── 3. LASSO SCREENING LOOP ───────────────────────────────
print("\n" + "="*65)
print("  LASSO SCREENING — one model per candidate target")
print("="*65)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
rows = []

# storage for heatmap
coef_matrix = {}

for target_col in all_targets:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # LassoCV
    lcv = LassoCV(cv=kf, max_iter=30000, random_state=42, n_alphas=80)
    lcv.fit(X_s, y_s)

    best_alpha = lcv.alpha_
    coefs      = lcv.coef_

    # CV R2 from stored mse path
    # sklearn LassoCV stores mse_path_ shape (n_alphas, n_folds)
    alpha_idx  = list(lcv.alphas_).index(best_alpha) if best_alpha in lcv.alphas_ else 0
    cv_mse     = lcv.mse_path_[alpha_idx].mean()
    y_var      = y_s.var()
    cv_r2      = max(0, 1 - cv_mse / y_var)

    n_selected = (coefs != 0).sum()
    selected   = [feature_cols[i] for i, c in enumerate(coefs) if c != 0]

    # Holdout RMSE for reference
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    lasso_fit = Lasso(alpha=best_alpha, max_iter=30000)
    lasso_fit.fit(X_tr, y_tr)
    y_hat = lasso_fit.predict(X_te)
    rmse_scaled = np.sqrt(mean_squared_error(y_te, y_hat))
    r2_holdout  = r2_score(y_te, y_hat)

    sk_val = skew(y)
    kt_val = kurtosis(y, fisher=True)

    rows.append(dict(
        target       = target_col,
        best_alpha   = round(best_alpha, 6),
        cv_R2        = round(cv_r2, 4),
        holdout_R2   = round(r2_holdout, 4),
        holdout_RMSE = round(rmse_scaled, 4),
        n_selected   = int(n_selected),
        skewness     = round(sk_val, 3),
        kurtosis_exc = round(kt_val, 3),
        top3_features= ", ".join(
            sorted(selected,
                   key=lambda f: abs(coefs[feature_cols.index(f)]),
                   reverse=True)[:3]
        ) if selected else "none"
    ))

    coef_matrix[target_col] = {feature_cols[i]: coefs[i]
                                for i in range(len(feature_cols))}

    print(f"\n  [{target_col}]")
    print(f"    best alpha   = {best_alpha:.6f}")
    print(f"    CV-R2        = {cv_r2:.4f}")
    print(f"    Holdout R2   = {r2_holdout:.4f}  RMSE={rmse_scaled:.4f}")
    print(f"    # features selected = {n_selected} / {len(feature_cols)}")
    if selected:
        top5 = sorted(selected,
                      key=lambda f: abs(coefs[feature_cols.index(f)]),
                      reverse=True)[:5]
        print(f"    Top features: {top5}")

# ── 4. RANKING TABLE ──────────────────────────────────────
print("\n" + "="*65)
print("  RANKING — Best Targets by Holdout R2")
print("="*65)

result_df = pd.DataFrame(rows).sort_values("holdout_R2", ascending=False)
print(result_df.to_string(index=False))
result_df.to_csv(os.path.join(SAVE_DIR, "target_screening_results.csv"), index=False)
print("\n  >> target_screening_results.csv saved")

# ── 5. VISUALIZATION A: R2 Ranking Bar Chart ──────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Lasso Target Screening — Predictability Ranking", fontsize=14, fontweight="bold")

colors = ["#2ecc71" if r >= 0.5 else "#e67e22" if r >= 0.25 else "#e74c3c"
          for r in result_df["holdout_R2"]]

axes[0].barh(result_df["target"], result_df["holdout_R2"], color=colors, edgecolor="white")
axes[0].axvline(0.5,  color="green",  linestyle="--", linewidth=1.5, label="R2=0.5 (good)")
axes[0].axvline(0.25, color="orange", linestyle="--", linewidth=1.5, label="R2=0.25 (moderate)")
axes[0].set_xlabel("Holdout R2"); axes[0].set_title("Predictability (R2) per Target")
axes[0].legend(fontsize=9); axes[0].invert_yaxis()

axes[1].barh(result_df["target"], result_df["n_selected"],
             color="#4C72B0", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("# Features Selected by Lasso")
axes[1].set_title("Feature Sparsity (fewer = cleaner signal)")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figTS1_target_ranking.png"))
plt.close()
print("  >> figTS1_target_ranking.png saved")

# ── 6. VISUALIZATION B: Lasso Coefficient Heatmap ────────
# Build full coef DataFrame (targets x features)
all_features_union = sorted(set(
    feat for d in coef_matrix.values() for feat in d
))
heatmap_data = pd.DataFrame(index=all_targets, columns=all_features_union, dtype=float)
for tgt, feat_dict in coef_matrix.items():
    for feat, val in feat_dict.items():
        heatmap_data.loc[tgt, feat] = val
heatmap_data = heatmap_data.fillna(0)

# Sort rows by holdout R2
order = result_df["target"].tolist()
heatmap_data = heatmap_data.loc[order]

fig, ax = plt.subplots(figsize=(max(18, len(all_features_union)*0.9),
                                max(8,  len(all_targets)*0.7)))
sns.heatmap(heatmap_data.astype(float), cmap="coolwarm", center=0,
            linewidths=0.3, annot=False, ax=ax,
            cbar_kws={"label": "Lasso coefficient (standardized)"})
ax.set_title("Lasso Coefficient Heatmap — Target x Feature\n"
             "(rows = targets ranked by R2, cols = features; "
             "white = zeroed by Lasso)", fontsize=12, fontweight="bold")
ax.set_xlabel("Feature"); ax.set_ylabel("Target")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figTS2_lasso_coef_heatmap.png"))
plt.close()
print("  >> figTS2_lasso_coef_heatmap.png saved")

# ── 7. VISUALIZATION C: Lasso Regularization Paths ───────
# Show path for the TOP 4 targets by R2
top4_targets = result_df.head(4)["target"].tolist()
alphas_path  = np.logspace(-4, 2, 100)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Lasso Regularization Paths — Top-4 Targets\n"
             "(each line = one feature; path shows how L1 shrinks coefficients)",
             fontsize=13, fontweight="bold")

for ax, target_col in zip(axes.flatten(), top4_targets):
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    X_s = StandardScaler().fit_transform(X)
    y_s = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()

    coefs_path = []
    for a in alphas_path:
        l = Lasso(alpha=a, max_iter=30000).fit(X_s, y_s)
        coefs_path.append(l.coef_.copy())
    coefs_path = np.array(coefs_path)

    # best alpha from our results
    best_a = result_df[result_df["target"] == target_col]["best_alpha"].values[0]
    r2_val = result_df[result_df["target"] == target_col]["holdout_R2"].values[0]

    for i in range(coefs_path.shape[1]):
        ax.plot(np.log10(alphas_path), coefs_path[:, i], linewidth=0.8, alpha=0.75)

    ax.axvline(np.log10(best_a), color="black", linestyle="--", linewidth=2,
               label=f"Best alpha={best_a:.4f}")
    ax.set_title(f"Target: {target_col}  (R2={r2_val:.3f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("log10(alpha)"); ax.set_ylabel("Coefficient")
    ax.legend(fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figTS3_lasso_paths_top4.png"))
plt.close()
print("  >> figTS3_lasso_paths_top4.png saved")

# ── 8. VISUALIZATION D: Skewness vs R2 scatter ───────────
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(result_df["skewness"].abs(), result_df["holdout_R2"],
                c=result_df["n_selected"], cmap="viridis", s=120, edgecolors="k", linewidths=0.7)
for _, row in result_df.iterrows():
    ax.annotate(row["target"], (abs(row["skewness"]), row["holdout_R2"]),
                fontsize=7.5, ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("# Features selected by Lasso")
ax.set_xlabel("|Skewness| of target distribution")
ax.set_ylabel("Lasso Holdout R2")
ax.set_title("Target Quality Map: Skewness vs Predictability\n"
             "(ideal target: low skewness, high R2, few features selected)",
             fontsize=12, fontweight="bold")
ax.axhline(0.5, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label="R2=0.5")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figTS4_target_quality_map.png"))
plt.close()
print("  >> figTS4_target_quality_map.png saved")

# ── 9. RECOMMENDATION REPORT ─────────────────────────────
print("\n" + "="*65)
print("  FINAL TARGET RECOMMENDATIONS")
print("="*65)

excellent = result_df[result_df["holdout_R2"] >= 0.5]
good      = result_df[(result_df["holdout_R2"] >= 0.25) & (result_df["holdout_R2"] < 0.5)]
poor      = result_df[result_df["holdout_R2"] < 0.25]

print(f"""
  Classification based on Lasso holdout R2:

  [EXCELLENT targets  R2 >= 0.50]  -- strongly recommended
  {excellent[['target','holdout_R2','n_selected','top3_features']].to_string(index=False)}

  [GOOD targets  0.25 <= R2 < 0.50]  -- worth studying
  {good[['target','holdout_R2','n_selected','top3_features']].to_string(index=False)}

  [POOR targets  R2 < 0.25]  -- hard to predict linearly
  {poor[['target','holdout_R2','n_selected','top3_features']].to_string(index=False)}

  Interpretation:
  - Lasso R2 measures how much variance is linearly
    explainable from the other features.
  - n_selected = how many features survive L1 shrinkage.
    Fewer selected features = cleaner, more interpretable model.
  - Highly skewed targets (skewness > 2) may benefit from
    a log1p or Box-Cox transform before fitting.
""")

print("  [TARGET SELECTION LASSO STUDY COMPLETE]")
print("  Output files in ml/pre_model/:")
for f in sorted(os.listdir(SAVE_DIR)):
    if f.startswith("figTS") or f == "target_screening_results.csv":
        print(f"    - {f}")
