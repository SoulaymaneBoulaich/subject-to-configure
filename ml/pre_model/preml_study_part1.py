# -*- coding: utf-8 -*-
"""
============================================================
 PRE-ML STUDY -- PART 1
 Dataset : encoded_data.csv
 Goal    : Full exploratory & statistical groundwork
           before fitting any regression model
============================================================
"""

# ─── 0. IMPORTS ──────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest, skew, kurtosis, pearsonr, spearmanr

SAVE_DIR = os.path.dirname(__file__)
os.makedirs(SAVE_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# ─── 1. LOAD & BASIC CLEANING ────────────────────────────
print("="*65)
print("  STEP 1 -- LOADING & INITIAL CLEANING")
print("="*65)

df_raw = pd.read_csv(os.path.join(SAVE_DIR, "encoded_data.csv"), low_memory=False)
print(f"\n>> Raw shape : {df_raw.shape}")

# --- 1a. Fix stress_level (mixed textual + numeric) ------
def encode_stress(val):
    mapping = {"Low": 2, "Medium": 5, "High": 8}
    if str(val) in mapping:
        return mapping[str(val)]
    try:
        return float(val)
    except:
        return np.nan

df_raw["stress_level"] = df_raw["stress_level"].apply(encode_stress)

# --- 1b. Drop rows with NaN after fix --------------------
df = df_raw.dropna().copy()
df = df.astype({c: float for c in df.select_dtypes("object").columns})

# --- 1c. Remove impossible gender codes ------------------
df = df[df["gender"].isin([0, 1])].copy()   # keep binary only
df.reset_index(drop=True, inplace=True)

print(f">> Clean shape : {df.shape}")
print(f">> Rows dropped: {df_raw.shape[0] - df.shape[0]}")

# ─── 2. TARGET SELECTION -- MATHEMATICAL JUSTIFICATION ────
print("\n" + "="*65)
print("  STEP 2 -- TARGET SELECTION & CORRELATION ANALYSIS")
print("="*65)

TARGET = "mental_wellness_index"
FEATURES = [c for c in df.columns if c != TARGET]

print(f"\n>> Chosen target  : {TARGET}")
print(f">> Reason         : Continuous (0-97), no ceiling effect,")
print(f"                   represents overall psychological state -- ideal for OLS.")

# Pearson + Spearman correlations with target
corr_rows = []
for col in FEATURES:
    pear_r, pear_p = pearsonr(df[col], df[TARGET])
    spea_r, spea_p = spearmanr(df[col], df[TARGET])
    corr_rows.append(dict(feature=col,
                          pearson_r=round(pear_r,4), pearson_p=round(pear_p,6),
                          spearman_r=round(spea_r,4), spearman_p=round(spea_p,6)))

corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r", key=abs, ascending=False)
print("\n>> Correlation with target (sorted by |Pearson r|):")
print(corr_df.to_string(index=False))

# Save correlation table
corr_df.to_csv(os.path.join(SAVE_DIR, "correlation_table.csv"), index=False)

# ─── 3. DISTRIBUTION OF THE TARGET ───────────────────────
print("\n" + "="*65)
print("  STEP 3 -- TARGET DISTRIBUTION & NORMALITY TESTS")
print("="*65)

y = df[TARGET].values
sk = skew(y);  ku = kurtosis(y, fisher=True)
stat_sw, p_sw = shapiro(y[:5000])   # Shapiro on sample (max 5000)
stat_ks, p_ks = kstest(y, "norm", args=(y.mean(), y.std()))
stat_da, p_da = normaltest(y)

print(f"\n  Mean           : {y.mean():.4f}")
print(f"  Std            : {y.std():.4f}")
print(f"  Skewness       : {sk:.4f}  {'(approx symmetric)' if abs(sk)<0.5 else '(skewed)'}")
print(f"  Excess Kurtosis: {ku:.4f}  {'(mesokurtic)' if abs(ku)<1 else '(non-normal tails)'}")
print(f"\n  Shapiro-Wilk   W={stat_sw:.4f}  p={p_sw:.4e}  -> {'NORMAL OK' if p_sw>0.05 else 'NON-NORMAL FAIL'}")
print(f"  KS test        D={stat_ks:.4f}  p={p_ks:.4e}  -> {'NORMAL OK' if p_ks>0.05 else 'NON-NORMAL FAIL'}")
print(f"  D'Agostino     K^2={stat_da:.4f} p={p_da:.4e}  -> {'NORMAL OK' if p_da>0.05 else 'NON-NORMAL FAIL'}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"Target Distribution  --  {TARGET}", fontsize=14, fontweight="bold")

axes[0].hist(y, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("Histogram"); axes[0].set_xlabel(TARGET)

stats.probplot(y, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot vs Normal")

y_log = np.log1p(y - y.min() + 0.01)
axes[2].hist(y_log, bins=60, color="#DD8452", edgecolor="white", alpha=0.85)
axes[2].set_title("log1p(target) -- candidate transform")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig01_target_distribution.png"))
plt.close()
print("\n  >> fig01_target_distribution.png saved")

# ─── 4. FEATURE DISTRIBUTIONS & OUTLIERS ─────────────────
print("\n" + "="*65)
print("  STEP 4 -- FEATURE DISTRIBUTIONS, SKEWNESS & OUTLIERS")
print("="*65)

num_cols = df[FEATURES].select_dtypes("number").columns.tolist()
skew_df = pd.DataFrame({
    "feature": num_cols,
    "skewness": [round(skew(df[c].values), 4) for c in num_cols],
    "kurtosis": [round(kurtosis(df[c].values, fisher=True), 4) for c in num_cols]
}).sort_values("skewness", key=abs, ascending=False)
print("\n>> Skewness & Kurtosis per feature:")
print(skew_df.to_string(index=False))

# IQR-based outlier count
outlier_df_rows = []
for col in num_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    outlier_df_rows.append(dict(feature=col, n_outliers=n_out,
                                pct=round(100*n_out/len(df),2)))
outlier_df = pd.DataFrame(outlier_df_rows).sort_values("n_outliers", ascending=False)
print("\n>> IQR outlier counts:")
print(outlier_df.to_string(index=False))

# Boxplot grid
n_cols_plot = 4
n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, 3*n_rows_plot))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].boxplot(df[col].values, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    axes[i].set_title(col, fontsize=8)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Boxplots -- All Numeric Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig02_boxplots.png"))
plt.close()
print("\n  >> fig02_boxplots.png saved")

# ─── 5. CORRELATION HEATMAP ───────────────────────────────
print("\n" + "="*65)
print("  STEP 5 -- FULL CORRELATION HEATMAP")
print("="*65)

corr_matrix = df[num_cols + [TARGET]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.4,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Pearson Correlation Matrix (lower triangle)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig03_correlation_heatmap.png"))
plt.close()
print("  >> fig03_correlation_heatmap.png saved")

print("\n[PART 1 COMPLETE] -- proceed to part 2 for VIF, feature engineering & model prep.\n")
