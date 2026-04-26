# -*- coding: utf-8 -*-
"""
============================================================
 PRE-ML STUDY -- PART 2
 VIF, Multicollinearity, Feature Engineering,
 Scatter plots vs target, Train/Test split strategy
============================================================
"""

import warnings, os, sys
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, skew

SAVE_DIR = os.path.dirname(__file__)
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# ── Helper: load clean data (same pipeline as part 1) ─────
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
TARGET  = "mental_wellness_index"
FEATURES = [c for c in df.columns if c != TARGET]

# ─── 6. VARIANCE INFLATION FACTOR (VIF) ──────────────────
print("="*65)
print("  STEP 6 -- MULTICOLLINEARITY ANALYSIS (VIF)")
print("="*65)

from sklearn.linear_model import LinearRegression

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R^2_j)  where R^2_j is from regressing X_j on all others."""
    vif_vals = []
    cols = list(X.columns)
    for i, col in enumerate(cols):
        X_others = X.drop(columns=[col]).values
        X_j      = X[col].values
        reg = LinearRegression().fit(X_others, X_j)
        r2  = reg.score(X_others, X_j)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_vals.append({"feature": col, "VIF": round(vif, 3)})
    return pd.DataFrame(vif_vals).sort_values("VIF", ascending=False)

X_num = df[FEATURES].select_dtypes("number")
vif_df = compute_vif(X_num)
print("\n>> VIF Table  (rule: VIF > 10 -> severe multicollinearity)")
print(vif_df.to_string(index=False))

HIGH_VIF_THRESH = 10
drop_vif = vif_df[vif_df["VIF"] > HIGH_VIF_THRESH]["feature"].tolist()
print(f"\n  Features with VIF > {HIGH_VIF_THRESH}: {drop_vif}")
vif_df.to_csv(os.path.join(SAVE_DIR, "vif_table.csv"), index=False)

# Bar chart
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#e74c3c" if v > 10 else "#4C72B0" for v in vif_df["VIF"]]
ax.barh(vif_df["feature"], vif_df["VIF"], color=colors, edgecolor="white")
ax.axvline(5,  color="orange", linestyle="--", linewidth=1.5, label="VIF=5 (moderate)")
ax.axvline(10, color="red",    linestyle="--", linewidth=1.5, label="VIF=10 (severe)")
ax.set_xlabel("VIF"); ax.set_title("Variance Inflation Factor per Feature", fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig04_vif.png")); plt.close()
print("  >> fig04_vif.png saved")

# ─── 7. SCATTER PLOTS -- TOP FEATURES vs TARGET ───────────
print("\n" + "="*65)
print("  STEP 7 -- SCATTER PLOTS (top 9 features vs target)")
print("="*65)

corr_abs = X_num.corrwith(df[TARGET]).abs().sort_values(ascending=False)
top9 = corr_abs.head(9).index.tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(f"Top-9 Features vs {TARGET}", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), top9):
    ax.scatter(df[col], df[TARGET], alpha=0.15, s=8, color="#4C72B0")
    m, b, r, p, se = stats.linregress(df[col], df[TARGET])
    xr = np.linspace(df[col].min(), df[col].max(), 200)
    ax.plot(xr, m*xr + b, color="#e74c3c", linewidth=2)
    ax.set_title(f"{col}  (r={r:.3f})", fontsize=9)
    ax.set_xlabel(col, fontsize=8); ax.set_ylabel(TARGET, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fig05_scatter_top9.png")); plt.close()
print("  >> fig05_scatter_top9.png saved")

# ─── 8. FEATURE ENGINEERING ──────────────────────────────
print("\n" + "="*65)
print("  STEP 8 -- FEATURE ENGINEERING")
print("="*65)

df["total_screen"] = df["screen_time_hours"] + df["tech_usage_hours"] + df["work_screen_hours"]
df["leisure_screen"] = df["social_media_hours"] + df["gaming_hours"]
df["activity_ratio"] = df["physical_activity_hours"] / (df["screen_time_hours"] + 1)
df["sleep_stress_interact"] = df["sleep_hours"] * (10 - df["stress_level"])
df["support_online"] = df["support_systems_access"] * df["online_support_usage"]

new_feats = ["total_screen", "leisure_screen", "activity_ratio",
             "sleep_stress_interact", "support_online"]

for f in new_feats:
    r, p = pearsonr(df[f], df[TARGET])
    print(f"  {f:30s}  r={r:+.4f}  p={p:.3e}")

print("\n>> Engineered features added to dataset")

# ─── 9. FINAL FEATURE SET SELECTION ──────────────────────
print("\n" + "="*65)
print("  STEP 9 -- FINAL FEATURE SET")
print("="*65)

# Remove one of each highly collinear pair (use VIF guidance)
# Also drop binary dummies with very low variance
ALL_FEATURES_FINAL = [c for c in df.columns if c != TARGET]

# Remove features already captured by engineered counterparts
REMOVE = [c for c in drop_vif if c in ALL_FEATURES_FINAL]

# Also remove near-zero variance features
nzv = [c for c in ALL_FEATURES_FINAL
       if df[c].std() < 0.01 and c not in new_feats]
REMOVE += nzv
REMOVE = list(set(REMOVE))

FINAL_FEATURES = [c for c in ALL_FEATURES_FINAL if c not in REMOVE]
print(f"  Removed (high VIF / NZV) : {REMOVE}")
print(f"  Final feature count      : {len(FINAL_FEATURES)}")
print(f"  Final features           : {FINAL_FEATURES}")

# ─── 10. TRAIN / VALIDATION / TEST SPLIT ─────────────────
print("\n" + "="*65)
print("  STEP 10 -- STRATIFIED TRAIN / VAL / TEST SPLIT  (70/15/15)")
print("="*65)

from sklearn.model_selection import train_test_split

X = df[FINAL_FEATURES].values
y = df[TARGET].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42)

print(f"  Train : {X_train.shape[0]} rows")
print(f"  Val   : {X_val.shape[0]} rows")
print(f"  Test  : {X_test.shape[0]} rows")

for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"  {name} target -- mean={y_split.mean():.3f}  std={y_split.std():.3f}")

# ─── 11. FEATURE SCALING -- WHY AND HOW ──────────────────
print("\n" + "="*65)
print("  STEP 11 -- FEATURE SCALING (StandardScaler)")
print("="*65)

from sklearn.preprocessing import StandardScaler

print("""
  Mathematical rationale:
  OLS coefficients are scale-dependent; Ridge/Lasso penalize
  beta directly, so all features MUST be on the same scale.

  StandardScaler:  z = (x - mu) / sigma

  We fit ONLY on X_train to avoid data leakage, then
  transform X_val and X_test with the training statistics.
""")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print("  Scaler fitted on training set only -- no leakage OK")
print(f"  Train scaled -- mean≈0: {X_train_s.mean(axis=0).mean():.6f}")
print(f"  Train scaled -- std≈1 : {X_train_s.std(axis=0).mean():.6f}")

# ── Save artefacts for Part 3 ─────────────────────────────
np.save(os.path.join(SAVE_DIR, "X_train_s.npy"), X_train_s)
np.save(os.path.join(SAVE_DIR, "X_val_s.npy"),   X_val_s)
np.save(os.path.join(SAVE_DIR, "X_test_s.npy"),  X_test_s)
np.save(os.path.join(SAVE_DIR, "y_train.npy"),    y_train)
np.save(os.path.join(SAVE_DIR, "y_val.npy"),      y_val)
np.save(os.path.join(SAVE_DIR, "y_test.npy"),     y_test)
pd.Series(FINAL_FEATURES).to_csv(os.path.join(SAVE_DIR, "final_features.csv"), index=False, header=False)

print("\n  >> All numpy arrays + feature list saved for Part 3")
print("\n[PART 2 COMPLETE] -- proceed to part 3 for model fitting & regularization.\n")
