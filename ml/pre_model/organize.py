# -*- coding: utf-8 -*-
"""Organise pre_model outputs into themed sub-folders + generate .ipynb notebooks."""

import os, shutil, json
BASE = os.path.dirname(os.path.abspath(__file__))

# ── folder definitions ────────────────────────────────────
FOLDERS = {
    "01_eda": {
        "files": [
            "correlation_table.csv",
            "fig01_target_distribution.png",
            "fig02_boxplots.png",
            "fig03_correlation_heatmap.png",
        ]
    },
    "02_multicollinearity": {
        "files": [
            "vif_table.csv",
            "fig04_vif.png",
        ]
    },
    "03_scatter_features": {
        "files": [
            "fig05_scatter_top9.png",
        ]
    },
    "04_model_comparison": {
        "files": [
            "ols_coefficients.csv",
            "final_model_comparison.csv",
            "fig06_cv_boxplots.png",
            "fig07_regularization_paths.png",
            "fig08_best_fit_line.png",
            "fig09_residual_diagnostics.png",
        ]
    },
    "05_target_selection": {
        "files": [
            "target_screening_results.csv",
            "figTS1_target_ranking.png",
            "figTS2_lasso_coef_heatmap.png",
            "figTS3_lasso_paths_top4.png",
            "figTS4_target_quality_map.png",
        ]
    },
    "06_splits": {
        "files": [
            "final_features.csv",
            "X_train_s.csv", "X_val_s.csv", "X_test_s.csv",
            "y_train.csv",   "y_val.csv",   "y_test.csv",
        ]
    },
}

# ── move files ────────────────────────────────────────────
for folder, info in FOLDERS.items():
    dest = os.path.join(BASE, folder)
    os.makedirs(dest, exist_ok=True)
    for fname in info["files"]:
        src = os.path.join(BASE, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest, fname))
            print(f"  moved {fname} -> {folder}/")

print("\nAll files moved.\n")

# ── notebook builder helper ───────────────────────────────
def nb(cells):
    """Build a minimal nbformat v4 notebook dict."""
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"}
        },
        "cells": cells
    }

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": os.urandom(4).hex()}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source,
            "outputs": [], "execution_count": None, "id": os.urandom(4).hex()}

def save_nb(folder, name, cells):
    path = os.path.join(BASE, folder, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"  created {folder}/{name}")

# ═══════════════════════════════════════════════════════════
# NOTEBOOK 1 — EDA
# ═══════════════════════════════════════════════════════════
save_nb("01_eda", "eda_results.ipynb", [
    md("# Exploratory Data Analysis (EDA)\n\nThis notebook explains every artefact produced during the EDA phase of the Pre-ML study."),

    md("## 1. Correlation Table\n`correlation_table.csv` — Pearson **r** and Spearman **ρ** of every feature with the chosen target `mental_wellness_index`.\n\n| Column | Meaning |\n|---|---|\n| `pearson_r` | Linear correlation (sensitive to outliers) |\n| `pearson_p` | p-value — is the linear correlation statistically significant? |\n| `spearman_r` | Rank-based correlation (robust to outliers & non-linearity) |\n| `spearman_p` | p-value for the rank correlation |\n\n**Rule of thumb:** |r| > 0.3 = moderate, |r| > 0.5 = strong."),

    code("import pandas as pd\ncorr = pd.read_csv('correlation_table.csv')\ncorr.sort_values('pearson_r', key=abs, ascending=False)"),

    md("## 2. Target Distribution\n`fig01_target_distribution.png`\n\n- **Left panel** — raw histogram: the target is *extremely* right-skewed (skewness ≈ 17.5). Most values cluster around 14.8.\n- **Centre panel** — Q-Q plot: points deviate sharply from the diagonal → the target is **not normally distributed**.\n- **Right panel** — `log1p` transform flattens the spike somewhat. A power transform (Box-Cox / log) may be needed before modelling.\n\n**Normality tests performed:**\n\n| Test | Result |\n|---|---|\n| Shapiro-Wilk | PASS (artefact of large N) |\n| Kolmogorov-Smirnov | FAIL — distribution ≠ Normal |\n| D'Agostino K² | FAIL — excess skewness & kurtosis |"),

    code("from IPython.display import Image\nImage('fig01_target_distribution.png', width=900)"),

    md("## 3. Feature Boxplots\n`fig02_boxplots.png`\n\nBoxplots reveal the spread, median, and outlier count for every numeric feature.\n\n- **Box** = IQR (25th–75th percentile)\n- **Line inside box** = median\n- **Whiskers** = 1.5 × IQR\n- **Dots beyond whiskers** = outliers\n\nFeatures like `age`, `mood_level`, and `productivity_score` have high outlier counts (>25%). This is expected in survey data and does not necessarily mean errors."),

    code("Image('fig02_boxplots.png', width=950)"),

    md("## 4. Pearson Correlation Heatmap\n`fig03_correlation_heatmap.png`\n\nThe lower-triangular heatmap shows pairwise Pearson correlations across all features + target.\n\n- **Red** = strong positive correlation\n- **Blue** = strong negative correlation\n- **White** = near-zero correlation\n\nKey observations:\n- `productivity_score` & `mental_wellness_index` are positively correlated (r ≈ 0.49)\n- `work_screen_hours` and `sleep_quality` show a moderate negative relationship\n- The occupation dummies (`occ_*`) are collinear with each other (expected — they form one hot-encoded group)"),

    code("Image('fig03_correlation_heatmap.png', width=950)"),
])

# ═══════════════════════════════════════════════════════════
# NOTEBOOK 2 — VIF
# ═══════════════════════════════════════════════════════════
save_nb("02_multicollinearity", "vif_analysis.ipynb", [
    md("# Multicollinearity Analysis — Variance Inflation Factor (VIF)\n\nMulticollinearity occurs when two or more features are highly linearly related. It inflates coefficient variance and makes OLS estimates unstable."),

    md("## Mathematical Definition\n\n$$VIF_j = \\frac{1}{1 - R^2_j}$$\n\nwhere $R^2_j$ is obtained by regressing feature $j$ on **all other features**.\n\n| VIF | Interpretation |\n|---|---|\n| 1 | No collinearity |\n| 1–5 | Moderate (acceptable) |\n| 5–10 | High — investigate |\n| > 10 | Severe — remove or combine |\n| ∞ | Perfect collinearity (exact linear dependence) |"),

    md("## VIF Table"),
    code("import pandas as pd\nvif = pd.read_csv('vif_table.csv').sort_values('VIF', ascending=False)\nprint(vif.to_string(index=False))"),

    md("## VIF Chart\n`fig04_vif.png`\n\n- **Red bars** (VIF > 10) = the four occupation dummies (`occ_Employed`, `occ_Retired`, `occ_Other`, `occ_Unemployed`). They are perfect complements — knowing three determines the fourth. VIF = ∞.\n- **Blue bars** = all other features are well below 5 → no multicollinearity problem.\n\n**Action taken:** The four occupation dummies were dropped from the final feature set before model fitting."),

    code("from IPython.display import Image\nImage('fig04_vif.png', width=800)"),

    md("## Why Does This Matter?\n\nIf we keep perfectly collinear features:\n- $(X^TX)$ becomes **singular** (non-invertible) → OLS has no unique solution\n- Ridge can still handle this (adds $\\alpha I$ to diagonal) but coefficients will be arbitrary\n- Lasso will arbitrarily keep one of the correlated features and discard the others\n\nDropping all but one dummy (or all dummies as done here) is the standard fix."),
])

# ═══════════════════════════════════════════════════════════
# NOTEBOOK 3 — SCATTER FEATURES
# ═══════════════════════════════════════════════════════════
save_nb("03_scatter_features", "scatter_analysis.ipynb", [
    md("# Scatter Plots — Top Features vs Target\n\nThis notebook analyses the relationship between the top-9 most correlated features and the target `mental_wellness_index`."),

    md("## Figure: Top-9 Scatter Plots\n`fig05_scatter_top9.png`\n\nEach panel shows:\n- **Scatter points** (semi-transparent) — one dot per observation\n- **Red regression line** — OLS best-fit line through the cloud\n- **r value** — Pearson correlation in the title\n\n### Reading the plots\n| Pattern | Meaning |\n|---|---|\n| Points close to the line | Strong linear relationship |\n| Wide cloud | Weak or noisy relationship |\n| Curved pattern | Non-linear relationship → may need polynomial features |\n| Funnel shape | Heteroscedasticity — variance grows with x |"),

    code("from IPython.display import Image\nImage('fig05_scatter_top9.png', width=950)"),

    md("## Engineered Features\n\nBeyond the raw features, five interaction/ratio terms were created:\n\n| Feature | Formula | Rationale |\n|---|---|---|\n| `total_screen` | screen + tech + work hours | Total daily screen exposure |\n| `leisure_screen` | social_media + gaming hours | Passive/recreational screen time |\n| `activity_ratio` | physical_activity / (screen + 1) | Balance between movement and sedentary behaviour |\n| `sleep_stress_interact` | sleep_hours × (10 − stress_level) | Restorative sleep under low stress |\n| `support_online` | support_systems × online_support | Combined support network |\n\nThe interaction term `sleep_stress_interact` achieved r = +0.090 with the target — the strongest among engineered features."),
])

# ═══════════════════════════════════════════════════════════
# NOTEBOOK 4 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
save_nb("04_model_comparison", "model_comparison.ipynb", [
    md("# Model Comparison — OLS, Ridge, Lasso, ElasticNet\n\nThis notebook explains every figure and table produced during the modelling phase."),

    md("## 1. 10-Fold Cross-Validation\n`fig06_cv_boxplots.png`\n\nCross-validation gives a more reliable generalisation estimate than a single train/val split.\n\n**Protocol:** K=10, shuffled, random_state=42  \n**Metric reported:** R² and RMSE across 10 folds\n\nThe box shows the spread across folds — a **narrow box** means the model is stable; a **wide box** means it is sensitive to which data it trains on."),

    code("from IPython.display import Image\nImage('fig06_cv_boxplots.png', width=800)"),

    md("## 2. Regularization Paths\n`fig07_regularization_paths.png`\n\n### Ridge (L2) — left panel\n$$\\hat{\\beta}_{Ridge} = (X^TX + \\alpha I)^{-1} X^T y$$\n\nAs $\\alpha$ increases (left to right on the x-axis):\n- All coefficients shrink **smoothly toward 0**\n- No feature is ever exactly zeroed\n- The vertical dashed line marks the best $\\alpha$ chosen by RidgeCV\n\n### Lasso (L1) — right panel\n$$L_{Lasso} = \\|y - X\\beta\\|^2 + \\alpha \\|\\beta\\|_1$$\n\nAs $\\alpha$ increases:\n- Coefficients hit **exactly 0** (lines touch the axis) — feature selection!\n- The order lines reach 0 reveals feature importance\n- Best $\\alpha$ chosen by LassoCV (10-fold)"),

    code("Image('fig07_regularization_paths.png', width=950)"),

    md("## 3. Best-Fit Line — Predicted vs Actual\n`fig08_best_fit_line.png`\n\nThe diagonal dashed line = **perfect prediction** (ŷ = y).  \nPoints close to this line = accurate model.\n\n- A **fan shape** → heteroscedasticity (errors grow with y)\n- A **banana curve** → missing non-linearity\n- **Vertical stripes** → the target has discrete clusters (as seen here around y=14.8)\n\n### Results Summary"),

    code("import pandas as pd\ndf = pd.read_csv('final_model_comparison.csv')\ndf[df['label'].str.contains('Test')].sort_values('R2', ascending=False)"),

    code("Image('fig08_best_fit_line.png', width=950)"),

    md("## 4. Residual Diagnostics\n`fig09_residual_diagnostics.png`\n\nResiduals $\\varepsilon = y - \\hat{y}$ should satisfy the Gauss-Markov conditions:\n\n| Panel | Test | Ideal |\n|---|---|---|\n| Residuals vs Fitted | Homoscedasticity | Random cloud around 0 |\n| Q-Q Plot | Normality of errors | Points on diagonal |\n| Histogram | Normality | Bell-shaped, centred at 0 |\n| Scale-Location | Homoscedasticity | Flat trend line |\n| Residual Series | No autocorrelation | No pattern over time |\n| ACF | No autocorrelation | Bars inside blue band |\n\n**Durbin-Watson statistic = 1.99** (ideal ≈ 2) → no autocorrelation ✓  \n**Breusch-Pagan proxy p < 0.05** → heteroscedasticity detected → consider WLS or log-transform of target."),

    code("Image('fig09_residual_diagnostics.png', width=950)"),

    md("## 5. OLS Coefficients"),
    code("coef = pd.read_csv('ols_coefficients.csv').sort_values('coefficient', key=abs, ascending=False)\ncoef"),
])

# ═══════════════════════════════════════════════════════════
# NOTEBOOK 5 — TARGET SELECTION
# ═══════════════════════════════════════════════════════════
save_nb("05_target_selection", "target_selection.ipynb", [
    md("# Lasso-Based Target Selection\n\nInstead of assuming `mental_wellness_index` is the only interesting target, we asked: **which columns in our dataset are most predictable?**\n\nFor every continuous column we:\n1. Treated it as `y`, all other columns as `X`\n2. Fitted `LassoCV` (10-fold) to find the best regularisation\n3. Recorded the holdout R², selected features, and regularisation path"),

    md("## Why Lasso for target screening?\n\n| Property | Benefit |\n|---|---|\n| L1 penalty zeros weak features | Tells us which features *actually* explain the target |\n| Cross-validated alpha | Unbiased estimate of generalisation |\n| Sparsity | Few selected features = cleaner, more interpretable model |\n| R² score | Direct measure of predictability |"),

    md("## Results Table"),
    code("import pandas as pd\ndf = pd.read_csv('target_screening_results.csv')\ndf.sort_values('holdout_R2', ascending=False)"),

    md("## Figure 1 — Predictability Ranking\n`figTS1_target_ranking.png`\n\n- **Green bars** (R² ≥ 0.5) = excellent targets\n- **Orange bars** (0.25 ≤ R² < 0.5) = worth studying\n- **Red bars** (R² < 0.25) = poor linear targets\n\nThe right panel shows how many features Lasso selected — fewer features = cleaner signal."),

    code("from IPython.display import Image\nImage('figTS1_target_ranking.png', width=900)"),

    md("## Figure 2 — Lasso Coefficient Heatmap\n`figTS2_lasso_coef_heatmap.png`\n\nRows = candidate targets (sorted by R²), Columns = features.\n- **Red** = positive coefficient (feature increases target)\n- **Blue** = negative coefficient\n- **White** = coefficient zeroed by Lasso (feature not useful for this target)\n\nThis reveals *cross-target patterns*: `age` is selected by almost every target model, making it a universal predictor in this dataset."),

    code("Image('figTS2_lasso_coef_heatmap.png', width=1000)"),

    md("## Figure 3 — Regularization Paths (Top-4 Targets)\n`figTS3_lasso_paths_top4.png`\n\nEach line = one feature. As `alpha` increases (more regularisation), lines collapse to 0.\n- Lines that stay non-zero until large alpha = **robust predictors**\n- Lines that hit 0 early = **weak or redundant features**"),

    code("Image('figTS3_lasso_paths_top4.png', width=950)"),

    md("## Figure 4 — Target Quality Map\n`figTS4_target_quality_map.png`\n\nX-axis = |skewness| of the target; Y-axis = Lasso R²; Colour = # features selected.\n\n**Ideal target** (top-left corner): low skewness + high R² + few features selected.\n\n`mental_wellness_index` and `productivity_score` are the two clear winners, though both are heavily skewed — a **log1p or Box-Cox transform** before modelling is recommended."),

    code("Image('figTS4_target_quality_map.png', width=800)"),

    md("## Recommendations\n\n| Tier | Targets | Action |\n|---|---|---|\n| **Excellent** | `mental_wellness_index`, `productivity_score` | Use as primary targets; apply log-transform |\n| **Good** | `age`, `mental_health_status`, `online_support_usage`, `support_systems_access`, `stress_level`, `screen_time_hours` | Use as secondary studies |\n| **Poor** | `gaming_hours`, `social_media_hours`, `physical_activity_hours`, `social_hours_per_week` | Avoid — not linearly predictable |"),
])

# ═══════════════════════════════════════════════════════════
# NOTEBOOK 6 — SPLITS
# ═══════════════════════════════════════════════════════════
save_nb("06_splits", "splits_overview.ipynb", [
    md("# Train / Validation / Test Splits\n\nThis folder contains the final prepared arrays used for model training, selection, and evaluation."),

    md("## Split Strategy\n\nA **70 / 15 / 15** random split was used:\n\n| Set | Purpose | Rows |\n|---|---|---|\n| **Train** | Fit model parameters (β) | 24,518 |\n| **Validation** | Tune hyperparameters (α in Ridge/Lasso) | 5,255 |\n| **Test** | Final unbiased evaluation — **never touched during training** | 5,255 |\n\n`random_state=42` ensures full reproducibility."),

    md("## Feature Scaling\n\nAll X files are **StandardScaler-transformed**:\n\n$$z = \\frac{x - \\mu_{train}}{\\sigma_{train}}$$\n\n- `\\mu` and `\\sigma` computed **only on X_train** → no data leakage\n- The same statistics are applied to X_val and X_test\n- y vectors are kept in **original scale** (not standardised)"),

    md("## Feature List"),
    code("import pandas as pd\nfeats = pd.read_csv('final_features.csv', header=None, names=['feature'])\nprint(f'Total features: {len(feats)}')\nfeats"),

    md("## Data Preview — X_train_s"),
    code("X_train = pd.read_csv('X_train_s.csv')\nprint(f'Shape: {X_train.shape}')\nX_train.describe().T[['mean','std','min','max']]"),

    md("## Data Preview — y_train"),
    code("y_train = pd.read_csv('y_train.csv')\nprint(f'Shape: {y_train.shape}')\ny_train.describe()"),

    md("## Distribution Check — Does Each Split Have Similar Target Stats?"),
    code("""import pandas as pd, matplotlib.pyplot as plt

y_tr = pd.read_csv('y_train.csv')['mental_wellness_index']
y_va = pd.read_csv('y_val.csv')['mental_wellness_index']
y_te = pd.read_csv('y_test.csv')['mental_wellness_index']

print(f'Train  mean={y_tr.mean():.3f}  std={y_tr.std():.3f}')
print(f'Val    mean={y_va.mean():.3f}  std={y_va.std():.3f}')
print(f'Test   mean={y_te.mean():.3f}  std={y_te.std():.3f}')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(y_tr, bins=60, alpha=0.5, label='Train',  color='#2ecc71')
ax.hist(y_va, bins=60, alpha=0.5, label='Val',    color='#3498db')
ax.hist(y_te, bins=60, alpha=0.5, label='Test',   color='#e74c3c')
ax.set_title('Target Distribution Across Splits')
ax.legend(); plt.tight_layout(); plt.show()
"""),

    md("## Anti-Leakage Checklist\n\n| Rule | Status |\n|---|---|\n| Scaler fitted only on X_train | OK |\n| Hyperparameter search uses CV on train+val only | OK |\n| Test set touched only for final evaluation | OK |\n| No target information in features | OK |"),
])

print("\nAll notebooks created successfully.")
print("\nFinal structure:")
for folder in sorted(FOLDERS.keys()):
    fpath = os.path.join(BASE, folder)
    files = os.listdir(fpath)
    print(f"\n  {folder}/")
    for f in sorted(files):
        print(f"    - {f}")
