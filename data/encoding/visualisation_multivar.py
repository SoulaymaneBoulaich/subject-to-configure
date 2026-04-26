"""
Visualisation Multivariée — Digital Health Dataset
====================================================
7 graphes complets avec interprétation
Prérequis : pip install pandas matplotlib seaborn scipy

Usage : python visualisation_multivar.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy import stats

# ─── CONFIG GLOBALE ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#999999",
    "ytick.color":      "#999999",
    "text.color":       "#eeeeee",
    "grid.color":       "#2a2a2a",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlepad":    12,
})

MH_COLORS  = {"Excellent": "#3266ad", "Good": "#5ca85c", "Fair": "#e8a838", "Poor": "#c94040"}
MH_ORDER   = ["Excellent", "Good", "Fair", "Poor"]
GAD_ORDER  = ["Minimal", "Mild", "Moderate", "Severe"]
PHQ_ORDER  = ["None-Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"]
PLAT_COLOR = "#5e9bdc"

# ─── CHARGEMENT DES DONNÉES ───────────────────────────────────────────────────
CSV_PATH = "master_data_imputed.csv"

print("Chargement des données...")
df = pd.read_csv(CSV_PATH)   # ← ligne critique corrigée
print(f"  → {len(df):,} lignes × {df.shape[1]} colonnes")

# Nettoyage stress_level (mixte texte/numérique)
def clean_stress(v):
    try:
        f = float(v)
        if f < 4:  return "Low"
        if f > 7:  return "High"
        return "Medium"
    except:
        return str(v)

df["stress_level_clean"] = df["stress_level"].apply(clean_stress)

# Conversion numérique
for col in ["screen_time_hours", "sleep_hours", "mental_wellness_index",
            "gad7_score", "phq9_score", "social_media_hours", "physical_activity_hours"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("  → Nettoyage OK\n")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 1 — Distribution santé mentale (Donut)
# ══════════════════════════════════════════════════════════════════════════════
def graphe1():
    counts = df["mental_health_status"].value_counts().reindex(MH_ORDER).fillna(0)
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    wedges, _ = ax.pie(
        counts,
        colors=[MH_COLORS[m] for m in MH_ORDER],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0f0f0f", linewidth=2),
        radius=1.0,
    )

    ax.text(0, 0.08, f"{total:,.0f}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#eeeeee")
    ax.text(0, -0.18, "utilisateurs", ha="center", va="center",
            fontsize=10, color="#888888")

    legend_labels = [
        f"{m}  —  {counts[m]:,.0f}  ({counts[m]/total*100:.1f}%)"
        for m in MH_ORDER
    ]
    patches = [mpatches.Patch(color=MH_COLORS[m], label=l)
               for m, l in zip(MH_ORDER, legend_labels)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=10, ncol=2, labelcolor="#cccccc")

    ax.set_title("Graphe 1 — Distribution de la santé mentale", color="#eeeeee")
    plt.tight_layout()
    plt.savefig("graphe1_donut.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 1 sauvegardé → graphe1_donut.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 2 — Temps écran × Santé mentale (Boxplot)
# ══════════════════════════════════════════════════════════════════════════════
def graphe2():
    data = [df[df["mental_health_status"] == m]["screen_time_hours"].dropna()
            for m in MH_ORDER]

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=True,
        widths=0.5,
        medianprops=dict(color="#ffffff", linewidth=2),
        whiskerprops=dict(color="#666666"),
        capprops=dict(color="#666666"),
        flierprops=dict(marker="o", markerfacecolor="#555555",
                        markersize=2, alpha=0.3, linestyle="none"),
    )

    for patch, mh in zip(bp["boxes"], MH_ORDER):
        patch.set_facecolor(MH_COLORS[mh])
        patch.set_alpha(0.85)

    means = [d.mean() for d in data]
    ax.plot(range(1, len(MH_ORDER)+1), means, "D",
            color="#ffffff", markersize=6, zorder=5, label="Moyenne")

    ax.set_xticks(range(1, len(MH_ORDER)+1))
    ax.set_xticklabels(MH_ORDER)
    ax.set_xlabel("État de santé mentale")
    ax.set_ylabel("Temps d'écran (h/jour)")
    ax.set_title("Graphe 2 — Temps d'écran vs Santé mentale")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graphe2_boxplot.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 2 sauvegardé → graphe2_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 3 — Sommeil × Stress (Violinplot)
# ══════════════════════════════════════════════════════════════════════════════
def graphe3():
    stress_order  = ["Low", "Medium", "High"]
    stress_colors = {"Low": "#5ca85c", "Medium": "#e8a838", "High": "#c94040"}

    sub = df[df["stress_level_clean"].isin(stress_order)].copy()
    sub["sleep_hours"] = pd.to_numeric(sub["sleep_hours"], errors="coerce")
    sub = sub.dropna(subset=["sleep_hours"])

    fig, ax = plt.subplots(figsize=(9, 6))

    parts = ax.violinplot(
        [sub[sub["stress_level_clean"] == s]["sleep_hours"].values
         for s in stress_order],
        positions=range(len(stress_order)),
        showmedians=True,
        showextrema=False,
        widths=0.65,
    )

    for body, s in zip(parts["bodies"], stress_order):
        body.set_facecolor(stress_colors[s])
        body.set_alpha(0.75)
    parts["cmedians"].set_color("#ffffff")
    parts["cmedians"].set_linewidth(2)

    for i, s in enumerate(stress_order):
        d = sub[sub["stress_level_clean"] == s]["sleep_hours"]
        q1, med, q3 = d.quantile([0.25, 0.5, 0.75])
        ax.plot([i, i], [q1, q3], color="#ffffff", linewidth=4,
                solid_capstyle="round", zorder=3)
        ax.plot(i, med, "o", color="#0f0f0f", markersize=5, zorder=4)

    ax.set_xticks(range(len(stress_order)))
    ax.set_xticklabels(stress_order)
    ax.set_xlabel("Niveau de stress")
    ax.set_ylabel("Heures de sommeil (h/jour)")
    ax.set_title("Graphe 3 — Sommeil vs Niveau de stress")
    ax.grid(axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=stress_colors[s], label=s) for s in stress_order]
    ax.legend(handles=patches, frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig("graphe3_violin.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 3 sauvegardé → graphe3_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 4 — Archétypes utilisateurs (Bar chart horizontal)
# ══════════════════════════════════════════════════════════════════════════════
def graphe4():
    counts = df["user_archetype"].value_counts().sort_values()
    arch_colors = ["#3266ad", "#5ca85c", "#e8a838", "#c94040",
                   "#a44db5", "#e8783a", "#2aa8a8"]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(counts.index, counts.values,
                   color=arch_colors[:len(counts)], height=0.55,
                   edgecolor="#0f0f0f", linewidth=0.5)

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(val + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({val/total*100:.1f}%)",
                va="center", fontsize=9, color="#cccccc")

    ax.set_xlabel("Nombre d'utilisateurs")
    ax.set_title("Graphe 4 — Archétypes d'utilisateurs numériques")
    ax.set_xlim(0, counts.max() * 1.22)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig("graphe4_archetypes.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 4 sauvegardé → graphe4_archetypes.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 5 — Plateformes × Santé mentale (Stacked bar)
# ══════════════════════════════════════════════════════════════════════════════
def graphe5():
    platforms = df["primary_platform"].value_counts().index.tolist()

    records = []
    for p in platforms:
        sub   = df[df["primary_platform"] == p]
        total = len(sub)
        for mh in MH_ORDER:
            pct = (sub["mental_health_status"] == mh).sum() / total * 100
            records.append({"platform": p, "mh": mh, "pct": pct})

    pivot = pd.DataFrame(records).pivot(
        index="platform", columns="mh", values="pct"
    ).reindex(columns=MH_ORDER).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6))

    bottom = np.zeros(len(pivot))
    for mh in MH_ORDER:
        ax.bar(pivot.index, pivot[mh], bottom=bottom,
               color=MH_COLORS[mh], label=mh,
               edgecolor="#0f0f0f", linewidth=0.3)
        bottom += pivot[mh].values

    for i, p in enumerate(pivot.index):
        total = len(df[df["primary_platform"] == p])
        ax.text(i, 103, f"n={total:,}", ha="center",
                fontsize=8, color="#888888")

    ax.set_ylabel("% d'utilisateurs")
    ax.set_title("Graphe 5 — Plateformes vs Santé mentale")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig("graphe5_plateformes.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 5 sauvegardé → graphe5_plateformes.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 6 — Matrice de corrélation (Heatmap)
# ══════════════════════════════════════════════════════════════════════════════
def graphe6():
    num_cols = [
        "screen_time_hours", "sleep_hours", "mental_wellness_index",
        "gad7_score", "phq9_score", "social_media_hours", "physical_activity_hours"
    ]
    labels = [
        "Écran", "Sommeil", "Wellness\nIndex", "GAD-7\nScore",
        "PHQ-9\nScore", "Social\nMedia", "Activité\nphysique"
    ]

    corr = df[num_cols].dropna().corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 9, "color": "#eeeeee"},
        linewidths=0.5, linecolor="#0f0f0f",
        square=True, cbar_kws={"shrink": 0.75},
        xticklabels=labels, yticklabels=labels,
    )

    ax.set_title("Graphe 6 — Matrice de corrélation (Pearson)")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig("graphe6_correlation.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 6 sauvegardé → graphe6_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE 7 — GAD-7 × PHQ-9 (Heatmap de comptage)
# ══════════════════════════════════════════════════════════════════════════════
def graphe7():
    pivot = (
        df.groupby(["gad7_severity", "phq9_severity"])
        .size()
        .reset_index(name="count")
        .pivot(index="gad7_severity", columns="phq9_severity", values="count")
        .reindex(index=GAD_ORDER, columns=PHQ_ORDER)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap,
        annot=True, fmt=",",
        annot_kws={"size": 10},
        linewidths=0.5, linecolor="#0f0f0f",
        cbar_kws={"shrink": 0.75, "label": "Nombre d'utilisateurs"},
    )

    ax.set_xlabel("Sévérité PHQ-9 (dépression)", labelpad=10)
    ax.set_ylabel("Sévérité GAD-7 (anxiété)", labelpad=10)
    ax.set_title("Graphe 7 — Croisement GAD-7 (anxiété) × PHQ-9 (dépression)")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig("graphe7_gad7_phq9.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.show()
    print("✓ Graphe 7 sauvegardé → graphe7_gad7_phq9.png")


# ─── LANCEMENT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  VISUALISATION MULTIVARIÉE — Digital Health Dataset")
    print("=" * 55)

    graphe1()
    graphe2()
    graphe3()
    graphe4()
    graphe5()
    graphe6()
    graphe7()

    print("\n✅ Tous les graphes générés !")
    print("   Les fichiers PNG sont dans le dossier courant.")