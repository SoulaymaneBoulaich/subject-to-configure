import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'data\encoding\master_data_final.csv')

STATUS_MAP = {0:'Good', 1:'Moderate', 2:'Poor', 3:'Critical'}
df['Status'] = df['mental_health_status'].map(STATUS_MAP)
COLORS = ['#2ecc71','#f39c12','#e74c3c','#8e44ad']
PALETTE = {v:c for v,c in zip(['Good','Moderate','Poor','Critical'], COLORS)}
order = ['Good','Moderate','Poor','Critical']

# ── GRAPHE 1 : Donut ──
fig, ax = plt.subplots(figsize=(7,7), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
counts = df['Status'].value_counts().reindex(order)
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=COLORS, startangle=90,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops={'color':'white','fontsize':13,'fontweight':'bold'}
)
for at in autotexts:
    at.set_fontsize(11)
ax.set_title('Mental Health Status Distribution', color='white',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('graph1_donut.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

# ── GRAPHE 2 : Boxplot ──
fig, ax = plt.subplots(figsize=(9,6), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
sns.boxplot(data=df, x='Status', y='screen_time_hours', order=order,
            palette=PALETTE, ax=ax, width=0.5,
            boxprops=dict(edgecolor='white'),
            whiskerprops=dict(color='white'),
            capprops=dict(color='white'),
            medianprops=dict(color='yellow', linewidth=2),
            flierprops=dict(marker='o', color='white', alpha=0.3, markersize=3))
ax.set_title('Screen Time by Mental Health Status', color='white', fontsize=15, fontweight='bold')
ax.set_xlabel('Mental Health Status', color='white', fontsize=12)
ax.set_ylabel('Screen Time (hours)', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')
plt.tight_layout()
plt.savefig('graph2_boxplot.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

# ── GRAPHE 3 : Violin avec moyenne ──
fig, ax = plt.subplots(figsize=(9,6), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
sns.violinplot(data=df, x='Status', y='sleep_hours', order=order,
               palette=PALETTE, ax=ax, inner='quartile', linewidth=1.5)
means = df.groupby('Status')['sleep_hours'].mean().reindex(order)
for i, (status, mean_val) in enumerate(means.items()):
    ax.scatter(i, mean_val, color='white', s=80, zorder=5,
               edgecolors='black', linewidth=1.5, label='Mean' if i==0 else '')
ax.legend(['Mean'], facecolor='#2a2a3e', labelcolor='white', fontsize=11,
          loc='upper right', markerscale=1.2)
ax.set_title('Sleep Hours by Mental Health Status', color='white', fontsize=15, fontweight='bold')
ax.set_xlabel('Mental Health Status', color='white', fontsize=12)
ax.set_ylabel('Sleep Hours', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')
plt.tight_layout()
plt.savefig('graph3_violin.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

# ── GRAPHE 4 : Heatmap Corrélation ──
fig, ax = plt.subplots(figsize=(11,9), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
num_cols = ['age','screen_time_hours','sleep_hours','stress_level',
            'tech_usage_hours','social_media_hours','gaming_hours',
            'physical_activity_hours','mental_health_status']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, linecolor='#333',
            annot_kws={'size':9, 'color':'white'},
            cbar_kws={'shrink':0.8})
ax.set_title('Correlation Heatmap', color='white', fontsize=15, fontweight='bold')
ax.tick_params(colors='white', labelsize=9)
plt.tight_layout()
plt.savefig('graph4_correlation.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

# ── GRAPHE 5 : Bar Stress Level ──
fig, ax = plt.subplots(figsize=(9,6), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
stress_avg = df.groupby('Status')['stress_level'].mean().reindex(order)
bars = ax.bar(order, stress_avg, color=COLORS, edgecolor='white', linewidth=1.2, width=0.5)
for bar, val in zip(bars, stress_avg):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
ax.set_title('Average Stress Level by Mental Health Status', color='white', fontsize=15, fontweight='bold')
ax.set_xlabel('Mental Health Status', color='white', fontsize=12)
ax.set_ylabel('Average Stress Level', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')
plt.tight_layout()
plt.savefig('graph5_stress_bar.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

# ── GRAPHE 6 : Scatter Screen Time vs Sleep Hours ──
fig, ax = plt.subplots(figsize=(9,6), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
for status, color in PALETTE.items():
    sub = df[df['Status']==status]
    ax.scatter(sub['screen_time_hours'], sub['sleep_hours'],
               c=color, alpha=0.4, s=10, label=status)
ax.set_title('Screen Time vs Sleep Hours', color='white', fontsize=15, fontweight='bold')
ax.set_xlabel('Screen Time (hours)', color='white', fontsize=12)
ax.set_ylabel('Sleep Hours', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')
legend = ax.legend(title='Status', facecolor='#2a2a3e', labelcolor='white', title_fontsize=11)
legend.get_title().set_color('white')
plt.tight_layout()
plt.savefig('graph6_scatter.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

# ── GRAPHE 7 : KDE Age Distribution ──
fig, ax = plt.subplots(figsize=(10,6), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
for status, color in PALETTE.items():
    sub = df[df['Status']==status]
    sns.kdeplot(sub['age'], ax=ax, color=color, linewidth=2.5,
                label=status, fill=True, alpha=0.2)
ax.set_title('Age Distribution by Mental Health Status', color='white', fontsize=15, fontweight='bold')
ax.set_xlabel('Age', color='white', fontsize=12)
ax.set_ylabel('Density', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')
legend = ax.legend(title='Status', facecolor='#2a2a3e', labelcolor='white', title_fontsize=11)
legend.get_title().set_color('white')
plt.tight_layout()
plt.savefig('graph7_age_kde.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()

print("✅ Tous les graphiques sauvegardés !")