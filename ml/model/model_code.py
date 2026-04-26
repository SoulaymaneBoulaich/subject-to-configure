# ============================================================
#  Mental Health - Régression Linéaire
#  Cibles : mental_wellness_index & productivity_score
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
df = pd.read_csv(r'data\encoding\master_data_final.csv')
print("Shape:", df.shape)

# Supprimer les colonnes avec variance nulle
cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
if cols_to_drop:
    print(f"Colonnes supprimées: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# ============================================================
# 2. FONCTION D'ÉVALUATION
# ============================================================
def evaluer_modele(nom_cible, df):
    print(f"\n{'='*55}")
    print(f"  RÉGRESSION LINÉAIRE → {nom_cible}")
    print('='*55)

    # Features et cible
    X = df.drop(columns=['mental_wellness_index', 'productivity_score', 'mental_health_status'], errors='ignore')
    y = df[nom_cible]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Modèle
    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    # Métriques
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # ── Graphe 1 : Valeurs réelles vs prédites ──
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title(f'Réel vs Prédit — {nom_cible}')
    plt.tight_layout()
    plt.savefig(f'reel_vs_predit_{nom_cible}.png', dpi=150)
    plt.show()

    # ── Graphe 2 : Distribution des résidus ──
    residus = y_test - y_pred
    plt.figure(figsize=(7, 4))
    sns.histplot(residus, kde=True, color='coral', bins=40)
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Distribution des résidus — {nom_cible}')
    plt.xlabel('Résidu')
    plt.tight_layout()
    plt.savefig(f'residus_{nom_cible}.png', dpi=150)
    plt.show()

    # ── Graphe 3 : Top 10 coefficients ──
    coef_df = pd.DataFrame({
        'Feature':     X.columns,
        'Coefficient': model.coef_
    }).reindex(pd.Series(model.coef_).abs().sort_values(ascending=False).index)
    coef_df = coef_df.head(10)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
    plt.title(f'Top 10 Coefficients — {nom_cible}')
    plt.tight_layout()
    plt.savefig(f'coefficients_{nom_cible}.png', dpi=150)
    plt.show()

    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}

# ============================================================
# 3. LANCER LES 2 RÉGRESSIONS
# ============================================================
res1 = evaluer_modele('mental_wellness_index', df)
res2 = evaluer_modele('productivity_score', df)

# ============================================================
# 4. COMPARAISON FINALE
# ============================================================
print("\n" + "="*55)
print("  COMPARAISON FINALE")
print("="*55)
print(f"{'Métrique':<10} {'mental_wellness_index':>22} {'productivity_score':>20}")
print("-"*55)
for key in ['R2', 'MAE', 'RMSE']:
    print(f"{key:<10} {res1[key]:>22.4f} {res2[key]:>20.4f}")

print("\n✅ Terminé ! Graphiques sauvegardés.")
# ============================================================
#  Mental Health - Régression Logistique
#  Cible : mental_health_status (0, 1, 2, 3)
# ===========================================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# 3. MODÈLE
# ============================================================
model = LogisticRegression(max_iter=1000, random_state=42, multi_class='auto')
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

# ============================================================
# 4. MÉTRIQUES
# ============================================================
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.title('Confusion Matrix - Régression Logistique')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig('confusion_matrix_logistic.png', dpi=150)
plt.show()

# ============================================================
# 6. TOP 10 COEFFICIENTS PAR CLASSE
# ============================================================
classes = model.classes_
coef_df = pd.DataFrame(model.coef_, columns=X.columns,
                        index=[f'Classe {c}' for c in classes])

plt.figure(figsize=(12, 6))
for i, classe in enumerate(coef_df.index):
    top10 = coef_df.loc[classe].abs().sort_values(ascending=False).head(10)
    plt.subplot(2, 2, i+1)
    sns.barplot(x=coef_df.loc[classe][top10.index].values,
                y=top10.index, palette='coolwarm')
    plt.title(f'Top 10 Features — {classe}')
    plt.axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('coefficients_logistic.png', dpi=150)
plt.show()

print("\n✅ Terminé ! Accuracy:", f"{acc:.4f}")