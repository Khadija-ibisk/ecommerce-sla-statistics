"""
Mini-Projet Data Science : Statistiques, Régression et Tests Paramétriques
"""

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# IMPORTATION DES BIBLIOTHÈQUES
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, expon, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 1 : CHARGEMENT ET EXPLORATION DES DONNÉES
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("="*80)
print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
print("="*80)

# Charger le dataset
df = pd.read_excel('data/mini_projet_dataset.xlsx')

# Afficher les premières lignes
print("\nPremières lignes du dataset:")
print(df.head())

# Informations générales
print("\nInformations sur le dataset:")
print(df.info())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes:")
print(df.isnull().sum())

print("\n✓ Données chargées avec succès : 500 observations et 4 variables")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 2 : ANALYSE DESCRIPTIVE
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 2 : ANALYSE DESCRIPTIVE")
print("="*80)

# Calcul des statistiques descriptives
stats_desc = df.describe()
print("\nStatistiques descriptives:")
print(stats_desc)

# Calcul détaillé pour chaque variable
print("\n--- Paramètres de Position et de Dispersion ---\n")

for col in df.columns:
    print(f"\n{col.upper()}:")
    print(f"  Moyenne (μ) : {df[col].mean():.2f}")
    print(f"  Médiane : {df[col].median():.2f}")
    print(f"  Mode : {df[col].mode()[0]:.2f}")
    print(f"  Écart-type (σ) : {df[col].std():.2f}")
    print(f"  Variance (σ²) : {df[col].var():.2f}")
    print(f"  Coefficient de variation : {(df[col].std()/df[col].mean())*100:.2f}%")
    print(f"  Étendue : {df[col].max() - df[col].min():.2f}")
    print(f"  Q1 (25%) : {df[col].quantile(0.25):.2f}")
    print(f"  Q3 (75%) : {df[col].quantile(0.75):.2f}")
    print(f"  IQR : {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
    print(f"  Asymétrie (Skewness) : {df[col].skew():.2f}")
    print(f"  Aplatissement (Kurtosis) : {df[col].kurtosis():.2f}")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 3 : VISUALISATIONS
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 3 : VISUALISATIONS")
print("="*80)

# Création d'une figure avec plusieurs sous-graphiques
fig = plt.figure(figsize=(16, 12))

# 1. Histogrammes
for i, col in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Histogramme - {col}')
    plt.xlabel(col)
    plt.ylabel('Fréquence')
    plt.grid(True, alpha=0.3)

# 2. Boxplots
for i, col in enumerate(df.columns, 1):
    plt.subplot(4, 4, i+4)
    plt.boxplot(df[col], vert=True, patch_artist=True)
    plt.title(f'Boxplot - {col}')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)

# 3. Courbes ECDF (Empirical Cumulative Distribution Function)
for i, col in enumerate(df.columns, 1):
    plt.subplot(4, 4, i+8)
    sorted_data = np.sort(df[col])
    y = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, y, marker='.', linestyle='none', markersize=3)
    plt.title(f'ECDF - {col}')
    plt.xlabel(col)
    plt.ylabel('Probabilité cumulative')
    plt.grid(True, alpha=0.3)

# 4. Q-Q plots pour vérifier la normalité
for i, col in enumerate(df.columns, 1):
    plt.subplot(4, 4, i+12)
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot - {col}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisations_completes.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques sauvegardés dans 'visualisations_completes.png'")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 4 : INTERPRÉTATION DE LA DISPERSION
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 4 : INTERPRÉTATION DE LA DISPERSION")
print("="*80)

print("\n--- VISITES ---")
cv_visites = (df['Visites'].std() / df['Visites'].mean()) * 100
print(f"Coefficient de variation: {cv_visites:.2f}%")
if cv_visites < 15:
    print("Interprétation: Dispersion FAIBLE - Les visites sont relativement homogènes")
elif cv_visites < 30:
    print("Interprétation: Dispersion MODÉRÉE - Les visites présentent une variabilité moyenne")
else:
    print("Interprétation: Dispersion ÉLEVÉE - Les visites varient considérablement d'une observation à l'autre")
print(f"L'étendue des visites est de {df['Visites'].max() - df['Visites'].min():.0f}, allant de {df['Visites'].min():.0f} à {df['Visites'].max():.0f}")

print("\n--- VENTES ---")
cv_ventes = (df['Ventes'].std() / df['Ventes'].mean()) * 100
print(f"Coefficient de variation: {cv_ventes:.2f}%")
if cv_ventes < 15:
    print("Interprétation: Dispersion FAIBLE - Les ventes sont stables")
elif cv_ventes < 30:
    print("Interprétation: Dispersion MODÉRÉE - Les ventes fluctuent modérément")
else:
    print("Interprétation: Dispersion ÉLEVÉE - Les ventes sont très variables, nécessitant une attention particulière")
print(f"L'étendue des ventes est de {df['Ventes'].max() - df['Ventes'].min():.0f}, allant de {df['Ventes'].min():.0f} à {df['Ventes'].max():.0f}")

print("\n--- TEMPS_REPONSE ---")
cv_temps = (df['Temps_Reponse'].std() / df['Temps_Reponse'].mean()) * 100
print(f"Coefficient de variation: {cv_temps:.2f}%")
if cv_temps < 15:
    print("Interprétation: Dispersion FAIBLE - Le temps de réponse est constant")
elif cv_temps < 30:
    print("Interprétation: Dispersion MODÉRÉE - Le temps de réponse présente une variabilité acceptable")
else:
    print("Interprétation: Dispersion ÉLEVÉE - Le temps de réponse est instable, problème de performance serveur")
print(f"L'étendue du temps de réponse est de {df['Temps_Reponse'].max() - df['Temps_Reponse'].min():.2f}s, allant de {df['Temps_Reponse'].min():.2f}s à {df['Temps_Reponse'].max():.2f}s")

print("\n--- CLIC ---")
cv_clic = (df['Clic'].std() / df['Clic'].mean()) * 100
print(f"Coefficient de variation: {cv_clic:.2f}%")
if cv_clic < 15:
    print("Interprétation: Dispersion FAIBLE - Le nombre de clics est homogène")
elif cv_clic < 30:
    print("Interprétation: Dispersion MODÉRÉE - Le nombre de clics varie modérément")
else:
    print("Interprétation: Dispersion ÉLEVÉE - Le nombre de clics est très variable selon les sessions")
print(f"L'étendue des clics est de {df['Clic'].max() - df['Clic'].min():.0f}, allant de {df['Clic'].min():.0f} à {df['Clic'].max():.0f}")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 5 : TEST DE NORMALITÉ
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 5 : TEST DE NORMALITÉ")
print("="*80)

print("\nLes tests de normalité permettent de vérifier si les données suivent une loi normale.")
print("H0 (hypothèse nulle): Les données suivent une loi normale")
print("H1 (hypothèse alternative): Les données ne suivent pas une loi normale")
print("Seuil de significativité: α = 0.05")
print("Décision: Si p-value < 0.05, on rejette H0 (les données ne sont pas normales)")

print("\n--- Test de Shapiro-Wilk ---")
print("(Plus approprié pour des échantillons de taille < 5000)")

for col in df.columns:
    stat, p_value = shapiro(df[col])
    print(f"\n{col}:")
    print(f"  Statistique: {stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✗ Rejet de H0 → Les données ne suivent PAS une loi normale")
    else:
        print(f"  ✓ Acceptation de H0 → Les données suivent une loi normale")

print("\n--- Test de Kolmogorov-Smirnov ---")
print("(Plus approprié pour des échantillons de grande taille)")

for col in df.columns:
    # Normaliser les données pour le test KS
    data_normalized = (df[col] - df[col].mean()) / df[col].std()
    stat, p_value = kstest(data_normalized, 'norm')
    print(f"\n{col}:")
    print(f"  Statistique: {stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✗ Rejet de H0 → Les données ne suivent PAS une loi normale")
    else:
        print(f"  ✓ Acceptation de H0 → Les données suivent une loi normale")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 6 : ESTIMATION DES PARAMÈTRES
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 6 : ESTIMATION DES PARAMÈTRES")
print("="*80)

# Estimation pour la Loi Normale (Temps_Reponse)
print("\n--- Estimation des Paramètres de la Loi Normale ---")
print("Variable: Temps_Reponse")
print("\nFormule d'estimation:")
print("  μ (moyenne) = (1/n) * Σxi")
print("  σ² (variance) = (1/n) * Σ(xi - μ)²")

mu_temps = df['Temps_Reponse'].mean()
sigma_temps = df['Temps_Reponse'].std()
variance_temps = df['Temps_Reponse'].var()

print(f"\nParamètres estimés:")
print(f"  μ (moyenne estimée) = {mu_temps:.4f} secondes")
print(f"  σ (écart-type estimé) = {sigma_temps:.4f} secondes")
print(f"  σ² (variance estimée) = {variance_temps:.4f} secondes²")
print(f"\nInterprétation: Le temps de réponse suit approximativement N({mu_temps:.2f}, {variance_temps:.2f})")

# Estimation pour la Loi Exponentielle
print("\n--- Estimation du Paramètre λ de la Loi Exponentielle ---")
print("Variable: Temps_Reponse")
print("\nFormule d'estimation:")
print("  λ = 1/μ  où μ est la moyenne observée")
print("  E(X) = 1/λ = μ")

lambda_temps = 1 / mu_temps
print(f"\nParamètre estimé:")
print(f"  λ (taux) = 1/{mu_temps:.4f} = {lambda_temps:.4f}")
print(f"  Moyenne théorique = 1/λ = {1/lambda_temps:.4f} secondes")
print(f"\nInterprétation: Si on modélise par une loi exponentielle, λ ≈ {lambda_temps:.4f}")

# Visualisation de l'ajustement
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ajustement Loi Normale
axes[0].hist(df['Temps_Reponse'], bins=30, density=True, alpha=0.7, edgecolor='black', label='Données observées')
x = np.linspace(df['Temps_Reponse'].min(), df['Temps_Reponse'].max(), 100)
axes[0].plot(x, norm.pdf(x, mu_temps, sigma_temps), 'r-', linewidth=2, label=f'Loi Normale N({mu_temps:.2f}, {sigma_temps:.2f})')
axes[0].set_xlabel('Temps de Réponse (s)')
axes[0].set_ylabel('Densité de probabilité')
axes[0].set_title('Ajustement par une Loi Normale')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Ajustement Loi Exponentielle
axes[1].hist(df['Temps_Reponse'], bins=30, density=True, alpha=0.7, edgecolor='black', label='Données observées')
axes[1].plot(x, expon.pdf(x, scale=1/lambda_temps), 'g-', linewidth=2, label=f'Loi Exponentielle λ={lambda_temps:.4f}')
axes[1].set_xlabel('Temps de Réponse (s)')
axes[1].set_ylabel('Densité de probabilité')
axes[1].set_title('Ajustement par une Loi Exponentielle')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ajustement_lois.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques d'ajustement sauvegardés dans 'ajustement_lois.png'")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 7 : RÉGRESSION LINÉAIRE (Ventes en fonction des Visites)
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 7 : RÉGRESSION LINÉAIRE")
print("="*80)

print("\nModèle: Ventes = b + a × Visites + ε")
print("où Y = Ventes (variable dépendante)")
print("    X = Visites (variable indépendante)")

# Préparation des données
X = df['Visites'].values.reshape(-1, 1)
Y = df['Ventes'].values

# Calcul du coefficient de corrélation
print("\n--- Coefficient de Corrélation ---")
print("\nFormule de Pearson:")
print("  r = Cov(X,Y) / (σX × σY)")
print("  r = Σ[(Xi - X̄)(Yi - Ȳ)] / √[Σ(Xi - X̄)² × Σ(Yi - Ȳ)²]")

correlation = np.corrcoef(df['Visites'], df['Ventes'])[0, 1]
print(f"\nCoefficient de corrélation r = {correlation:.4f}")
print(f"Coefficient de détermination R² = {correlation**2:.4f}")

# Interprétation de la force de la relation
print("\nInterprétation de la corrélation:")
r_abs = abs(correlation)
if r_abs < 0.3:
    interpretation = "FAIBLE - Relation linéaire faible entre Visites et Ventes"
elif r_abs < 0.7:
    interpretation = "MODÉRÉE - Relation linéaire modérée entre Visites et Ventes"
else:
    interpretation = "FORTE - Relation linéaire forte entre Visites et Ventes"
print(f"  |r| = {r_abs:.4f} → Relation {interpretation}")

if correlation > 0:
    print("  Corrélation POSITIVE: Plus de visites → Plus de ventes")
else:
    print("  Corrélation NÉGATIVE: Plus de visites → Moins de ventes")

# Calcul des coefficients a et b
print("\n--- Calcul des Coefficients de Régression ---")
print("\nFormules:")
print("  a (pente) = Cov(X,Y) / Var(X) = Σ[(Xi - X̄)(Yi - Ȳ)] / Σ(Xi - X̄)²")
print("  b (ordonnée à l'origine) = Ȳ - a × X̄")

# Méthode 1: Calcul manuel
X_mean = df['Visites'].mean()
Y_mean = df['Ventes'].mean()

numerator = sum((df['Visites'] - X_mean) * (df['Ventes'] - Y_mean))
denominator = sum((df['Visites'] - X_mean) ** 2)

a = numerator / denominator
b = Y_mean - a * X_mean

print(f"\nCoefficients calculés manuellement:")
print(f"  a (pente) = {a:.6f}")
print(f"  b (ordonnée) = {b:.6f}")

# Méthode 2: Utilisation de sklearn
model = LinearRegression()
model.fit(X, Y)
a_sklearn = model.coef_[0]
b_sklearn = model.intercept_

print(f"\nVérification avec sklearn:")
print(f"  a (pente) = {a_sklearn:.6f}")
print(f"  b (ordonnée) = {b_sklearn:.6f}")

# Équation de la droite
print(f"\n✓ Équation de régression:")
print(f"  Ventes = {b:.2f} + {a:.6f} × Visites")

# Prédictions
Y_pred = model.predict(X)

# Erreur moyenne de prédiction
erreur_moyenne = np.mean(np.abs(Y - Y_pred))
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)

print(f"\n--- Évaluation du Modèle ---")
print(f"  R² (coefficient de détermination) = {r2_score(Y, Y_pred):.4f}")
print(f"  Erreur moyenne absolue = {erreur_moyenne:.2f} ventes")
print(f"  RMSE (Root Mean Square Error) = {rmse:.2f} ventes")

# Interprétation métier
print(f"\n--- Interprétation Métier ---")
print(f"  • Pour 1000 visites supplémentaires, on peut s'attendre à {a*1000:.2f} ventes supplémentaires")
print(f"  • Le modèle explique {r2_score(Y, Y_pred)*100:.2f}% de la variabilité des ventes")
if r2_score(Y, Y_pred) > 0.7:
    print(f"  • Modèle de BONNE QUALITÉ pour la prédiction")
elif r2_score(Y, Y_pred) > 0.4:
    print(f"  • Modèle de QUALITÉ ACCEPTABLE, peut être amélioré")
else:
    print(f"  • Modèle de FAIBLE QUALITÉ, d'autres variables doivent être considérées")

# Visualisation de la régression
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['Visites'], df['Ventes'], alpha=0.5, label='Données observées')
plt.plot(df['Visites'], Y_pred, color='red', linewidth=2, label=f'Droite de régression: Y = {b:.2f} + {a:.6f}X')
plt.xlabel('Visites')
plt.ylabel('Ventes')
plt.title('Régression Linéaire: Ventes vs Visites')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residus = Y - Y_pred
plt.scatter(Y_pred, residus, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Graphique des Résidus')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_lineaire.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques de régression sauvegardés dans 'regression_lineaire.png'")

# Conclusion
print("\n--- CONCLUSION ---")
print(f"La relation entre les Visites et les Ventes est {interpretation}")
print(f"Le modèle linéaire: Ventes = {b:.2f} + {a:.6f} × Visites")
print(f"peut être utilisé pour estimer les ventes en fonction du nombre de visites avec un R² de {r2_score(Y, Y_pred):.4f}")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 8 : TEST DE STUDENT (Comparaison de Moyenne)
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 8 : TEST DE STUDENT")
print("="*80)

print("\nContexte: Le SLA (Service Level Agreement) de l'entreprise fixe un temps de réponse cible de 150 ms")
print("\nHypothèses:")
print("  H0 (hypothèse nulle): μ = 150 ms (le temps de réponse moyen est conforme au SLA)")
print("  H1 (hypothèse alternative): μ ≠ 150 ms (le temps de réponse moyen diffère du SLA)")
print("\nSeuil de significativité: α = 0.05")
print("Test bilatéral (on teste si différent, pas uniquement supérieur ou inférieur)")

# Conversion du SLA en secondes (cohérent avec les données)
sla_cible = 150 / 1000  # 150 ms = 0.150 s
print(f"\nSLA cible = {sla_cible} secondes")

# Calcul du test de Student
t_stat, p_value = stats.ttest_1samp(df['Temps_Reponse'], sla_cible)

print("\n--- Résultats du Test ---")
print(f"  Moyenne observée = {df['Temps_Reponse'].mean():.4f} secondes")
print(f"  Écart-type observé = {df['Temps_Reponse'].std():.4f} secondes")
print(f"  Taille de l'échantillon n = {len(df)}")
print(f"  Statistique t = {t_stat:.4f}")
print(f"  P-value = {p_value:.6f}")

# Décision
print("\n--- Décision ---")
if p_value < 0.05:
    print(f"✗ P-value ({p_value:.6f}) < α (0.05)")
    print("→ REJET de H0")
    print("→ Le temps de réponse moyen est SIGNIFICATIVEMENT DIFFÉRENT de 150 ms")
    if df['Temps_Reponse'].mean() > sla_cible:
        print(f"→ PROBLÈME: Le temps de réponse ({df['Temps_Reponse'].mean()*1000:.2f} ms) DÉPASSE le SLA ({sla_cible*1000:.0f} ms)")
        print("→ ACTION REQUISE: Optimisation des performances du serveur nécessaire")
    else:
        print(f"→ POSITIF: Le temps de réponse ({df['Temps_Reponse'].mean()*1000:.2f} ms) est MEILLEUR que le SLA ({sla_cible*1000:.0f} ms)")
else:
    print(f"✓ P-value ({p_value:.6f}) ≥ α (0.05)")
    print("→ ACCEPTATION de H0")
    print("→ Le temps de réponse moyen est CONFORME au SLA de 150 ms")
    print("→ Les performances du serveur respectent le Service Level Agreement")

# Calcul de l'intervalle de confiance à 95%
conf_interval = stats.t.interval(0.95, len(df)-1, 
                                  loc=df['Temps_Reponse'].mean(), 
                                  scale=stats.sem(df['Temps_Reponse']))
print(f"\nIntervalle de confiance à 95%:")
print(f"  [{conf_interval[0]*1000:.2f} ms, {conf_interval[1]*1000:.2f} ms]")
print(f"  Le SLA de 150 ms est {'DANS' if sla_cible >= conf_interval[0] and sla_cible <= conf_interval[1] else 'HORS DE'} cet intervalle")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Temps_Reponse'], bins=30, edgecolor='black', alpha=0.7, density=True)
plt.axvline(df['Temps_Reponse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne observée = {df["Temps_Reponse"].mean():.3f}s')
plt.axvline(sla_cible, color='green', linestyle='--', linewidth=2, label=f'SLA = {sla_cible:.3f}s')
plt.xlabel('Temps de Réponse (s)')
plt.ylabel('Densité')
plt.title('Distribution du Temps de Réponse vs SLA')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['Temps_Reponse'], vert=True, patch_artist=True)
plt.axhline(sla_cible, color='green', linestyle='--', linewidth=2, label=f'SLA = {sla_cible:.3f}s')
plt.axhline(df['Temps_Reponse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne = {df["Temps_Reponse"].mean():.3f}s')
plt.ylabel('Temps de Réponse (s)')
plt.title('Boxplot du Temps de Réponse')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_student.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques du test de Student sauvegardés dans 'test_student.png'")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# ÉTAPE 9 : TEST DU KHI-DEUX (Indépendance)
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("ÉTAPE 9 : TEST DU KHI-DEUX (χ²)")
print("="*80)

print("\nObjectif: Tester l'indépendance entre le nombre de Clics et le niveau de Visites")
print("\nÉtape 1: Catégorisation de la variable 'Visites'")

# Création de catégories pour Visites
# On utilise les quartiles pour créer 4 catégories
df['Visites_Cat'] = pd.cut(df['Visites'], 
                            bins=[df['Visites'].min()-1, 
                                  df['Visites'].quantile(0.25),
                                  df['Visites'].quantile(0.50),
                                  df['Visites'].quantile(0.75),
                                  df['Visites'].max()+1],
                            labels=['Très Faible', 'Faible', 'Élevé', 'Très Élevé'])

print(f"  Catégories créées:")
print(f"    - Très Faible: [{df['Visites'].min():.0f}, {df['Visites'].quantile(0.25):.0f}]")
print(f"    - Faible: [{df['Visites'].quantile(0.25):.0f}, {df['Visites'].quantile(0.50):.0f}]")
print(f"    - Élevé: [{df['Visites'].quantile(0.50):.0f}, {df['Visites'].quantile(0.75):.0f}]")
print(f"    - Très Élevé: [{df['Visites'].quantile(0.75):.0f}, {df['Visites'].max():.0f}]")

# Catégorisation des Clics
df['Clic_Cat'] = pd.cut(df['Clic'], 
                        bins=[df['Clic'].min()-1,
                              df['Clic'].quantile(0.33),
                              df['Clic'].quantile(0.67),
                              df['Clic'].max()+1],
                        labels=['Faible', 'Moyen', 'Élevé'])

print(f"\n  Catégories de Clics:")
print(f"    - Faible: [{df['Clic'].min():.0f}, {df['Clic'].quantile(0.33):.0f}]")
print(f"    - Moyen: [{df['Clic'].quantile(0.33):.0f}, {df['Clic'].quantile(0.67):.0f}]")
print(f"    - Élevé: [{df['Clic'].quantile(0.67):.0f}, {df['Clic'].max():.0f}]")

# Création du tableau de contingence
contingency_table = pd.crosstab(df['Clic_Cat'], df['Visites_Cat'])

print("\n--- Tableau de Contingence ---")
print(contingency_table)

# Test du Khi-deux
print("\n--- Test d'Indépendance du Khi-deux ---")
print("\nHypothèses:")
print("  H0: La variable 'Clic' est INDÉPENDANTE de la catégorie de 'Visites'")
print("  H1: Il existe une DÉPENDANCE entre 'Clic' et la catégorie de 'Visites'")
print("\nSeuil de significativité: α = 0.05")

chi2_stat, p_value_chi2, dof, expected_freq = stats.chi2_contingency(contingency_table)

print("\n--- Résultats du Test ---")
print(f"  Statistique χ² = {chi2_stat:.4f}")
print(f"  Degrés de liberté = {dof}")
print(f"  P-value = {p_value_chi2:.6f}")

# Affichage des fréquences attendues
print("\n--- Fréquences Attendues (sous H0) ---")
print(pd.DataFrame(expected_freq, 
                   index=contingency_table.index, 
                   columns=contingency_table.columns).round(2))

# Décision
print("\n--- Décision ---")
if p_value_chi2 < 0.05:
    print(f"✗ P-value ({p_value_chi2:.6f}) < α (0.05)")
    print("→ REJET de H0")
    print("→ Il existe une DÉPENDANCE SIGNIFICATIVE entre le nombre de Clics et le niveau de Visites")
    print("\nInterprétation Métier:")
    print("  • Le comportement de clic des utilisateurs est lié au niveau de trafic")
    print("  • Les campagnes marketing doivent tenir compte de cette relation")
    print("  • L'optimisation du taux de clic devrait être adaptée selon le volume de visites")
else:
    print(f"✓ P-value ({p_value_chi2:.6f}) ≥ α (0.05)")
    print("→ ACCEPTATION de H0")
    print("→ Les variables 'Clic' et 'Visites' sont INDÉPENDANTES")
    print("\nInterprétation Métier:")
    print("  • Le nombre de clics n'est pas influencé par le niveau de visites")
    print("  • Les stratégies de clic peuvent être uniformes quel que soit le trafic")

# Calcul du V de Cramer (mesure de l'intensité de la dépendance)
n = contingency_table.sum().sum()
min_dim = min(len(contingency_table.index), len(contingency_table.columns)) - 1
cramers_v = np.sqrt(chi2_stat / (n * min_dim))
print(f"\nV de Cramer = {cramers_v:.4f}")
if cramers_v < 0.1:
    print("  → Association TRÈS FAIBLE")
elif cramers_v < 0.3:
    print("  → Association FAIBLE")
elif cramers_v < 0.5:
    print("  → Association MODÉRÉE")
else:
    print("  → Association FORTE")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap du tableau de contingence
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Fréquence'})
axes[0].set_title('Tableau de Contingence: Clics vs Visites')
axes[0].set_xlabel('Catégorie de Visites')
axes[0].set_ylabel('Catégorie de Clics')

# Graphique en barres groupées
contingency_table.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('Distribution des Clics selon les Visites')
axes[1].set_xlabel('Catégorie de Clics')
axes[1].set_ylabel('Fréquence')
axes[1].legend(title='Visites', bbox_to_anchor=(1.05, 1))
axes[1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('test_khi2.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphiques du test du Khi-deux sauvegardés dans 'test_khi2.png'")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
# SYNTHÈSE FINALE
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*
print("\n" + "="*80)
print("SYNTHÈSE FINALE ET RECOMMANDATIONS MÉTIER")
print("="*80)

print("\n📊 STATISTIQUES DESCRIPTIVES")
print(f"  • Dataset de {len(df)} observations avec 4 variables")
print(f"  • Visites: moyenne = {df['Visites'].mean():.0f}, CV = {cv_visites:.1f}%")
print(f"  • Ventes: moyenne = {df['Ventes'].mean():.0f}, CV = {cv_ventes:.1f}%")
print(f"  • Temps_Reponse: moyenne = {df['Temps_Reponse'].mean():.3f}s, CV = {cv_temps:.1f}%")
print(f"  • Clics: moyenne = {df['Clic'].mean():.0f}, CV = {cv_clic:.1f}%")

print("\n📈 MODÈLE DE RÉGRESSION")
print(f"  • Équation: Ventes = {b:.2f} + {a:.6f} × Visites")
print(f"  • Corrélation r = {correlation:.4f} ({interpretation})")
print(f"  • Qualité du modèle R² = {r2_score(Y, Y_pred):.4f} ({r2_score(Y, Y_pred)*100:.1f}% de variance expliquée)")

print("\n⚡ PERFORMANCE SERVEUR (Test de Student)")
if p_value < 0.05:
    if df['Temps_Reponse'].mean() > sla_cible:
        print(f"  ⚠️ ALERTE: Temps de réponse ({df['Temps_Reponse'].mean()*1000:.0f} ms) SUPÉRIEUR au SLA (150 ms)")
        print(f"  • Écart significatif détecté (p = {p_value:.6f})")
        print(f"  • Action requise: Optimisation urgente des performances")
    else:
        print(f"  ✅ EXCELLENT: Temps de réponse ({df['Temps_Reponse'].mean()*1000:.0f} ms) INFÉRIEUR au SLA (150 ms)")
else:
    print(f"  ✅ CONFORME: Temps de réponse respecte le SLA de 150 ms")

print("\n🔗 RELATION CLICS-VISITES (Test du Khi-deux)")
if p_value_chi2 < 0.05:
    print(f"  • Dépendance SIGNIFICATIVE détectée (p = {p_value_chi2:.6f})")
    print(f"  • Intensité de l'association: V de Cramer = {cramers_v:.4f}")
    print(f"  • Recommandation: Adapter les stratégies selon le niveau de trafic")
else:
    print(f"  • Variables INDÉPENDANTES (p = {p_value_chi2:.6f})")
    print(f"  • Recommandation: Stratégies de clic uniformes possibles")

print("\n💡 RECOMMANDATIONS STRATÉGIQUES")
print("  1. Utiliser le modèle de régression pour prévoir les ventes")
print("  2. Surveiller le temps de réponse pour maintenir la performance")
print("  3. Analyser l'impact des campagnes marketing sur les visites et clics")
print("  4. Optimiser le taux de conversion (Ventes/Visites)")
print("  5. Mettre en place des alertes pour détecter les anomalies")

print("\n" + "="*80)
print("FIN DE L'ANALYSE - PROJET COMPLÉTÉ AVEC SUCCÈS ✓")
print("="*80)

# Sauvegarde des résultats dans un fichier texte
with open('resultats_analyse.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RÉSULTATS DE L'ANALYSE STATISTIQUE\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. STATISTIQUES DESCRIPTIVES\n")
    f.write(f"   Visites: μ = {df['Visites'].mean():.2f}, σ = {df['Visites'].std():.2f}\n")
    f.write(f"   Ventes: μ = {df['Ventes'].mean():.2f}, σ = {df['Ventes'].std():.2f}\n")
    f.write(f"   Temps_Reponse: μ = {df['Temps_Reponse'].mean():.4f}s, σ = {df['Temps_Reponse'].std():.4f}s\n")
    f.write(f"   Clics: μ = {df['Clic'].mean():.2f}, σ = {df['Clic'].std():.2f}\n\n")
    
    f.write("2. RÉGRESSION LINÉAIRE\n")
    f.write(f"   Équation: Ventes = {b:.2f} + {a:.6f} × Visites\n")
    f.write(f"   Corrélation: r = {correlation:.4f}\n")
    f.write(f"   R² = {r2_score(Y, Y_pred):.4f}\n\n")
    
    f.write("3. TEST DE STUDENT\n")
    f.write(f"   Statistique t = {t_stat:.4f}\n")
    f.write(f"   P-value = {p_value:.6f}\n")
    f.write(f"   Décision: {'Rejet de H0' if p_value < 0.05 else 'Acceptation de H0'}\n\n")
    
    f.write("4. TEST DU KHI-DEUX\n")
    f.write(f"   Statistique χ² = {chi2_stat:.4f}\n")
    f.write(f"   P-value = {p_value_chi2:.6f}\n")
    f.write(f"   V de Cramer = {cramers_v:.4f}\n")
    f.write(f"   Décision: {'Rejet de H0' if p_value_chi2 < 0.05 else 'Acceptation de H0'}\n")

print("\n✓ Résultats sauvegardés dans 'resultats_analyse.txt'")