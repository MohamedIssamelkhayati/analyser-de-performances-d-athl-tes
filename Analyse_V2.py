# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, skew, kurtosis

# Génération des données fictives
np.random.seed(42)
data = pd.DataFrame({
    "Âge": np.random.randint(18, 40, size=30),
    "Taille (cm)": np.random.randint(150, 200, size=30),
    "Poids (kg)": np.random.randint(50, 100, size=30),
    "Type de sport": np.random.choice(["Athlétisme", "Natation", "Cyclisme"], size=30),
    "Temps d'entraînement (h)": np.random.uniform(5, 20, size=30).round(1),
    "Score de performance": np.random.uniform(50, 100, size=30).round(1)
})

# 1. ANALYSE UNIVARIÉE
print("### ANALYSE UNIVARIÉE ###")
print("Statistiques descriptives :\n", data.describe())

# Calcul de l'asymétrie (skewness) et du kurtosis
for col in ["Âge", "Taille (cm)", "Poids (kg)", "Temps d'entraînement (h)", "Score de performance"]:
    print(f"Asymétrie ({col}) : {skew(data[col]):.2f}, Kurtosis ({col}) : {kurtosis(data[col]):.2f}")

# Histogrammes et boxplots pour les variables numériques
for col in ["Âge", "Taille (cm)", "Poids (kg)", "Temps d'entraînement (h)", "Score de performance"]:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data[col], kde=True, bins=10, color="blue")
    plt.title(f"Distribution de {col}")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col], color="orange")
    plt.title(f"Boxplot de {col}")
    plt.show()

# Analyse de la variable catégorique (Type de sport)
sns.countplot(x="Type de sport", data=data, palette="viridis")
plt.title("Répartition des types de sport")
plt.show()

# 2. ANALYSE BIVARIÉE
print("\n### ANALYSE BIVARIÉE ###")
# Matrice de corrélation
correlation_matrix = data[["Âge", "Taille (cm)", "Poids (kg)", "Temps d'entraînement (h)", "Score de performance"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# Tests statistiques bivariés
# Comparaison de deux groupes avec un t-test
athl = data[data["Type de sport"] == "Athlétisme"]["Score de performance"]
cycl = data[data["Type de sport"] == "Cyclisme"]["Score de performance"]
ttest_result = ttest_ind(athl, cycl)
print(f"T-test entre Athlétisme et Cyclisme : t-stat={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.2f}")

# Test ANOVA pour les groupes (Type de sport)
anova_result = f_oneway(
    data[data["Type de sport"] == "Athlétisme"]["Score de performance"],
    data[data["Type de sport"] == "Natation"]["Score de performance"],
    data[data["Type de sport"] == "Cyclisme"]["Score de performance"]
)
print(f"ANOVA (Type de sport et Score de performance) : F-stat={anova_result.statistic:.2f}, p-value={anova_result.pvalue:.2f}")

# Test Chi² pour les variables qualitatives
contingency_table = pd.crosstab(data["Type de sport"], pd.cut(data["Temps d'entraînement (h)"], bins=3))
chi2_result = chi2_contingency(contingency_table)
print(f"Test Chi² : Chi²={chi2_result[0]:.2f}, p-value={chi2_result[1]:.2f}")

# Scatterplots avec régressions
sns.pairplot(data, hue="Type de sport", diag_kind="kde", palette="viridis")
plt.show()

# Régression linéaire simple (par exemple : Temps d'entraînement vs Score de performance)
sns.lmplot(x="Temps d'entraînement (h)", y="Score de performance", hue="Type de sport", data=data, palette="viridis")
plt.title("Régression linéaire : Temps d'entraînement vs Score de performance")
plt.show()

# 3. ANALYSE MULTIVARIÉE (ACP)
print("\n### ANALYSE MULTIVARIÉE ###")
# Préparation des données pour l'ACP
numerical_data = data[["Âge", "Taille (cm)", "Poids (kg)", "Temps d'entraînement (h)", "Score de performance"]]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(numerical_data)

# Réalisation de l'ACP
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_
print("Variance expliquée par composante :", explained_variance)

# Cercle des corrélations
components = pd.DataFrame(pca.components_, columns=numerical_data.columns, index=["PC1", "PC2"])
print("Composantes principales :\n", components)

# Biplot ACP
pca_data = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
sns.scatterplot(x="PC1", y="PC2", hue=data["Type de sport"], data=pca_data, palette="viridis")
plt.title("Projection des individus (ACP)")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.show()

# Cercle des corrélations
plt.figure(figsize=(8, 8))
for i in range(len(components.columns)):
    plt.arrow(0, 0, components.iloc[0, i], components.iloc[1, i],
              head_width=0.05, head_length=0.05, fc="red", ec="red")
    plt.text(components.iloc[0, i] * 1.15, components.iloc[1, i] * 1.15,
             components.columns[i], color="green", ha="center", va="center")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Cercle des corrélations")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()
