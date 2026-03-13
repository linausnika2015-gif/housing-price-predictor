import pandas as pd
from sklearn.datasets import fetch_california_housing

# Daten laden
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target # Das ist unser Zielwert (Hauspreis)

# --- PHASE 1: Sichtung ---
print("--- Die ersten 5 Zeilen der Tabelle ---")
print(df.head())

print("\n--- Datentypen und fehlende Werte ---")
print(df.info())

# --- PHASE 2: Statistik ---
print("\n--- Statistische Zusammenfassung ---")
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# --- PHASE 3: Visualisierung der Verteilung ---
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title('Verteilung der Hauspreise (in 100.000 $)')
plt.show()

# --- PHASE 4: Korrelationsanalyse ---

# 1. Korrelationsmatrix berechnen
corr_matrix = df.corr()

# 2. Heatmap erstellen
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
plt.title("Korrelations-Heatmap der Merkmale")
plt.show()

# 3. Das wichtigste Merkmal im Detail (Einkommen vs. Preis)
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='MedInc', y='MedHouseVal', 
            scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title("Einfluss des Einkommens auf den Hauspreis")
plt.show()