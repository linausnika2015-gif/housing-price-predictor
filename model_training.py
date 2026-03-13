from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pandas as pd

# 1. Daten laden (wie zuvor)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# 2. X (Features) und y (Target) definieren
X = df.drop('MedHouseVal', axis=1) # Alles außer dem Preis
y = df['MedHouseVal']              # Nur der Preis

# 3. Train-Test-Split (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training-Daten: {X_train.shape[0]} Zeilen")
print(f"Test-Daten: {X_test.shape[0]} Zeilen")

#================ Code für das training ====================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Das Modell erstellen
# n_estimators=100 bedeutet, wir nutzen 100 kleine "Entscheidungsbäume"
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Das Training (Die "fit" Methode)
print("Das Modell lernt jetzt... bitte warten...")
model.fit(X_train, y_train)

# 3. Eine Vorhersage machen
# Wir lassen die KI jetzt die Preise für die TEST-Daten schätzen
predictions = model.predict(X_test)

# 4. Den Fehler messen
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"Fertig! Der durchschnittliche Fehler (RMSE) liegt bei: {rmse:.4f}")


#=============== Visualisierung =========================================
import matplotlib.pyplot as plt

# Wir erstellen ein Diagramm
plt.figure(figsize=(10, 6))

# Die blauen Punkte zeigen: Echter Preis (x) vs. Vorhersage (y)
plt.scatter(y_test, predictions, alpha=0.2, color='blue')

# Die rote Linie zeigt: Wo die Punkte liegen müssten, wenn alles perfekt wäre
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)

plt.xlabel('Tatsächliche Preise (in 100k $)')
plt.ylabel('Vorhergesagte Preise (in 100k $)')
plt.title('Analyse der Vorhersagegenauigkeit')
plt.grid(True)
plt.show()

#=============== wichtige Elemente ============================

# Welche Merkmale waren am wichtigsten?
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n--- Ranking der wichtigsten Merkmale ---")
print(feature_importance_df)


import numpy as np

def sage_preis_voraus(einkommen, alter, zimmer, schlafzimmer, bevölkerung, belegung, breitengrad, längengrad):
    # 1. Die Eingaben in das richtige Format bringen (eine Liste in einer Liste)
    neue_daten = np.array([[einkommen, alter, zimmer, schlafzimmer, bevölkerung, belegung, breitengrad, längengrad]])
    
    # 2. Die Vorhersage berechnen
    vorhersage = model.predict(neue_daten)
    
    # 3. Ergebnis schöner anzeigen (da die Daten in 100.000er Schritten sind)
    preis_in_dollar = vorhersage[0] * 100000
    print(f"\n🏠 Geschätzter Hauswert: ${preis_in_dollar:,.2f}")

# --- JETZT TESTEN WIR ES ---
# Beispiel: Ein Viertel mit gutem Einkommen (5.0), 20 Jahre altes Haus, in Los Angeles
print("Vorhersage für ein fiktives Haus in LA:")
sage_preis_voraus(5.0, 20.0, 5.0, 1.0, 1000.0, 3.0, 34.05, -118.24)