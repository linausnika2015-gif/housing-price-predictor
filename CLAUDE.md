# CLAUDE.md - California Housing ML Project

## Projekt-Übersicht
Ein Machine-Learning-Projekt zur Vorhersage von Immobilienpreisen in Kalifornien basierend auf dem Scikit-Learn `fetch_california_housing` Datensatz. Das Modell nutzt einen **Random Forest Regressor** und ist als interaktive Web-App über Streamlit verfügbar.

---

## Kern-Stack
- **Sprache:** Python 3.10+
- **ML-Framework:** Scikit-Learn
- **Datenanalyse:** Pandas, NumPy
- **Visualisierung:** Matplotlib, Seaborn
- **GUI:** Streamlit
- **Hosting:** Streamlit Community Cloud (kostenlos, öffentlich erreichbar)

---

## Projektstruktur

```
ml_housing_project/
├── app.py              # Streamlit Web-App (Haupt-Einstiegspunkt)
├── model_training.py   # Modell-Training, Evaluation, Vorhersagefunktion
├── eda.py              # Explorative Datenanalyse & Visualisierungen
├── requirements.txt    # Alle Python-Abhängigkeiten (für lokale & Cloud-Installation)
├── .gitignore          # Dateien, die Git/GitHub ignorieren soll (z.B. .venv)
└── CLAUDE.md           # Diese Dokumentation
```

---

## Modell-Details

| Eigenschaft | Wert |
|---|---|
| Algorithmus | Random Forest Regressor |
| Anzahl Bäume | 100 (n_estimators=100) |
| Datensatz | California Housing (Scikit-Learn built-in) |
| Datensatz-Größe | ~20.640 Datenpunkte |
| Train/Test-Split | 80% Training / 20% Test |
| Zufalls-Seed | random_state=42 (Reproduzierbarkeit) |
| Output | Vorhersage × 100.000 = Preis in USD |

### Features (Eingabevariablen)
| Feature | Bedeutung |
|---|---|
| MedInc | Medianes Haushaltseinkommen im Viertel (in $10.000) |
| HouseAge | Alter des Hauses (in Jahren) |
| AveRooms | Durchschnittliche Zimmeranzahl pro Haushalt |
| AveBedrms | Durchschnittliche Schlafzimmeranzahl (App-intern: 1.0) |
| Population | Bevölkerung des Blocks (App-intern: 1000) |
| AveOccup | Durchschnittliche Personen pro Haushalt |
| Latitude | Breitengrad des Standorts |
| Longitude | Längengrad des Standorts |

### Ziel-Variable
- **MedHouseVal** — Medianer Hauswert (in $100.000 Einheiten)

---

## Wichtige Befehle

### 1. Umgebung & Installation (lokal)
```bash
# Virtuelle Umgebung erstellen
python -m venv .venv

# Aktivieren (Windows)
.\.venv\Scripts\activate

# Aktivieren (Mac/Linux)
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. App lokal starten
```bash
streamlit run app.py
# Browser öffnet sich automatisch auf: http://localhost:8501
```

### 3. Explorative Datenanalyse ausführen
```bash
python eda.py
```

### 4. Modell-Training einzeln ausführen
```bash
python model_training.py
```

---

## Deployment: App weltweit teilen (Streamlit Community Cloud)

### Was ist Streamlit Community Cloud?
Ein kostenloser Hosting-Service von Streamlit. Du verbindest dein GitHub-Repository und bekommst einen öffentlichen Link, den du weltweit teilen kannst (z.B. `https://deinname-housing.streamlit.app`).

### Voraussetzungen
1. Kostenloses [GitHub-Konto](https://github.com)
2. Kostenloses [Streamlit Community Cloud-Konto](https://share.streamlit.io) (Login mit GitHub)
3. Dieses Projekt muss als **öffentliches GitHub-Repository** vorliegen

### Schritt-für-Schritt Deployment

**A) GitHub Repository erstellen & Code hochladen**
1. Auf [github.com](https://github.com) einloggen → "New Repository" klicken
2. Repository-Name eingeben (z.B. `housing-price-predictor`) → "Public" auswählen → erstellen
3. Im Terminal (im Projektordner):
```bash
git init
git add .
git commit -m "Initial commit: California Housing ML App"
git branch -M main
git remote add origin https://github.com/DEIN-USERNAME/DEIN-REPO-NAME.git
git push -u origin main
```

**B) App auf Streamlit Community Cloud deployen**
1. Auf [share.streamlit.io](https://share.streamlit.io) einloggen (mit GitHub-Account)
2. "New app" klicken
3. Dein Repository auswählen
4. "Main file path" → `app.py` eintragen
5. "Deploy!" klicken → Streamlit installiert alles automatisch via `requirements.txt`
6. Nach 1-2 Minuten bekommst du deinen öffentlichen Link

### Warum braucht man GitHub dazwischen?
Streamlit Community Cloud kann nicht direkt auf deinen lokalen PC zugreifen. GitHub dient als "Zwischenlager" in der Cloud — Streamlit liest den Code von dort und führt ihn auf deren Servern aus.

---

## Bekannte Probleme & Lösungen

### App startet auf falschem Port (z.B. 8502 statt 8501)
**Ursache:** Ein alter Streamlit-Prozess läuft noch im Hintergrund und blockiert Port 8501.
**Lösung (Windows):**
```bash
taskkill //F //IM streamlit.exe
# Danach neu starten:
streamlit run app.py
```

### `streamlit` nicht gefunden nach `pip install -r requirements.txt`
**Ursache:** Virtuelle Umgebung nicht aktiviert.
**Lösung:** Erst `.venv` aktivieren, dann installieren.

---

## Entwicklungshistorie
- Initiale Erstellung: EDA, Modell-Training, Streamlit-App
- `streamlit` zu `requirements.txt` hinzugefügt (war initial vergessen)
- `.gitignore` hinzugefügt für sauberes GitHub-Repository
- Deployment-Anleitung für Streamlit Community Cloud ergänzt
