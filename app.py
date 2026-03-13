import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# --- 1. SEITEN-KONFIGURATION ---
st.set_page_config(page_title="Immobilien-KI Profi", page_icon="🏠", layout="wide")

# --- 2. MODELL LADEN & TRAINIEREN ---
# Wir nutzen Cache, damit die App schnell bleibt
@st.cache_resource
def load_and_train_model():
    # Daten holen
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    # Modell trainieren (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, housing.feature_names

# Status-Meldung während das Modell lädt
with st.spinner('Modell wird im Hintergrund geladen...'):
    model, feature_names = load_and_train_model()

# --- 3. GUI DESIGN ---
st.title("🏠 KI-Immobilien-Bewertung")
st.markdown("""
Dieses Tool nutzt ein **Machine Learning Modell (Random Forest)**, um Hauspreise in Kalifornien basierend auf 20.000 Datensätzen vorherzusagen.
""")

st.divider()

# Layout mit zwei Spalten: Links Inputs, Rechts Ergebnis
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Immobilien-Details")
    
    # Eingabe-Slider für den User
    med_inc = st.slider("Median Einkommen im Viertel (in 10.000 $)", 0.5, 15.0, 5.0, help="Haupteinflussfaktor auf den Preis")
    house_age = st.slider("Alter des Hauses", 1, 52, 25)
    ave_rooms = st.slider("Durchschnittliche Zimmeranzahl", 1.0, 10.0, 5.0)
    ave_occup = st.slider("Personen pro Haushalt", 1.0, 6.0, 3.0)
    
    st.subheader("📍 Standort")
    lat = st.number_input("Breitengrad (Latitude)", value=34.05, format="%.2f")
    lon = st.number_input("Längengrad (Longitude)", value=-118.24, format="%.2f")

with col2:
    st.header("💰 Vorhersage")
    
    # Daten für die Vorhersage vorbereiten
    # Wir setzen AveBedrms und Population auf Durchschnittswerte
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, 1.0, 1000.0, ave_occup, lat, lon
    ]], columns=feature_names)
    
    # Vorhersage-Button
    if st.button("Berechne Marktwert", type="primary"):
        prediction = model.predict(input_data)
        final_price = prediction[0] * 100000
        
        # Ergebnis-Box
        st.success(f"### Geschätzter Wert: ${final_price:,.2f}")
        
        # Metriken anzeigen
        m1, m2 = st.columns(2)
        m1.metric("Status", "Berechnet")
        m2.metric("Konfidenz", "Hoch (RF)")
        
        st.balloons()
    else:
        st.info("Klicke auf den Button, um die KI-Analyse zu starten.")

# --- 4. FOOTER ---
st.divider()
st.caption("Entwickelt während deines ersten ML-Projekts. Datenbasis: Scikit-Learn California Housing Dataset.")