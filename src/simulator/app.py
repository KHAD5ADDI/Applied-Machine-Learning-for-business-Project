# pip install streamlit joblib scikit-learn pandas numpy folium streamlit-folium

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Chicago Crime Predictor",
    page_icon="🚓",
    layout="wide"
)

st.markdown(
    """
    <style>
    .big-box {
        padding: 18px 20px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .context-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Aller vers",
    ["Prédiction en temps réel", "Carte des hotspots", "Comparaison inter-villes"],
)

MODEL_PATH = "rf_chicago_v2.pkl"
KMEANS_PATH = "kmeans_chicago.pkl"
SCALER_PATH = "scaler_chicago.pkl"

CLASS_LABELS = {
    0: "VIOLENT",
    1: "PROPERTY",
    2: "DRUG",
    3: "PUBLIC ORDER",
}

CLASS_COLORS = {
    "VIOLENT": "#E74C3C",
    "PROPERTY": "#E67E22",
    "DRUG": "#9B59B6",
    "PUBLIC ORDER": "#3498DB",
}

LOCATION_GROUPS = {
    "Street": 0,
    "Residence": 1,
    "Commercial": 2,
    "Transport": 3,
    "Public": 4,
    "Other": 5,
}

DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
MONTHS = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Aout", "Septembre", "Octobre", "Novembre", "Decembre"]


def load_model(path):
    if not os.path.exists(path):
        st.error("Modele introuvable — placez rf_chicago_v2.pkl dans le meme dossier que app.py")
        return None
    return joblib.load(path)


def load_kmeans(path):
    if not os.path.exists(path):
        st.error("Modele K-Means introuvable — placez kmeans_chicago.pkl dans le meme dossier que app.py")
        return None
    return joblib.load(path)


def load_scaler(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def compute_cluster(kmeans_model, scaler_model, latitude, longitude):
    coords = np.array([[latitude, longitude]])
    if scaler_model is not None:
        coords = scaler_model.transform(coords)
    return int(kmeans_model.predict(coords)[0])


def build_feature_row(lat, lon, hour, day_of_week, month, is_weekend, domestic, cluster, location_group):
    return pd.DataFrame(
        [{
            "latitude": lat,
            "longitude": lon,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "domestic": domestic,
            "cluster": cluster,
            "location_group": location_group,
        }]
    )


def class_label_from_pred(pred):
    return CLASS_LABELS.get(int(pred), "UNKNOWN")


def render_prediction_box(label):
    color = CLASS_COLORS.get(label, "#7F8C8D")
    st.markdown(
        f'<div class="big-box" style="background:{color}">{label}</div>',
        unsafe_allow_html=True,
    )


if page == "Prédiction en temps réel":
    st.title("🚓 Prédiction en temps réel")
    st.write("Simulez un crime à Chicago et obtenez une prédiction en direct.")

    st.sidebar.subheader("Parametres")
    hour = st.sidebar.slider("Heure", 0, 23, 12)
    day = st.sidebar.selectbox("Jour de la semaine", DAYS, index=2)
    month = st.sidebar.selectbox("Mois", MONTHS, index=3)
    location = st.sidebar.selectbox(
        "Type de lieu",
        ["Street", "Residence", "Commercial", "Transport", "Public", "Other"],
        index=0,
    )
    domestic = st.sidebar.checkbox("Crime domestique", value=False)
    latitude = st.sidebar.number_input("Latitude", value=41.8781, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=-87.6298, format="%.4f")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametres calcules")
        day_of_week = DAYS.index(day)
        month_idx = MONTHS.index(month) + 1
        is_weekend = 1 if day_of_week >= 5 else 0
        location_group = LOCATION_GROUPS[location]

        st.write(f"Jour (0-6): {day_of_week}")
        st.write(f"Mois (1-12): {month_idx}")
        st.write(f"Weekend: {is_weekend}")
        st.write(f"Location group: {location_group}")

    with col2:
        st.subheader("Action")
        predict_btn = st.button("🔍 Predire le type de crime")

    if predict_btn:
        rf_model = load_model(MODEL_PATH)
        kmeans_model = load_kmeans(KMEANS_PATH)
        scaler_model = load_scaler(SCALER_PATH)

        if rf_model is not None and kmeans_model is not None:
            cluster = compute_cluster(kmeans_model, scaler_model, latitude, longitude)

            X = build_feature_row(
                latitude,
                longitude,
                hour,
                day_of_week,
                month_idx,
                is_weekend,
                1 if domestic else 0,
                cluster,
                location_group,
            )

            # Aligner l ordre des colonnes si le modele fournit feature_names_in_
            if hasattr(rf_model, "feature_names_in_"):
                X = X[rf_model.feature_names_in_]

            pred = rf_model.predict(X)[0]
            proba = rf_model.predict_proba(X)[0]
            classes = rf_model.classes_

            label = class_label_from_pred(pred)
            render_prediction_box(label)

            proba_df = pd.DataFrame({
                "Classe": [class_label_from_pred(c) for c in classes],
                "Probabilite": proba,
            }).sort_values("Probabilite", ascending=True)

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.barh(proba_df["Classe"], proba_df["Probabilite"], color="#5DADE2")
            ax.set_xlabel("Probabilite")
            ax.set_xlim(0, 1)
            ax.set_title("Probabilites par classe")
            for i, v in enumerate(proba_df["Probabilite"]):
                ax.text(v + 0.01, i, f"{v:.2f}", va="center")
            st.pyplot(fig)

            st.markdown(
                "<div class='context-box'>⚠️ Contexte : Cette prediction represente la classe la plus probable selon les patterns historiques observes a Chicago.</div>",
                unsafe_allow_html=True,
            )

elif page == "Carte des hotspots":
    st.title("🗺️ Carte des hotspots Chicago")
    st.write("Zones geographiques identifiees par K-Means.")

    # Centroides et top crimes (hardcodes)
    centroids = [
        {"cluster": 0, "lat": 41.8781, "lon": -87.6298, "top": "PROPERTY"},
        {"cluster": 1, "lat": 41.8917, "lon": -87.6077, "top": "VIOLENT"},
        {"cluster": 2, "lat": 41.8500, "lon": -87.6500, "top": "DRUG"},
        {"cluster": 3, "lat": 41.9400, "lon": -87.7000, "top": "PUBLIC ORDER"},
        {"cluster": 4, "lat": 41.7700, "lon": -87.6200, "top": "PROPERTY"},
    ]

    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10, tiles="cartodbpositron")

    color_map = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#C7CEEA"]
    for c in centroids:
        folium.CircleMarker(
            location=[c["lat"], c["lon"]],
            radius=12,
            color=color_map[c["cluster"] % len(color_map)],
            fill=True,
            fill_color=color_map[c["cluster"] % len(color_map)],
            fill_opacity=0.8,
            tooltip=f"Cluster {c['cluster']} — Top crime: {c['top']}",
        ).add_to(m)

    st.subheader("Zones geographiques identifiees par K-Means")
    st_folium(m, width=1000, height=500)

    st.markdown("**Legende (Top crime par cluster)**")
    for c in centroids:
        st.write(f"Cluster {c['cluster']} : {c['top']}")

elif page == "Comparaison inter-villes":
    st.title("📊 Comparaison inter-villes")
    st.write("Comparaison des performances entre Chicago et Los Angeles.")

    data = [
        {
            "Modele": "RF Chicago v2",
            "Donnees test": "Chicago",
            "Accuracy": 0.5131,
            "F1 Violent": 0.523,
            "F1 Property": 0.483,
            "F1 Drug": 0.659,
            "F1 Public Order": 0.371,
        },
        {
            "Modele": "RF Chicago v2",
            "Donnees test": "LA sans retrain",
            "Accuracy": 0.2678,
            "F1 Violent": 0.004,
            "F1 Property": 0.529,
            "F1 Drug": 0.000,
            "F1 Public Order": 0.003,
        },
        {
            "Modele": "RF LA natif",
            "Donnees test": "LA",
            "Accuracy": 0.6103,
            "F1 Violent": 0.631,
            "F1 Property": 0.618,
            "F1 Drug": 0.000,
            "F1 Public Order": 0.000,
        },
    ]

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    metrics = ["Accuracy", "F1 Violent", "F1 Property", "F1 Drug", "F1 Public Order"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, row in df.iterrows():
        ax.bar(x + i * width, [row[m] for m in metrics], width, label=f"{row['Modele']} - {row['Donnees test']}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des metriques par modele")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("**Conclusion :** Les modeles de criminalite sont city-specific. Seule la categorie PROPERTY se transfere entre villes.")
