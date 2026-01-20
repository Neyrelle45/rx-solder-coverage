import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine

st.set_page_config(page_title="Station Analyse RX", layout="wide")

# --- Initialisation Session ---
if 'history' not in st.session_state: st.session_state.history = []

st.sidebar.title("🛠️ Outils & Réglages")

# 2) CURSEUR DE CONTRASTE
contrast_val = st.sidebar.slider("Ajustement Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)

model_file = st.sidebar.file_uploader("1. Charger le modèle", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    
    col_u, col_m = st.columns(2)
    with col_u: rx_upload = st.file_uploader("Image RX", type=["png", "jpg"])
    with col_m: mask_upload = st.file_uploader("Masque", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # Utilisation du curseur de contraste
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # ... (Logique d'alignement zone_adj identique) ...

        with st.spinner("Analyse IA..."):
            feats = engine.compute_features(img_gray)
            # Calcul de la prédiction
            pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
            # 1) CALCUL DE LA CONFIANCE
            conf_score = engine.compute_confidence(clf, feats.reshape(-1, feats.shape[-1]))

        # --- AFFICHAGE DE LA CONFIANCE ---
        st.subheader("🛡️ État de l'analyse")
        # Barre de progression colorée selon la confiance
        color = "green" if conf_score > 0.85 else "orange" if conf_score > 0.70 else "red"
        st.markdown(f"Indice de confiance du modèle : <span style='color:{color}; font-weight:bold;'>{conf_score*100:.1f}%</span>", unsafe_allow_html=True)
        st.progress(conf_score)

        if conf_score < 0.75:
            st.warning("⚠️ Confiance faible : l'image est peut-être trop différente de l'entraînement. Ajustez le contraste ou ré-alignez le masque.")

        # ... (Reste du code de visualisation et historique) ...
        # [Utilisez pred_bin et voids_sorted comme avant]
