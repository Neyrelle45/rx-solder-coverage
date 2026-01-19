import streamlit as st
import cv2
import numpy as np
import os
import joblib
from PIL import Image
import analyse_rx_soudure as engine

st.set_page_config(page_title="Analyse RX Soudure", layout="wide")

st.title("🔍 Analyse de couverture de soudure RX")

# --- Barre latérale : Configuration ---
st.sidebar.header("Configuration")
model_file = st.sidebar.file_uploader("Charger le modèle (.joblib)", type=["joblib"])
mode = st.sidebar.radio("Mode de masque", ["Masques individuels", "Masque unique"])

# --- Chargement des données ---
col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("Image RX (Composant)", type=["png", "jpg", "jpeg", "bmp"])
with col2:
    mask_file = st.file_uploader("Masque d'inspection (Vert/Noir)", type=["png", "jpg", "jpeg"])

if img_file and mask_file and model_file:
    # Sauvegarde temporaire pour compatibilité avec ton script
    with open("temp_img.png", "wb") as f: f.write(img_file.getbuffer())
    with open("temp_mask.png", "wb") as f: f.write(mask_file.getbuffer())
    
    # Charger le modèle
    clf = joblib.load(model_file)
    
    # 1. Prédire la soudure (moteur)
    img_gray = engine.load_gray("temp_img.png")
    features = engine.compute_features(img_gray)
    H, W = img_gray.shape
    pred = clf.predict(features.reshape(-1, features.shape[-1])).reshape(H, W).astype(np.uint8)
    pred_bin = (pred == 1).astype(np.uint8) * 255

    # 2. Gérer le masque
    zone_u8, _, _ = engine.compute_zone_and_holes("temp_mask.png")
    if zone_u8.shape != (H, W):
        zone_u8 = cv2.resize(zone_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    # --- Interface d'alignement ---
    st.subheader("Alignement et Résultats")
    
    if mode == "Masque unique":
        st.info("Utilisez les curseurs pour aligner le masque sur le composant.")
        c_alt1, c_alt2, c_alt3, c_alt4 = st.columns(4)
        tx = c_alt1.slider("Translation X", -100.0, 100.0, 0.0)
        ty = c_alt2.slider("Translation Y", -100.0, 100.0, 0.0)
        rot = c_alt3.slider("Rotation", -180.0, 180.0, 0.0)
        scale = c_alt4.slider("Échelle", 0.5, 1.5, 1.0)
        
        # Calcul du centre pour la rotation
        ys, xs = np.where(zone_u8 > 0)
        cx, cy = (float(xs.mean()), float(ys.mean())) if ys.size > 0 else (W/2, H/2)
        
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_u8*255, M, (W, H), flags=cv2.INTER_NEAREST)
        zone_adj = (zone_adj > 0).astype(np.uint8)
    else:
        # Mode auto (individuel) simplifié pour la démo
        zone_adj = (zone_u8 > 0).astype(np.uint8)

    # --- Calcul des métriques ---
    zone_area = int(zone_adj.sum())
    solder_px = int(((pred_bin > 0) & (zone_adj > 0)).sum())
    missing_ratio = (1 - (solder_px / zone_area)) if zone_area > 0 else 0
    
    # --- Affichage de l'Overlay ---
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay[ (zone_adj > 0) & (pred_bin > 0) ] = [0, 255, 255] # Jaune
    overlay[ (zone_adj > 0) & (pred_bin == 0) ] = [0, 0, 255]   # Rouge
    
    st.image(overlay, caption=f"Analyse : {missing_ratio*100:.2f}% de manque", use_container_width=True)
    
    st.metric("Taux de manque (Soudure)", f"{missing_ratio*100:.2f} %")
    if missing_ratio > 0.2:
        st.error("⚠️ Alerte : Taux de vide trop élevé")
    else:
        st.success("✅ Conformité OK")
