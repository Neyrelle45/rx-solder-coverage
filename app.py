import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os
from datetime import datetime

st.set_page_config(page_title="Station Analyse RX - ALL Circuits", layout="wide")

# --- Initialisation de l'historique ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'gallery' not in st.session_state:
    st.session_state.gallery = {}

# --- Sidebar ---
st.sidebar.title("🔍 Configuration")
st.sidebar.markdown(f"**Dossier Travail :**\n`C:\\Users\\kiefferp\\OneDrive - ALL Circuits\\Bureau\\WIP\\_IA\\Analyse RX`")

model_file = st.sidebar.file_uploader("1. Charger le modèle (.joblib)", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    
    st.header("🎯 Analyse et Métriques")
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg", "bmp"])
    with col_m:
        mask_upload = st.file_uploader("Masque d'inspection", type=["png", "jpg", "jpeg"])

    if rx_upload and mask_upload:
        # Lecture avec le nouveau patch CLAHE
        img_gray = engine.load_gray(rx_upload)
        H, W = img_gray.shape

        # Réglages alignement
        st.sidebar.subheader("🕹️ Alignement")
        tx = st.sidebar.number_input("X (px)", value=0.0, step=1.0)
        ty = st.sidebar.number_input("Y (px)", value=0.0, step=1.0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        # Préparation du masque
        with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())
        zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")
        if zone_base.shape != (H, W):
            zone_base = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)

        # Matrice d'alignement
        ys, xs = np.where(zone_base > 0)
        cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_base, M, (W, H), flags=cv2.INTER_NEAREST)

        # Analyse IA
        with st.spinner("Analyse..."):
            feats = engine.compute_features(img_gray)
            pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
            pred_bin = (pred == 1).astype(np.uint8) * 255 

        # Voids (Top 5 uniquement)
        solder_map = np.zeros((H,W), dtype=np.uint8)
        solder_map[(zone_adj > 0) & (pred_bin > 0)] = 255
        contours, hierarchy = cv2.findContours(solder_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        voids_data = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    area = cv2.contourArea(contours[i])
                    if area > 2: voids_data.append({"area": area, "cnt": contours[i]})
        voids_sorted = sorted(voids_data, key=lambda x: x['area'], reverse=True)[:5]

        # Visuel
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[zone_adj > 0] = [255, 0, 0] # Rouge
        overlay[(zone_adj > 0) & (pred_bin > 0)] = [255, 255, 0] # Jaune
        for v in voids_sorted:
            cv2.drawContours(overlay, [v['cnt']], -1, [0, 255, 255], 2) # Cyan épaisseur 2

        # Métriques
        total_px = int(np.sum(zone_adj > 0))
        solder_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
        missing_pct = round(((total_px - solder_px) / total_px * 100), 2)

        st.image(overlay, caption=f"Analyse : {rx_upload.name}", use_container_width=True)

        if st.button("📥 Enregistrer dans l'historique"):
            row = {"Image": rx_upload.name, "Manque Global %": missing_pct}
            for i in range(5):
                val = round((voids_sorted[i]['area']/total_px*100), 2) if i < len(voids_sorted) else 0.0
                row[f"Void {i+1} %"] = val
            
            st.session_state.history.append(row)
            st.session_state.gallery[rx_upload.name] = overlay
            st.success("Données archivées en bas de page.")

# --- SECTION BAS DE PAGE ---
st.divider()
if st.session_state.history:
    st.subheader("📊 Tableau Récapitulatif Cumulé")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)
    
    # Export CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📊 Télécharger le rapport (CSV)", csv, "rapport_analyse_rx.csv", "text/csv")

    st.divider()
    st.subheader("🖼️ Galerie des images inspectées")
    cols = st.columns(5)
    for idx, (name, img) in enumerate(st.session_state.gallery.items()):
        with cols[idx % 5]:
            st.image(img, caption=name, use_container_width=True)
            if st.button("Réouvrir", key=f"re_{idx}"):
                st.session_state['active_zoom'] = img

if 'active_zoom' in st.session_state:
    st.divider()
    st.subheader("🔍 Vue détaillée de l'archive")
    st.image(st.session_state['active_zoom'], use_container_width=True)
