import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os
from PIL import Image

st.set_page_config(page_title="Station Analyse RX", layout="wide")

# --- Initialisation de l'historique (Session State) ---
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=[
        "Fichier", "Manque Global %", "Void 1 %", "Void 2 %", "Void 3 %", "Void 4 %", "Void 5 %"
    ])
if 'analyzed_images' not in st.session_state:
    st.session_state.analyzed_images = {} # Stockage des images traitées {nom: image_rgb}

# --- Cache Modèle ---
@st.cache_resource
def load_trained_model(file_upload):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

# --- Sidebar ---
st.sidebar.title("🔍 Configuration")
st.sidebar.info("📂 Dossier de travail :\n`C:\\Users\\kiefferp\\OneDrive - ALL Circuits\\Bureau\\WIP\\_IA\\Analyse RX`")

model_file = st.sidebar.file_uploader("1. Charger le modèle", type=["joblib"])

if model_file:
    clf = load_trained_model(model_file)
    
    st.header("🎯 Analyse et Historique")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg", "bmp"])
    with col_u2:
        mask_upload = st.file_uploader("Masque d'inspection", type=["png", "jpg", "jpeg"])

    if rx_upload and mask_upload:
        # Traitement
        with open("temp_rx.png", "wb") as f: f.write(rx_upload.getbuffer())
        with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())

        img_gray = engine.load_gray("temp_rx.png")
        H, W = img_gray.shape

        # Sidebar Ajustements
        st.sidebar.subheader("🕹️ Ajustement")
        tx = st.sidebar.number_input("X (px)", value=0.0, step=1.0)
        ty = st.sidebar.number_input("Y (px)", value=0.0, step=1.0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")
        if zone_base.shape != (H, W):
            zone_base = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(zone_base > 0)
        cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_base, M, (W, H), flags=cv2.INTER_NEAREST)

        with st.spinner("Analyse..."):
            feats = engine.compute_features(img_gray)
            pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
            pred_bin = (pred == 1).astype(np.uint8) * 255 

        # Détection Voids
        solder_map = np.zeros((H,W), dtype=np.uint8)
        solder_map[(zone_adj > 0) & (pred_bin > 0)] = 255
        contours, hierarchy = cv2.findContours(solder_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        voids_all = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if area > 2:
                        voids_all.append(area)
                        # On ne garde le contour que pour l'affichage (logic simplifiée ici)
        
        voids_sorted = sorted(voids_all, reverse=True)
        top_5_voids = voids_sorted[:5]
        
        # Calculs
        total_px = int(np.sum(zone_adj > 0))
        solder_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
        missing_pct = ((total_px - solder_px) / total_px * 100) if total_px > 0 else 0
        
        # Préparation image résultat
        overlay_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay_rgb[zone_adj > 0] = [255, 0, 0] # Rouge
        overlay_rgb[(zone_adj > 0) & (pred_bin > 0)] = [255, 255, 0] # Jaune
        
        # Cerclage Top 5 sur l'image (refait les contours pour le dessin)
        # (On ré-extrait pour dessiner uniquement les 5 plus gros)
        voids_contours = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    voids_contours.append(contours[i])
        voids_contours = sorted(voids_contours, key=cv2.contourArea, reverse=True)[:5]
        for c in voids_contours:
            cv2.drawContours(overlay_rgb, [c], -1, [0, 255, 255], 2)

        # Affichage image actuelle
        st.image(overlay_rgb, caption=f"Analyse : {rx_upload.name}", use_container_width=True)

        # --- Bouton d'enregistrement dans l'historique ---
        if st.button("📥 Enregistrer ce résultat dans l'historique"):
            new_row = {
                "Fichier": rx_upload.name,
                "Manque Global %": round(missing_pct, 2)
            }
            for i in range(5):
                val = (top_5_voids[i] / total_px * 100) if i < len(top_5_voids) else 0
                new_row[f"Void {i+1} %"] = round(val, 2)
            
            # Mise à jour DataFrame et Galerie
            st.session_state.history_df = pd.concat([st.session_state.history_df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.analyzed_images[rx_upload.name] = overlay_rgb
            st.success("Résultat ajouté au tableau.")

    # --- TABLEAU RECAPITULATIF (Bas de page) ---
    st.divider()
    st.subheader("📊 Historique des analyses")
    st.dataframe(st.session_state.history_df, use_container_width=True)

    # --- GALERIE DE VIGNETTES ---
    st.subheader("🖼️ Galerie des inspections (cliquer pour agrandir)")
    if st.session_state.analyzed_images:
        cols = st.columns(5) # 5 vignettes par ligne
        for idx, (name, img) in enumerate(st.session_state.analyzed_images.items()):
            with cols[idx % 5]:
                st.image(img, caption=name, use_container_width=True)
                if st.button("Voir", key=f"btn_{name}"):
                    st.image(img, caption=f"Zoom sur {name}", use_container_width=True)
    
else:
    st.info("Veuillez charger votre modèle .joblib pour commencer.")
