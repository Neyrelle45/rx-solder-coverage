import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os

st.set_page_config(page_title="Contrôle Qualité RX - Voids", layout="wide")

@st.cache_resource
def load_trained_model(file_upload):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

st.sidebar.title("🔍 Configuration")
model_file = st.sidebar.file_uploader("Charger le modèle (.joblib)", type=["joblib"])

if model_file:
    clf = load_trained_model(model_file)
    st.header("🎯 Analyse des Voids (Bulles Internes)")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg", "bmp"])
    with col_u2:
        mask_upload = st.file_uploader("Masque d'inspection", type=["png", "jpg", "jpeg"])

    if rx_upload and mask_upload:
        # Sauvegardes temporaires pour le moteur
        with open("temp_rx.png", "wb") as f: f.write(rx_upload.getbuffer())
        with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())

        img_gray = engine.load_gray("temp_rx.png")
        H, W = img_gray.shape

        # --- Sidebar Alignement ---
        st.sidebar.subheader("🕹️ Ajustement Masque")
        tx = st.sidebar.number_input("X (px)", value=0.0)
        ty = st.sidebar.number_input("Y (px)", value=0.0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")
        if zone_base.shape != (H, W):
            zone_base = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(zone_base > 0)
        cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_base, M, (W, H), flags=cv2.INTER_NEAREST)

        with st.spinner("Analyse IA en cours..."):
            feats = engine.compute_features(img_gray)
            pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
            pred_bin = (pred == 1).astype(np.uint8) * 255 # 255 = Soudure

        # --- LOGIQUE DE DÉTECTION DES VOIDS ROBUSTES ---
        # 1. On crée une image binaire où la soudure est BLANCHE (255)
        # 2. On cherche les TROUS (trous blancs) à l'intérieur de ces formes
        
        # On ne travaille que dans la zone d'inspection
        solder_in_zone = np.zeros((H,W), dtype=np.uint8)
        solder_in_zone[(zone_adj > 0) & (pred_bin > 0)] = 255
        
        # Trouver les contours avec hiérarchie (RETR_CCOMP détecte les trous internes)
        contours, hierarchy = cv2.findContours(solder_in_zone, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        voids_found = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                # h[3] est l'index du parent. Si h[3] != -1, c'est un trou interne (un void)
                if h[3] != -1:
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if area > 3: # Filtre petit bruit
                        voids_found.append({"area": area, "contour": cnt})

        voids_found = sorted(voids_found, key=lambda x: x['area'], reverse=True)

        # --- VISUALISATION ---
        overlay_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # Jaune = Soudure, Rouge = Manque (fond de zone)
        overlay_rgb[zone_adj > 0] = [255, 0, 0] # Fond rouge par défaut
        overlay_rgb[(zone_adj > 0) & (pred_bin > 0)] = [255, 255, 0] # Jaune là où il y a soudure

        # Cyan = Voids
        for i, v in enumerate(voids_found[:5]):
            cv2.drawContours(overlay_rgb, [v['contour']], -1, [0, 255, 255], -1 if i==0 else 2)

        # --- AFFICHAGE ---
        st.divider()
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            st.image(overlay_rgb, caption="Visualisation : Cyan = Voids réels (bulles emprisonnées)", use_container_width=True)
            
            # Export
            result_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', result_bgr)
            st.download_button("💾 Télécharger l'image", buffer.tobytes(), "analyse_voids.png")

        with res_col2:
            total_px = int(np.sum(zone_adj > 0))
            solder_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
            missing_px = total_px - solder_px
            missing_pct = (missing_px / total_px * 100) if total_px > 0 else 0
            
            st.subheader("📊 Taux Global")
            st.table(pd.DataFrame({
                "Métriques": ["Zone inspection", "Soudure réelle", "Manque total"],
                "Pixels": [total_px, solder_px, missing_px]
            }))
            st.metric("Taux de Manque Total", f"{missing_pct:.2f} %")

            st.divider()
            st.subheader("🔝 Top 5 Voids Internes")
            if voids_found:
                v_list = []
                for i, v in enumerate(voids_found[:5]):
                    v_pct = (v['area'] / total_px * 100)
                    v_list.append({"Rang": i+1, "Surface (px)": int(v['area']), "% de la Zone": round(v_pct, 2)})
                st.table(pd.DataFrame(v_list))
            else:
                st.success("Aucun void interne détecté.")
