import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os

st.set_page_config(page_title="Contrôle Qualité RX", layout="wide")

@st.cache_resource
def load_trained_model(file_upload):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

st.sidebar.title("🔍 Configuration")
model_file = st.sidebar.file_uploader("Modèle (.joblib)", type=["joblib"])

if model_file:
    clf = load_trained_model(model_file)
    st.header("🎯 Analyse Précise des Voids")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg", "bmp"])
    with col_u2:
        mask_upload = st.file_uploader("Masque d'inspection", type=["png", "jpg", "jpeg"])

    if rx_upload and mask_upload:
        with open("temp_rx.png", "wb") as f: f.write(rx_upload.getbuffer())
        with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())

        img_gray = engine.load_gray("temp_rx.png")
        H, W = img_gray.shape

        # --- Réglages ---
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

        with st.spinner("Analyse IA..."):
            feats = engine.compute_features(img_gray)
            pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
            pred_bin = (pred == 1).astype(np.uint8) * 255 

        # --- DÉTECTION DES VOIDS (CERCLAGE UNIQUEMENT) ---
        solder_map = np.zeros((H,W), dtype=np.uint8)
        solder_map[(zone_adj > 0) & (pred_bin > 0)] = 255
        
        # On utilise la hiérarchie pour trouver les trous DANS la soudure
        contours, hierarchy = cv2.findContours(solder_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        voids_found = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1: # C'est un trou interne
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if area > 2:
                        voids_found.append({"area": area, "contour": cnt})

        voids_found = sorted(voids_found, key=lambda x: x['area'], reverse=True)

        # --- VISUALISATION ---
        overlay_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay_rgb[zone_adj > 0] = [255, 0, 0] # Fond Rouge
        overlay_rgb[(zone_adj > 0) & (pred_bin > 0)] = [255, 255, 0] # Soudure Jaune

        # Cerclage CYAN FIN pour tous les voids
        for v in voids_found[:10]: # On cercle les 10 plus gros
            cv2.drawContours(overlay_rgb, [v['contour']], -1, [0, 255, 255], 1) # Épaisseur 1 pour la finesse

        # --- TABLEAUX ---
        st.divider()
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            st.image(overlay_rgb, caption="Jaune=Soudure | Rouge=Manque | Cyan=Cerclage Voids", use_container_width=True)
            
            result_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', result_bgr)
            st.download_button("💾 Télécharger", buffer.tobytes(), "analyse_voids.png")

        with res_col2:
            total_px = int(np.sum(zone_adj > 0))
            solder_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
            missing_px = total_px - solder_px
            
            st.subheader("📊 Taux Global")
            st.table(pd.DataFrame({
                "Métriques": ["Surface Totale", "Soudure", "Manque"],
                "Pixels": [total_px, solder_px, missing_px]
            }))
            st.metric("Manque Total / Zone", f"{(missing_px/total_px*100):.2f} %")

            st.divider()
            st.subheader("🔝 Top 5 Voids")
            if voids_found:
                v_list = []
                for i, v in enumerate(voids_found[:5]):
                    # On affiche le % par rapport à la zone totale
                    # Si vous voulez le % par rapport à la soudure détectée : (v['area']/solder_px*100)
                    v_pct = (v['area'] / total_px * 100)
                    v_list.append({
                        "Rang": i+1, 
                        "Surface (px)": int(v['area']), 
                        "% de la Zone": f"{v_pct:.2f} %"
                    })
                st.table(pd.DataFrame(v_list))
            else:
                st.info("Aucun void détecté.")
