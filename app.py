import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine 

st.set_page_config(page_title="Station Analyse RX - Expert Voids", layout="wide")

# Initialisation du stockage en session
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

st.sidebar.title("🛠️ Configuration")

# 1. RÉGLAGES IMAGE & IA
contrast_val = st.sidebar.slider("Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)
model_file = st.sidebar.file_uploader("Charger modèle (.joblib)", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    st.sidebar.success("Modèle opérationnel")

    st.header("🔍 Analyse de Soudure")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- CHARGEMENT ---
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # --- ALIGNEMENT DU MASQUE (PAS DE 1) ---
        st.sidebar.subheader("🕹️ Alignement manuel")
        tx = st.sidebar.number_input("Translation X (px)", value=0, step=1)
        ty = st.sidebar.number_input("Translation Y (px)", value=0, step=1)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0, 0.5)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        with open("temp_app_mask.png", "wb") as f:
            f.write(mask_upload.getbuffer())
        
        # Extraction masques
        insp = cv2.imread("temp_app_mask.png", cv2.IMREAD_COLOR)
        b_c, g_c, r_c = cv2.split(insp)
        mask_green_raw = (g_c > 100).astype(np.uint8) 
        mask_black_raw = ((b_c < 50) & (g_c < 50) & (r_c < 50) & (mask_green_raw > 0)).astype(np.uint8)
        zone_utile_raw = ((mask_green_raw > 0) & (mask_black_raw == 0)).astype(np.uint8)
        
        if zone_utile_raw.shape != (H, W):
            zone_utile_raw = cv2.resize(zone_utile_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_green_raw = cv2.resize(mask_green_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_black_raw = cv2.resize(mask_black_raw, (W, H), interpolation=cv2.INTER_NEAREST)

        M = engine.compose_similarity(scale, rot, float(tx), float(ty), W/2, H/2)
        zone_adj = cv2.warpAffine(zone_utile_raw, M, (W, H), flags=cv2.INTER_NEAREST)
        envelope_adj = cv2.warpAffine(mask_green_raw, M, (W, H), flags=cv2.INTER_NEAREST)
        holes_adj = cv2.warpAffine(mask_black_raw, M, (W, H), flags=cv2.INTER_NEAREST)

        with st.spinner("Analyse IA..."):
            features = engine.compute_features(img_gray)
            pred_map = clf.predict(features.reshape(-1, 3)).reshape(H, W)

        # --- FILTRAGE VOIDS ---
        valid_solder = (pred_map == 1) & (zone_adj > 0)
        valid_voids_all = (pred_map == 0) & (zone_adj > 0)
        area_total_px = np.sum(zone_adj > 0)
        missing_pct = (1.0 - (np.sum(valid_solder) / area_total_px)) * 100.0 if area_total_px > 0 else 0

        void_mask_u8 = (valid_voids_all.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(void_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        internal_voids = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 3.0: continue
            
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1) 
            
            # FILTRES : Pas de trou noir dedans ET ne touche pas le bord externe
            contains_via = np.any((c_mask > 0) & (holes_adj > 0))
            border_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(border_mask, [c], -1, 255, 1)
            dilated = cv2.dilate(border_mask, np.ones((3,3)))
            touches_edge = np.any((dilated > 0) & (envelope_adj == 0))
            
            if not touches_edge and not contains_via:
                internal_voids.append({'area': area, 'poly': c})
        
        top_5 = sorted(internal_voids, key=lambda x: x['area'], reverse=True)[:5]

        # --- OVERLAY ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[valid_solder] = [255, 255, 0] 
        overlay[valid_voids_all] = [255, 0, 0]
        for v in top_5:
            cv2.drawContours(overlay, [v['poly']], -1, [0, 255, 255], 2)

        # --- RÉSULTATS & ARCHIVAGE ---
        st.divider()
        c_res, c_img = st.columns([1, 2])
        with c_res:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            void_stats = {}
            for i in range(5):
                # Calcul sécurisé du pourcentage par void
                v_pct = (top_5[i]['area']
