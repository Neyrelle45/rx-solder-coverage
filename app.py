import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import analyse_rx_soudure as engine 

st.set_page_config(page_title="Station Analyse RX - Expert Voids", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🛠️ Configuration")

# 1. RÉGLAGES IMAGE
contrast_val = st.sidebar.slider("Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)

# 2. CHARGEMENT DU MODÈLE IA
model_file = st.sidebar.file_uploader("Charger modèle (.joblib)", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    st.sidebar.success("Modèle opérationnel")

    st.header("🔍 Analyse de Soudure & Voids Internes")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # --- ALIGNEMENT DU MASQUE ---
        st.sidebar.subheader("🕹️ Alignement manuel")
        tx = st.sidebar.number_input("Translation X", value=0.0)
        ty = st.sidebar.number_input("Translation Y", value=0.0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        with open("temp_app_mask.png", "wb") as f:
            f.write(mask_upload.getbuffer())
        
        # Récupération de la zone utile (Vert sans Noir) ET de la zone brute (Vert avec Noir)
        # On modifie légèrement la logique pour obtenir l'enveloppe externe
        insp = cv2.imread("temp_app_mask.png", cv2.IMREAD_COLOR)
        b,g,r = cv2.split(insp)
        mask_full_green = (g > 100).astype(np.uint8) # Enveloppe externe complète
        
        zone_utile, _, _ = engine.compute_zone_and_holes("temp_app_mask.png")
        
        if zone_utile.shape != (H, W):
            zone_utile = cv2.resize(zone_utile, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_full_green = cv2.resize(mask_full_green, (W, H), interpolation=cv2.INTER_NEAREST)

        cx, cy = (W/2, H/2)
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_utile, M, (W, H), flags=cv2.INTER_NEAREST)
        envelope_adj = cv2.warpAffine(mask_full_green, M, (W, H), flags=cv2.INTER_NEAREST)

        # --- PRÉDICTION IA ---
        with st.spinner("Analyse IA..."):
            features = engine.compute_features(img_gray)
            flat_features = features.reshape(-1, features.shape[-1])
            pred_flat = clf.predict(flat_features)
            pred_map = pred_flat.reshape(H, W)

        # --- CALCULS ---
        valid_solder = (pred_map == 1) & (zone_adj > 0)
        valid_voids_all = (pred_map == 0) & (zone_adj > 0)
        
        area_total_px = np.sum(zone_adj > 0)
        missing_pct_global = (1.0 - (np.sum(valid_solder) / area_total_px)) * 100.0 if area_total_px > 0 else 0

        # --- FILTRAGE DES VOIDS ---
        void_mask_u8 = (valid_voids_all.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(void_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        internal_voids = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1.5: continue
            
            # Création du contour dilaté
            c_mask = np.zeros(void_mask_u8.shape, dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, 1)
            dilated = cv2.dilate(c_mask, np.ones((3,3), np.uint8), iterations=1)
            
            # NOUVELLE LOGIQUE : Le void est valide s'il ne touche pas l'extérieur 
            # de l'enveloppe verte globale (envelope_adj)
            touches_external_edge = np.any((dilated > 0) & (envelope_adj == 0))
            
            if not touches_external_edge:
                internal_voids.append({'area': area, 'poly': c})
        
        internal_voids = sorted(internal_voids, key=lambda x: x['area'], reverse=True)
        top_5 = internal_voids[:5]

        # --- OVERLAY ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[valid_solder] = [255, 255, 0] 
        overlay[valid_voids_all] = [255, 0, 0]
        
        cyan_rgb = [0, 255, 255]
        for v in top_5:
            cv2.drawContours(overlay, [v['poly']], -1, cyan_rgb, 2)

        # --- AFFICHAGE ---
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Manque Total", f"{missing_pct_global:.2f} %")
            st.write("📏 **Top 5 Voids Internes**")
            void_stats = {}
            for i in range(5):
                val = (top_5[i]['area'] / area_total_px * 100) if i < len(top_5) else 0.0
                st.caption(f"Void {i+1} : {val:.3f} %")
                void_stats[f"V{i+1}_%"] = round(val, 3)
            
            if st.button("📥 Archiver"):
                data = {"Fichier": rx_upload.name, "Global_%": round(missing_pct_global, 2)}
                data.update(void_stats)
                st.session_state.history.append(data)
                st.rerun()

        with c2:
            st.image(overlay, caption="Jaune: OK | Rouge: Manque | Cyan: Voids Internes", use_container_width=True)

if st.session_state.history:
    st.divider()
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
