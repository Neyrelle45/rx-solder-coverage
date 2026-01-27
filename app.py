import streamlit as st
import cv2
import numpy as np
import joblib
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - FIX COULEURS", layout="wide")

st.sidebar.title("🛠️ Configuration")
model_file = st.sidebar.file_uploader("1. Charger modèle (.joblib)", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

if model_file:
    @st.cache_resource
    def load_my_model(file): return joblib.load(file)
    clf = load_my_model(model_file)

    st.header("🔍 Analyse Comparative")
    
    c_u, c_m = st.columns(2)
    with c_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with c_m: mask_upload = st.file_uploader("2. Masque de référence", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- RÉGLAGES ALIGNEMENT ---
        tx = st.sidebar.number_input("Trans X", value=0)
        ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

        # --- CHARGEMENT ---
        rx_upload.seek(0)
        img_raw = cv2.imdecode(np.frombuffer(rx_upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Application CLAHE
        clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8)) if contrast_val > 0 else None
        img_gray = clahe.apply(img_raw) if clahe else img_raw
        H, W = img_gray.shape

        # --- TRAITEMENT MASQUE ---
        mask_upload.seek(0)
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), cv2.IMREAD_COLOR)
        r_r, g_r, b_r = cv2.split(cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB))
        m_green = (g_r > 100).astype(np.uint8)
        m_black = ((r_r < 100) & (g_r < 100) & (b_r < 100) & (m_green > 0)).astype(np.uint8)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj > 0) & (hol_adj == 0)

        # --- PRÉDICTION IA ---
        features = engine.compute_features(img_gray)
        pred_map = clf.predict(features.reshape(-1, features.shape[-1])).reshape(H, W)

        # --- ATTRIBUTION RADICALE DES COULEURS ---
        # Sur tes images, la zone de soudure (matière) est la Classe 1.
        # Nous voulons : Soudure = BLEU, Manque = ROUGE.
        
        mask_solder = ((pred_map == 1) & (z_utile)) # Classe 1 forcée en bleu
        mask_voids = ((pred_map == 0) & (z_utile))  # Classe 0 forcée en rouge

        # --- RENDU FINAL ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Coloration SOUDURE (Bleu)
        overlay[mask_solder] = [0, 50, 160] 
        
        # Coloration MANQUE (Rouge)
        overlay[mask_voids] = [255, 0, 0]

        # --- VOID MAJEUR (CYAN) ---
        v_max_area, v_max_poly = 0, None
        cnts, _ = cv2.findContours(mask_voids.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > v_max_area:
                v_max_area, v_max_poly = area, c
        if v_max_poly is not None:
            cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 3)

        # --- AFFICHAGE ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Image Originale (Réf)")
            st.image(img_gray, use_container_width=True)
            area_tot = np.sum(z_utile)
            st.metric("Manque Total (Rouge)", f"{(np.sum(mask_voids)/area_tot*100):.2f} %" if area_tot > 0 else "0 %")
            
        with col2:
            st.subheader("🤖 Analyse IA")
            st.image(overlay, use_container_width=True)
            st.button("📥 Archiver", key="save_final_fix")
