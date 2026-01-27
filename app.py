import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import datetime
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Analyse Unitaire", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

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
        st.sidebar.divider()
        tx = st.sidebar.number_input("Trans X", value=0)
        ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

        # --- CHARGEMENT ---
        rx_upload.seek(0)
        img_raw = cv2.imdecode(np.frombuffer(rx_upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if img_raw is None:
            st.error("Erreur de lecture.")
            st.stop()

        # Application CLAHE locale
        if contrast_val > 0:
            clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8))
            img_gray = clahe.apply(img_raw)
        else:
            img_gray = img_raw

        H, W = img_gray.shape

        # --- MASQUE ---
        mask_upload.seek(0)
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), cv2.IMREAD_COLOR)
        r_r, g_r, b_r = cv2.split(cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB))
        m_green = (g_r > 100).astype(np.uint8)
        m_black = ((r_r < 100) & (g_r < 100) & (b_r < 100) & (m_green > 0)).astype(np.uint8)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj > 0) & (hol_adj == 0)

        # --- IA ET LOGIQUE DES CLASSES (INVERSÉE) ---
        features = engine.compute_features(img_gray)
        pred_map = np.argmax(clf.predict_proba(features.reshape(-1, features.shape[-1])), axis=1).reshape(H, W)
        
        # Ici, on inverse la logique pour correspondre à tes images :
        # Si pred_map == 1 était la soudure (Rouge sur ton image précédente)
        # On définit maintenant MANQUE = pred_map == 1 (Pour le mettre en ROUGE)
        # Et SOUDURE = pred_map == 0 (Pour le mettre en BLEU)
        
        void_raw = ((pred_map == 1) & (z_utile)).astype(np.uint8) 
        kernel = np.ones((3,3), np.uint8)
        clean_voids = cv2.morphologyEx(void_raw, cv2.MORPH_OPEN, kernel)
        
        clean_solder = (z_utile) & (clean_voids == 0)

        # --- CALCUL VOID MAJEUR ---
        v_max_area, v_max_poly = 0, None
        z_stricte = cv2.erode(z_utile.astype(np.uint8), kernel, iterations=1)
        cnts, _ = cv2.findContours((clean_voids * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10: continue
            c_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_m, [c], -1, 255, -1)
            if not np.any((c_m > 0) & (hol_adj > 0)) and not np.any((c_m > 0) & (z_stricte == 0)):
                if area > v_max_area:
                    v_max_area, v_max_poly = area, c

        # --- RENDU VISUEL ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # SOUDURE PRÉSENTE -> BLEU FONCÉ
        overlay[clean_solder] = [0, 50, 150]
        
        # MANQUES / VOIDS -> ROUGE
        overlay[clean_voids > 0] = [255, 0, 0]
        
        # VOID MAX -> CONTOUR CYAN ÉPAIS
        if v_max_poly is not None:
            cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 3)

        # --- AFFICHAGE ---
        st.divider()
        col_ref, col_ia = st.columns(2)
        with col_ref:
            st.subheader("🖼️ Image Originale (Réf)")
            st.image(img_gray, use_container_width=True)
            area_tot = np.sum(z_utile)
            st.metric("Manque Total (Rouge)", f"{(np.sum(clean_voids)/area_tot*100):.2f} %" if area_tot > 0 else "0 %")
            st.metric("Void Majeur (Cyan)", f"{(v_max_area/area_tot*100):.3f} %" if area_tot > 0 else "0 %")

        with col_ia:
            st.subheader("🤖 Analyse IA")
            st.image(overlay, use_container_width=True)
            if st.button("📥 Archiver", key="final_btn_v5", use_container_width=True):
                st.toast("Résultat archivé")
