import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Correction Définitive", layout="wide")

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

        # --- CHARGEMENT ET CLAHE ---
        rx_upload.seek(0)
        img_raw = cv2.imdecode(np.frombuffer(rx_upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_raw is None: st.stop()

        clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8)) if contrast_val > 0 else None
        img_gray = clahe.apply(img_raw) if clahe else img_raw
        H, W = img_gray.shape

        # --- MASQUE ET ZONE UTILE ---
        mask_upload.seek(0)
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), cv2.IMREAD_COLOR)
        r_r, g_r, b_r = cv2.split(cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB))
        m_green = (g_r > 100).astype(np.uint8)
        m_black = ((r_r < 100) & (g_r < 100) & (b_r < 100) & (m_green > 0)).astype(np.uint8)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj > 0) & (hol_adj == 0)

        # --- ANALYSE IA ---
        features = engine.compute_features(img_gray)
        pred_map = clf.predict(features.reshape(-1, features.shape[-1])).reshape(H, W)

        # --- LOGIQUE DE COULEUR FORCÉE ---
        # 1. On crée l'image de base (niveaux de gris)
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # 2. On identifie la soudure (matière sombre)
        # On force le BLEU sur la soudure (Classe identifiée comme matière)
        # Si le rouge persiste, on inverse manuellement les labels ici :
        label_soudure = 1 # Change en 0 si la soudure reste rouge
        label_manque = 0  # Change en 1 si le manque reste bleu
        
        mask_soudure = ((pred_map == label_soudure) & (z_utile))
        mask_manque = ((pred_map == label_manque) & (z_utile))

        # --- APPLICATION DES COULEURS ---
        # SOUDURE = BLEU
        overlay[mask_soudure] = [0, 50, 160]
        
        # MANQUES = ROUGE (On applique après pour qu'ils soient au-dessus)
        overlay[mask_manque] = [255, 0, 0]

        # --- VOID MAJEUR (CYAN) ---
        v_max_area, v_max_poly = 0, None
        cnts, _ = cv2.findContours((mask_manque.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > v_max_area:
                v_max_area, v_max_poly = area, c
        
        if v_max_poly is not None:
            cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 3)

        # --- AFFICHAGE ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🖼️ Image Originale")
            st.image(img_gray, use_container_width=True)
            area_tot = np.sum(z_utile)
            st.metric("Manque Total (Rouge)", f"{(np.sum(mask_manque)/area_tot*100):.2f} %")
        with c2:
            st.subheader("🤖 Analyse IA")
            st.image(overlay, use_container_width=True)
            
            # BOUTON DE SECOURS : Si les couleurs sont encore inversées
            if st.button("🔄 Inverser Soudure/Manque"):
                st.info("Inversez les variables 'label_soudure' et 'label_manque' dans le code ligne 54-55.")
