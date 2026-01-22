import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Stable Final", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def highlight_max(s):
    return ['background-color: #ffcccc' if v == s.max() and v > 0 else '' for v in s]

st.title("📦 Analyse de Série (Batch)")

# --- PARAMÈTRES ---
st.sidebar.title("⚙️ Configuration")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider les résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

# --- CHARGEMENT ---
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

st.divider()
trigger = st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary")

if trigger:
    if not model_file or not uploaded_rx or not mask_file:
        st.error("Veuillez charger le modèle, les images et le masque.")
    else:
        clf = joblib.load(model_file)
        st.session_state.batch_history = [] 
        
        # CHARGEMENT DU MASQUE SOURCE (Lecture propre)
        m_bytes = mask_file.read()
        m_raw = cv2.imdecode(np.frombuffer(m_bytes, np.uint8), cv2.IMREAD_COLOR)
        m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
        r_src, g_src, b_src = cv2.split(m_rgb)
        
        base_green = (g_src > 100).astype(np.uint8)
        base_black = ((r_src < 60) & (g_src < 60) & (b_src < 60) & (base_green > 0)).astype(np.uint8)

        progress = st.progress(0)
        
        for idx, rx_f in enumerate(uploaded_rx):
            # 1. Image Grise & Dimensions
            img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
            H, W = img_gray.shape

            # 2. Synchronisation GÉOMÉTRIQUE (Évite le bug gauche/droite)
            m_g_local = cv2.resize(base_green, (W, H), interpolation=cv2.INTER_NEAREST)
            m_b_local = cv2.resize(base_black, (W, H), interpolation=cv2.INTER_NEAREST)
            
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(m_g_local, M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(m_b_local, M, (W, H), flags=cv2.INTER_NEAREST)
            
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_px = np.sum(z_utile)

            # 3. IA
            feats = engine.compute_features(img_gray)
            probs = clf.predict_proba(feats.reshape(-1, feats.shape[-1]))
            pred = np.argmax(probs, axis=1).reshape(H, W)

            # 4. Filtrage Bulles (Rouge si > 0.1%, sinon Jaune)
            void_mask = (pred == 0) & z_utile
            cnts, _ = cv2.findContours(void_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask_red = np.zeros((H, W), dtype=bool)
            for c in cnts:
                if area_px > 0 and (cv2.contourArea(c) / area_px * 100) >= 0.1:
                    cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

            # 5. Identification Macro-Void (Cyan) - Test d'enclavement
            v_max_area = 0
            v_max_poly = None
            red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for rc in red_cnts:
                area = cv2.contourArea(rc)
                test_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(test_mask, [rc], -1, 255, 1) 
                if not np.any((test_mask > 0) & (hol_adj > 0)):
                    if area > v_max_area:
                        v_max_area = area
                        v_max_poly = rc

            # 6. Overlay & Rendu
            res_rgb = cv
