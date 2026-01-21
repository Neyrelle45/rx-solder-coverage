import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import gc
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Optimized", layout="wide")

if 'history' not in st.session_state: st.session_state.history = []
if 'selected_img' not in st.session_state: st.session_state.selected_img = None

st.sidebar.title("🛠️ Config")
if st.sidebar.button("🗑️ Vider Historique"):
    st.session_state.history = []
    st.session_state.selected_img = None

model_file = st.sidebar.file_uploader("Modèle", type=["joblib"])
contrast_val = st.sidebar.slider("Contraste", 0.0, 10.0, 2.0, 0.5)

if model_file:
    @st.cache_resource
    def load_my_model(file): return joblib.load(file)
    clf = load_my_model(model_file)

    col_u, col_m = st.columns(2)
    with col_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "tif"])
    with col_m: mask_upload = st.file_uploader("2. Masque", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # Alignement (Step 1)
        tx = st.sidebar.number_input("Trans X", value=0, step=1)
        ty = st.sidebar.number_input("Trans Y", value=0, step=1)
        rot = st.sidebar.slider("Rotation", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Echelle", 0.8, 1.2, 1.0)

        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape
        
        # Traitement masque
        mask_bytes = mask_upload.getvalue()
        insp = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), 1)
        b_c, g_c, r_c = cv2.split(insp)
        m_green = cv2.resize((g_c > 100).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        m_black = cv2.resize(((b_c < 50) & (g_c < 50) & (m_green > 0)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        
        M = engine.compose_similarity(scale, rot, float(tx), float(ty), W/2, H/2)
        z_adj = cv2.warpAffine((m_green & ~m_black), M, (W, H), flags=cv2.INTER_NEAREST)
        env_adj = cv2.warpAffine(m_green, M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(m_black, M, (W, H), flags=cv2.INTER_NEAREST)

        # IA & Voids
        features = engine.compute_features(img_gray)
        probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
        pred_map = np.argmax(probs, axis=1).reshape(H, W)
        mean_conf = np.mean(np.max(probs, axis=1)[z_adj.flatten() > 0]) * 100
        
        valid_solder = (pred_map == 1) & (z_adj > 0)
        missing_pct = (1.0 - (np.sum(valid_solder) / np.sum(z_adj > 0))) * 100

        # Overlay
        ov = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        ov[valid_solder] = [255, 255, 0]
        ov[(pred_map == 0) & (z_adj > 0)] = [255, 0, 0]
        
        # Top 5 et affichage
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            st.metric("Confiance", f"{mean_conf:.1f} %")
            if st.button("📥 Archiver"):
                # COMPRESSION JPEG POUR LA RAM
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                st.session_state.history.append({
                    "Fichier": rx_upload.name, "Total_%": round(missing_pct, 2), "img_bytes": img_jpg.tobytes()
                })
                st.toast("Archivé")
        with c2: st.image(ov, use_container_width=True)

if st.session_state.history:
    st.divider()
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 6]:
            if st.button(f"🔎 {idx+1}", key=f"btn_{idx}"): st.session_state.selected_img = item['img_bytes']
            st.image(item['img_bytes'])
    if st.session_state.selected_img:
        st.image(st.session_state.selected_img, use_container_width=True)
        if st.button("Fermer"): st.session_state.selected_img = None
