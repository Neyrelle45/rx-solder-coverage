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
if 'selected_img' not in st.session_state: st.session_img = None

def highlight_extremes(s):
    if len(s) < 2: return [''] * len(s)
    is_max = s == s.max()
    is_min = s == s.min()
    return ['background-color: #ffcccc' if v else 'background-color: #ccf2ff' if m else '' for v, m in zip(is_max, is_min)]

st.sidebar.title("🛠️ Config")
if st.sidebar.button("🗑️ Vider Historique"): st.session_state.history = []

model_file = st.sidebar.file_uploader("Modèle", type=["joblib"])
contrast_val = st.sidebar.slider("Contraste", 0.0, 10.0, 2.0, 0.5)

if model_file:
    @st.cache_resource
    def load_my_model(file): return joblib.load(file)
    clf = load_my_model(model_file)

    rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "tif"])
    mask_upload = st.file_uploader("2. Masque", type=["png", "jpg"])

    if rx_upload and mask_upload:
        tx = st.sidebar.number_input("Trans X", value=0)
        ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Echelle", 0.8, 1.2, 1.0)

        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape
        insp = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), 1)
        m_green = cv2.resize((insp[:,:,1] > 100).astype(np.uint8), (W, H))
        
        M = engine.compose_similarity(scale, rot, float(tx), float(ty), W/2, H/2)
        z_adj = cv2.warpAffine(m_green, M, (W, H), flags=cv2.INTER_NEAREST)

        features = engine.compute_features(img_gray)
        probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
        pred_map = np.argmax(probs, axis=1).reshape(H, W)
        missing_pct = (1.0 - (np.sum((pred_map == 1) & (z_adj > 0)) / np.sum(z_adj > 0))) * 100

        ov = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        ov[(pred_map == 1) & (z_adj > 0)] = [255, 255, 0]
        ov[(pred_map == 0) & (z_adj > 0)] = [255, 0, 0]
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            if st.button("📥 Archiver"):
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                st.session_state.history.append({"Fichier": rx_upload.name, "Total_%": round(missing_pct, 2), "img_bytes": img_jpg.tobytes()})
        with c2: st.image(ov)

if st.session_state.history:
    st.divider()
    df_hist = pd.DataFrame(st.session_state.history).drop(columns=['img_bytes'])
    styled_hist = df_hist.style.apply(highlight_extremes, subset=['Total_%'], axis=0)
    st.dataframe(styled_hist, use_container_width=True)
