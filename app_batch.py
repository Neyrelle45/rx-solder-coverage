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

st.set_page_config(page_title="RX Batch Processor - Pro", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

# --- FONCTION DE COLORATION ---
def highlight_extremes(s):
    is_max = s == s.max()
    is_min = s == s.min()
    styles = ['background-color: #ffcccc' if v else 'background-color: #ccf2ff' if m else '' for v, m in zip(is_max, is_min)]
    return styles

st.sidebar.title("🚀 Mode Batch")
if st.sidebar.button("🗑️ Vider les résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.sidebar.success("Résultats effacés !")

st.sidebar.divider()
contrast_val = st.sidebar.slider("Contraste", 0.0, 10.0, 2.0, 0.5)
model_file = st.sidebar.file_uploader("1. Charger modèle (.joblib)", type=["joblib"])

if model_file:
    @st.cache_resource
    def load_my_model(file): return joblib.load(file)
    clf = load_my_model(model_file)

    st.header("📦 Traitement par lot")
    col_m, col_f = st.columns([1, 2])
    with col_m:
        mask_upload = st.file_uploader("2. Masque Unique", type=["png", "jpg"])
    with col_f:
        rx_uploads = st.file_uploader("3. Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)

    if mask_upload and rx_uploads:
        if st.button("▶️ Lancer l'analyse", use_container_width=True):
            mask_bytes = np.frombuffer(mask_upload.read(), np.uint8)
            insp = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)
            b_c, g_c, r_c = cv2.split(insp)
            m_green = (g_c > 100).astype(np.uint8) 
            m_black = ((b_c < 50) & (g_c < 50) & (r_c < 50) & (m_green > 0)).astype(np.uint8)
            z_utile = ((m_green > 0) & (m_black == 0)).astype(np.uint8)

            status_text = st.empty()
            progress_bar = st.progress(0)
            
            for idx, rx_file in enumerate(rx_uploads):
                status_text.text(f"Analyse : {rx_file.name}...")
                img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
                H, W = img_gray.shape
                z_adj = cv2.resize(z_utile, (W, H), interpolation=cv2.INTER_NEAREST)
                
                features = engine.compute_features(img_gray)
                probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
                pred_map = np.argmax(probs, axis=1).reshape(H, W)
                conf_map = np.max(probs, axis=1).reshape(H, W)
                mean_conf = np.mean(conf_map[z_adj > 0]) * 100 if np.sum(z_adj > 0) > 0 else 0

                valid_solder = (pred_map == 1) & (z_adj > 0)
                missing_pct = (1.0 - (np.sum(valid_solder) / np.sum(z_adj > 0))) * 100.0 if np.sum(z_adj > 0) > 0 else 0

                # Top 5 Voids simple
                v_mask = ((pred_map == 0) & (z_adj > 0)).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                top_5_areas = sorted([cv2.contourArea(c) for c in cnts], reverse=True)[:5]

                overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                overlay[valid_solder] = [255, 255, 0]
                overlay[(pred_map == 0) & (z_adj > 0)] = [255, 0, 0]
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

                entry = {"Fichier": rx_file.name, "Total_%": round(missing_pct, 2), "Confiance_%": round(mean_conf, 1), "img_bytes": img_jpg.tobytes()}
                for i in range(5): entry[f"V{i+1}_%"] = round((top_5_areas[i]/np.sum(z_adj>0)*100), 3) if i < len(top_5_areas) else 0.0
                st.session_state.batch_history.append(entry)
                
                del img_gray, features, probs, pred_map, overlay
                gc.collect()
                progress_bar.progress((idx + 1) / len(rx_uploads))
            status_text.success("Traitement terminé !")

if st.session_state.batch_history:
    st.divider()
    df_full = pd.DataFrame(st.session_state.batch_history)
    df_display = df_full.drop(columns=['img_bytes'])

    # --- APPLICATION DU STYLE ---
    # On surligne la ligne entière basée sur la colonne 'Total_%'
    styled_df = df_display.style.apply(highlight_extremes, subset=['Total_%'], axis=0)

    st.subheader("📊 Tableau récapitulatif (Rouge=Max Manque, Bleu=Min Manque)")
    st.dataframe(styled_df, use_container_width=True)
    
    # Galerie
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.batch_history):
        with cols[idx % 6]: st.image(item['img_bytes'], caption=f"{item['Fichier']}")
