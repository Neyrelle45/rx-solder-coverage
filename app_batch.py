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

st.set_page_config(page_title="RX Batch Processor - Optimized", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

st.sidebar.title("🚀 Mode Batch (Optimisé)")

if st.sidebar.button("🗑️ Vider la session", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

st.sidebar.divider()
contrast_val = st.sidebar.slider("Contraste", 0.0, 10.0, 2.0, 0.5)
model_file = st.sidebar.file_uploader("1. Charger modèle (.joblib)", type=["joblib"])

if model_file:
    @st.cache_resource
    def load_my_model(file):
        return joblib.load(file)
    clf = load_my_model(model_file)

    st.header("📦 Traitement par lot")
    
    col_m, col_f = st.columns([1, 2])
    with col_m:
        mask_upload = st.file_uploader("2. Masque Unique", type=["png", "jpg"])
    with col_f:
        rx_uploads = st.file_uploader("3. Images RX (Série)", type=["png", "jpg", "tif"], accept_multiple_files=True)

    if mask_upload and rx_uploads:
        if st.button("▶️ Lancer l'analyse", use_container_width=True):
            # Préparation masque
            insp = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), 1)
            b_c, g_c, r_c = cv2.split(insp)
            m_green = (g_c > 100).astype(np.uint8) 
            m_black = ((b_c < 50) & (g_c < 50) & (r_c < 50) & (m_green > 0)).astype(np.uint8)
            z_utile = ((m_green > 0) & (m_black == 0)).astype(np.uint8)

            prog = st.progress(0)
            for idx, rx_file in enumerate(rx_uploads):
                # Analyse
                img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
                H, W = img_gray.shape
                z_adj = cv2.resize(z_utile, (W, H), interpolation=cv2.INTER_NEAREST)
                env_adj = cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST)
                hol_adj = cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST)

                features = engine.compute_features(img_gray)
                probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
                pred_map = np.argmax(probs, axis=1).reshape(H, W)
                conf_map = np.max(probs, axis=1).reshape(H, W)
                mean_conf = np.mean(conf_map[z_adj > 0]) * 100

                valid_solder = (pred_map == 1) & (z_adj > 0)
                missing_pct = (1.0 - (np.sum(valid_solder) / np.sum(z_adj > 0))) * 100

                # Top 5 Voids (Logique complète conservée)
                v_mask = ((pred_map == 0) & (z_adj > 0)).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                internals = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 3.0: continue
                    c_m = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(c_m, [c], -1, 255, -1)
                    if not np.any((c_m > 0) & (hol_adj > 0)):
                        b_m = np.zeros((H, W), dtype=np.uint8)
                        cv2.drawContours(b_m, [c], -1, 255, 1)
                        if not np.any((cv2.dilate(b_m, np.ones((3,3))) > 0) & (env_adj == 0)):
                            internals.append({'area': area})
                
                top_5 = sorted(internals, key=lambda x: x['area'], reverse=True)[:5]

                # Overlay & Compression JPEG (Sécurité RAM)
                ov = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                ov[valid_solder] = [255, 255, 0]
                ov[(pred_map == 0) & (z_adj > 0)] = [255, 0, 0]
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

                entry = {"Fichier": rx_file.name, "Total_%": round(missing_pct, 2), "Confiance_%": round(mean_conf, 1), "img_bytes": img_jpg.tobytes()}
                for i in range(5): entry[f"V{i+1}_%"] = round((top_5[i]['area']/np.sum(z_adj>0)*100), 3) if i < len(top_5) else 0.0
                st.session_state.batch_history.append(entry)

                # Nettoyage RAM agressif
                del img_gray, features, probs, pred_map, conf_map, ov
                gc.collect()
                prog.progress((idx + 1) / len(rx_uploads))

if st.session_state.batch_history:
    st.divider()
    df_csv = pd.DataFrame(st.session_state.batch_history).drop(columns=['img_bytes'])
    
    # ZIP Optimisé
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rapport.csv", df_csv.to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"analyses/{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("🎁 Télécharger Pack ZIP", buf.getvalue(), "batch_export.zip", "application/zip", use_container_width=True)
    st.dataframe(df_csv, use_container_width=True)
    
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.batch_history):
        with cols[idx % 6]:
            st.image(item['img_bytes'], caption=f"{item['Total_%']}%")
