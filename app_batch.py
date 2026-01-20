import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Batch Processor - Expert Voids", layout="wide")

# --- INITIALISATION ---
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

st.sidebar.title("🚀 Mode Batch (Fixe)")

# BOUTON RESET
if st.sidebar.button("🗑️ Vider la session actuelle", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

st.sidebar.divider()

# 1. RÉGLAGES COMMUNS
contrast_val = st.sidebar.slider("Contraste (appliqué à tous)", 0.0, 10.0, 2.0, 0.5)
model_file = st.sidebar.file_uploader("1. Charger modèle (.joblib)", type=["joblib"])

if model_file:
    @st.cache_resource
    def load_my_model(file):
        return joblib.load(file)
    
    clf = load_my_model(model_file)
    st.sidebar.success("Modèle prêt")

    st.header("📦 Traitement par lot (Masque Unique)")
    
    col_m, col_f = st.columns([1, 2])
    
    with col_m:
        st.subheader("Configuration du Masque")
        mask_upload = st.file_uploader("2. Charger le Masque Unique", type=["png", "jpg"])
        if mask_upload:
            st.image(mask_upload, caption="Masque de référence", use_container_width=True)

    with col_f:
        st.subheader("Images à analyser")
        rx_uploads = st.file_uploader("3. Charger les images RX (Série)", type=["png", "jpg", "tif"], accept_multiple_files=True)

    if mask_upload and rx_uploads:
        if st.button("▶️ Lancer l'analyse de la série", use_container_width=True):
            
            # --- PRÉPARATION DU MASQUE UNIQUE ---
            with open("temp_batch_mask.png", "wb") as f:
                f.write(mask_upload.getbuffer())
            
            insp = cv2.imread("temp_batch_mask.png", cv2.IMREAD_COLOR)
            b_c, g_c, r_c = cv2.split(insp)
            m_green = (g_c > 100).astype(np.uint8) 
            m_black = ((b_c < 50) & (g_c < 50) & (r_c < 50) & (m_green > 0)).astype(np.uint8)
            z_utile = ((m_green > 0) & (m_black == 0)).astype(np.uint8)

            progress_bar = st.progress(0)
            
            for idx, rx_file in enumerate(rx_uploads):
                # 1. Charger et traiter image RX
                img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
                H, W = img_gray.shape

                # 2. Ajuster masque à la taille de l'image (sans repositionnement manuel)
                z_adj = cv2.resize(z_utile, (W, H), interpolation=cv2.INTER_NEAREST)
                env_adj = cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST)
                hol_adj = cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST)

                # 3. Analyse IA
                features = engine.compute_features(img_gray)
                n_f = features.shape[-1]
                probs = clf.predict_proba(features.reshape(-1, n_f))
                pred_map = np.argmax(probs, axis=1).reshape(H, W)
                conf_map = np.max(probs, axis=1).reshape(H, W)
                mean_conf = np.mean(conf_map[z_adj > 0]) * 100 if np.any(z_adj > 0) else 0

                # 4. Calculs et Voids
                valid_solder = (pred_map == 1) & (z_adj > 0)
                valid_voids = (pred_map == 0) & (z_adj > 0)
                area_px = np.sum(z_adj > 0)
                missing_pct = (1.0 - (np.sum(valid_solder) / area_px)) * 100.0 if area_px > 0 else 0

                # Filtrage Voids (Top 5)
                v_mask = (valid_voids.astype(np.uint8)) * 255
                cnts, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                internals = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 3.0: continue
                    c_m = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(c_m, [c], -1, 255, -1)
                    # Exclusion si contient trou noir ou touche bord
                    if not np.any((c_m > 0) & (hol_adj > 0)):
                        b_m = np.zeros((H, W), dtype=np.uint8)
                        cv2.drawContours(b_m, [c], -1, 255, 1)
                        if not np.any((cv2.dilate(b_m, np.ones((3,3))) > 0) & (env_adj == 0)):
                            internals.append({'area': area, 'poly': c})
                
                top_5 = sorted(internals, key=lambda x: x['area'], reverse=True)[:5]

                # 5. Création de l'Overlay
                overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                overlay[valid_solder] = [255, 255, 0] # Jaune
                overlay[valid_voids] = [255, 0, 0]   # Rouge
                for v in top_5:
                    cv2.drawContours(overlay, [v['poly']], -1, [0, 255, 255], 2)

                # 6. Archivage
                entry = {
                    "Fichier": rx_file.name,
                    "Total_%": round(missing_pct, 2),
                    "Confiance_%": round(mean_conf, 1),
                    "image": overlay.copy()
                }
                for i in range(5):
                    v_val = (top_5[i]['area'] / area_px * 100) if i < len(top_5) else 0.0
                    entry[f"V{i+1}_%"] = round(v_val, 3)
                
                st.session_state.batch_history.append(entry)
                progress_bar.progress((idx + 1) / len(rx_uploads))

            st.success(f"Analyse terminée : {len(rx_uploads)} images traitées.")

# --- AFFICHAGE DES RÉSULTATS (TABLEAU ET VIGNETTES) ---
if st.session_state.batch_history:
    st.divider()
    st.subheader("📊 Résultats de la Série")

    # Section Téléchargement
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        df_full = pd.DataFrame(st.session_state.batch_history)
        df_csv = df_full.drop(columns=['image'])
        z.writestr("rapport_batch.csv", df_csv.to_csv(index=False))
        for i, item in enumerate(st.session_state.batch_history):
            _, img_enc = cv2.imencode(".png", cv2.cvtColor(item['image'], cv2.COLOR_RGB2BGR))
            z.writestr(f"analyses/{item['Fichier']}", img_enc.tobytes())

    st.download_button("🎁 Télécharger l'ensemble des résultats (.zip)", buf.getvalue(), f"batch_export_{datetime.datetime.now().strftime('%H%M')}.zip", "application/zip", use_container_width=True)

    # Tableau Global
    st.dataframe(df_csv, use_container_width=True)

    # Vignettes
    st.write("### 🖼️ Galerie des analyses")
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.batch_history):
        with cols[idx % 6]:
            st.image(cv2.resize(item['image'], (180, 180)), use_container_width=True)
            st.caption(f"{item['Fichier']} - {item['Total_%']}%")
