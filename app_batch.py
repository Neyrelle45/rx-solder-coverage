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

# --- INITIALISATION DE LA SESSION ---
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
        # --- SÉLECTION MULTIPLE ACTIVÉE ---
        rx_uploads = st.file_uploader(
            "3. Sélectionner les images RX (Maintenir Ctrl pour plusieurs)", 
            type=["png", "jpg", "jpeg", "tif"], 
            accept_multiple_files=True
        )
        if rx_uploads:
            st.info(f"📁 {len(rx_uploads)} images chargées en file d'attente.")

    if mask_upload and rx_uploads:
        if st.button("▶️ Lancer l'analyse de la série", use_container_width=True):
            
            # --- PRÉPARATION DU MASQUE UNIQUE ---
            mask_bytes = np.frombuffer(mask_upload.read(), np.uint8)
            insp = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)
            b_c, g_c, r_c = cv2.split(insp)
            m_green = (g_c > 100).astype(np.uint8) 
            m_black = ((b_c < 50) & (g_c < 50) & (r_c < 50) & (m_green > 0)).astype(np.uint8)
            z_utile = ((m_green > 0) & (m_black == 0)).astype(np.uint8)

            status_text = st.empty()
            progress_bar = st.progress(0)
            
            for idx, rx_file in enumerate(rx_uploads):
                status_text.text(f"Analyse de {rx_file.name} ({idx+1}/{len(rx_uploads)})...")
                
                # 1. Charger et traiter image RX
                img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
                H, W = img_gray.shape

                # 2. Ajuster masque
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

                # 4. Calculs Manque soudure
                valid_solder = (pred_map == 1) & (z_adj > 0)
                valid_voids = (pred_map == 0) & (z_adj > 0)
                area_px = np.sum(z_adj > 0)
                missing_pct = (1.0 - (np.sum(valid_solder) / area_px)) * 100.0 if area_px > 0 else 0

                # 5. Filtrage Voids (Top 5)
                v_mask = (valid_voids.astype(np.uint8)) * 255
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

                # 6. Création de l'Overlay & COMPRESSION JPEG (Sécurité RAM)
                overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                overlay[valid_solder] = [255, 255, 0] # Jaune
                overlay[valid_voids] = [255, 0, 0]   # Rouge
                # Note: On ne dessine pas les contours cyan en batch pour gagner du temps, 
                # ou vous pouvez les ajouter ici si nécessaire.

                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

                # 7. Archivage optimisé
                entry = {
                    "Fichier": rx_file.name,
                    "Total_%": round(missing_pct, 2),
                    "Confiance_%": round(mean_conf, 1),
                    "img_bytes": img_jpg.tobytes() # Stockage compact
                }
                for i in range(5):
                    v_val = (top_5[i]['area'] / area_px * 100) if i < len(top_5) else 0.0
                    entry[f"V{i+1}_%"] = round(v_val, 3)
                
                st.session_state.batch_history.append(entry)
                
                # NETTOYAGE MÉMOIRE
                del img_gray, features, probs, pred_map, conf_map, overlay, z_adj, env_adj, hol_adj
                gc.collect()
                progress_bar.progress((idx + 1) / len(rx_uploads))

            status_text.success(f"Analyse terminée : {len(rx_uploads)} images traitées.")

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.batch_history:
    st.divider()
    st.subheader("📊 Résultats de la Série")

    # ZIP avec images compressées
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        df_full = pd.DataFrame(st.session_state.batch_history)
        df_csv = df_full.drop(columns=['img_bytes'])
        z.writestr("rapport_batch.csv", df_csv.to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"analyses/{item['Fichier']}.jpg", item['img_bytes'])

    st.download_button("🎁 Télécharger ZIP Complet", buf.getvalue(), "batch_export.zip", "application/zip", use_container_width=True)

    # Tableau
    st.dataframe(df_csv, use_container_width=True)

    # Galerie de miniatures (Lecture directe des bytes)
    st.write("### 🖼️ Galerie")
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.batch_history):
        with cols[idx % 6]:
            st.image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
