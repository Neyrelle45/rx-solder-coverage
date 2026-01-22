import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import os
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Analyse Batch Pro", layout="wide")

# --- INITIALISATION ---
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def highlight_extremes(s):
    if len(s) < 2: return [''] * len(s)
    is_max = s == s.max()
    is_min = s == s.min()
    return ['background-color: #ffcccc' if v else 'background-color: #ccf2ff' if m else '' for v, m in zip(is_max, is_min)]

st.title("📦 Analyse de Série (Batch)")

# --- CONFIGURATION SIDEBAR ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Charger modèle IA (.joblib)", type=["joblib"])

# MODIFICATION ICI : step=0.1 pour un réglage plus fin
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement commun")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Réinitialiser la session"):
    st.session_state.batch_history = []
    st.rerun()

# --- CHARGEMENT DES FICHIERS ---
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX (plusieurs possibles)", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence (Unique)", type=["png", "jpg"])

if model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    
    if st.button("🚀 Lancer l'analyse de la série", use_container_width=True):
        st.session_state.batch_history = [] 
        
        mask_bytes = mask_file.getvalue()
        insp_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), 1)
        b_r, g_r, r_r = cv2.split(insp_raw)
        m_green_orig = (g_r > 100).astype(np.uint8)
        m_black_orig = ((b_r < 50) & (g_r < 50) & (r_r < 50) & (m_green_orig > 0)).astype(np.uint8)

        progress_bar = st.progress(0)
        
        for idx, rx_file in enumerate(uploaded_rx):
            # 1. Chargement Image
            img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
            H, W = img_gray.shape

            # 2. Ajustement Masque
            m_green_res = cv2.resize(m_green_orig, (W, H), interpolation=cv2.INTER_NEAREST)
            m_black_res = cv2.resize(m_black_orig, (W, H), interpolation=cv2.INTER_NEAREST)
            
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(m_green_res, M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(m_black_res, M, (W, H), flags=cv2.INTER_NEAREST)
            z_utile = (env_adj & ~hol_adj)

            # 3. IA
            features = engine.compute_features(img_gray)
            probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)
            conf_map = np.max(probs, axis=1).reshape(H, W)
            mean_conf = np.mean(conf_map[z_utile > 0]) * 100 if np.any(z_utile) else 0

            # 4. Calculs Couverture
            valid_solder = (pred_map == 1) & (z_utile > 0)
            valid_voids = (pred_map == 0) & (z_utile > 0)
            area_total_px = np.sum(z_utile > 0)
            missing_pct = (1.0 - (np.sum(valid_solder) / area_total_px)) * 100.0 if area_total_px > 0 else 0

            # 5. Recherche du Void Majeur (Strictement enclavé dans le jaune)
            max_void_area = 0
            max_void_poly = None
            solder_u8 = valid_solder.astype(np.uint8) * 255
            solder_cnts, _ = cv2.findContours(solder_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for s_cnt in solder_cnts:
                s_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(s_mask, [s_cnt], -1, 255, -1)
                # Trous à l'intérieur de cet îlot jaune
                holes = cv2.bitwise_and(s_mask, cv2.bitwise_not(solder_u8))
                h_cnts, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for h_cnt in h_cnts:
                    area = cv2.contourArea(h_cnt)
                    if area < 10.0: continue
                    h_m = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(h_m, [h_cnt], -1, 255, -1)
                    # Exclusion si touche une zone noire (via)
                    if not np.any((h_m > 0) & (hol_adj > 0)):
                        if area > max_void_area:
                            max_void_area = area
                            max_void_poly = h_cnt

            max_void_pct = (max_void_area / area_total_px * 100) if area_total_px > 0 else 0

            # 6. Génération Overlay
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay[valid_solder] = [255, 255, 0]
            overlay[valid_voids] = [255, 0, 0]
            if max_void_poly is not None:
                cv2.drawContours(overlay, [max_void_poly], -1, [0, 255, 255], 2)

            # 7. Stockage JPEG (RAM Optimisée)
            _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            st.session_state.batch_history.append({
                "Fichier": rx_file.name,
                "Total_%": round(missing_pct, 2),
                "Void_Max_%": round(max_void_pct, 3),
                "Confiance_%": round(mean_conf, 1),
                "img_bytes": img_jpg.tobytes()
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_rx))
        
        st.success(f"Terminé : {len(uploaded_rx)} images analysées.")

# --- AFFICHAGE ET EXPORT ---
if st.session_state.batch_history:
    st.divider()
    st.subheader("📊 Rapport de Série")
    
    df_full = pd.DataFrame(st.session_state.batch_history)
    df_csv = df_full.drop(columns=['img_bytes'])
    
    # Tableau avec limitation au Void Max
    st.dataframe(df_csv.style.apply(highlight_extremes, subset=['Total_%'], axis=0), use_container_width=True)

    # --- BLOC D'EXTRACTION ZIP ---
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        # Ajout du fichier CSV
        z.writestr("rapport_batch.csv", df_csv.to_csv(index=False))
        # Ajout des images compressées
        for i, item in enumerate(st.session_state.batch_history):
            z.writestr(f"images/{item['Fichier']}_analysed.jpg", item['img_bytes'])
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Télécharger ZIP (Images + CSV)",
            data=zip_buffer.getvalue(),
            file_name=f"batch_rx_{datetime.datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="📄 Télécharger CSV Uniquement",
            data=df_csv.to_csv(index=False),
            file_name="rapport_batch.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Galerie de miniatures
    with st.expander("👁️ Visualiser les résultats de la série"):
        cols = st.columns(6)
        for idx, item in enumerate(st.session_state.batch_history):
            with cols[idx % 6]:
                st.image(item['img_bytes'], caption=f"{item['Total_%']}%")
