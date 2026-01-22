import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Final", layout="wide")

# --- INITIALISATION ---
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

# Correction de la fonction de coloration pour éviter l'erreur sur le subset
def highlight_max(s):
    if s.dtype == object or len(s) < 2: 
        return [''] * len(s)
    is_max = s == s.max()
    return ['background-color: #ffcccc' if v else '' for v in is_max]

st.title("📦 Analyse de Série (Batch)")

# --- CONFIGURATION SIDEBAR ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Charger modèle IA (.joblib)", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement commun")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider l'historique des résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

# --- CHARGEMENT DES FICHIERS ---
st.divider()
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX (Série)", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence (Unique)", type=["png", "jpg"])

# --- ACTIONS ---
st.divider()
if model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    
    if st.button("🚀 Lancer l'analyse de la série", use_container_width=True):
        st.session_state.batch_history = [] 
        
        # Lecture du masque identique à app.py
        mask_bytes = mask_file.getvalue()
        mask_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), 1)
        b_m, g_m, r_m = cv2.split(mask_raw)
        
        # Prégénération des masques de base
        m_green_base = (g_m > 100).astype(np.uint8)
        m_black_base = ((b_m < 50) & (g_m < 50) & (r_m < 50)).astype(np.uint8)

        progress_bar = st.progress(0)
        
        for idx, rx_file in enumerate(uploaded_rx):
            # 1. Image
            img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
            H, W = img_gray.shape

            # 2. Masques & Alignement
            m_g_res = cv2.resize(m_green_base, (W, H), interpolation=cv2.INTER_NEAREST)
            m_b_res = cv2.resize(m_black_base, (W, H), interpolation=cv2.INTER_NEAREST)
            
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(m_g_res, M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(m_b_res, M, (W, H), flags=cv2.INTER_NEAREST)
            
            # Zone utile : exclusion stricte des vias (noir)
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_total_px = np.sum(z_utile)

            # 3. IA
            features = engine.compute_features(img_gray)
            probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)

            # 4. Filtrage micro-bulles (en jaune si < 0.1%)
            void_raw = ((pred_map == 0) & z_utile).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(void_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_voids = np.zeros((H, W), dtype=bool)
            for c in cnts:
                area = cv2.contourArea(c)
                if area_total_px > 0 and (area / area_total_px * 100) >= 0.1:
                    cv2.drawContours(final_voids.view(np.uint8), [c], -1, 1, -1)

            # 5. Définition des zones finales
            display_solder = z_utile & (~final_voids)
            display_voids = final_voids
            
            # 6. Void Majeur (Cyan)
            max_v_area = 0
            max_v_poly = None
            v_cnts, _ = cv2.findContours(display_voids.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for vc in v_cnts:
                area = cv2.contourArea(vc)
                if area > max_v_area:
                    max_v_area = area
                    max_v_poly = vc

            # 7. Stats
            missing_pct = (np.sum(display_voids) / area_total_px * 100) if area_total_px > 0 else 0
            v_max_pct = (max_v_area / area_total_px * 100) if area_total_px > 0 else 0

            # 8. Overlay
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay[display_solder] = [255, 255, 0]
            overlay[display_voids] = [255, 0, 0]
            if max_v_poly is not None:
                cv2.drawContours(overlay, [max_v_poly], -1, [0, 255, 255], 2)

            # 9. Stockage
            _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            st.session_state.batch_history.append({
                "Fichier": rx_file.name,
                "Total_%": round(missing_pct, 2),
                "Void_Max_%": round(v_max_pct, 3),
                "img_bytes": img_jpg.tobytes()
            })
            progress_bar.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE ET EXPORTS ---
if st.session_state.batch_history:
    st.divider()
    df = pd.DataFrame(st.session_state.batch_history)
    df_display = df.drop(columns=['img_bytes'])
    
    # Affichage du tableau (Correction erreur ligne 127)
    st.dataframe(df_display.style.apply(highlight_max, subset=['Total_%']), use_container_width=True)

    # Export ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport.csv", df_display.to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button("📥 Télécharger ZIP", zip_buf.getvalue(), "batch.zip", "application/zip", use_container_width=True)
    col_dl2.download_button("📄 Télécharger CSV", df_display.to_csv(index=False), "rapport.csv", "text/csv", use_container_width=True)

    with st.expander("👁️ Résultats Visuels", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=item['Fichier'])
