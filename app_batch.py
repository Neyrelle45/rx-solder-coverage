import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Analyse Batch (Sync Fix)", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def highlight_extremes(s):
    if len(s) < 2: return [''] * len(s)
    is_max = s == s.max()
    return ['background-color: #ffcccc' if v else '' for v in is_max]

st.title("📦 Analyse de Série (Batch Correctif)")

st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement commun")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

c_run, c_clear = st.columns([3, 1])
run_analysis = c_run.button("🚀 Lancer l'analyse", use_container_width=True)
if c_clear.button("🗑️ Vider les résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

if model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    
    if run_analysis:
        st.session_state.batch_history = [] 
        
        # --- LECTURE SÉCURISÉE DU MASQUE ---
        mask_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
        b_r, g_r, r_r = cv2.split(mask_raw)
        m_green_base = (g_r > 100).astype(np.uint8)
        m_black_base = ((b_r < 50) & (g_r < 50) & (r_r < 50) & (m_green_base > 0)).astype(np.uint8)

        progress_bar = st.progress(0)
        
        for idx, rx_file in enumerate(uploaded_rx):
            # 1. Image
            img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
            H, W = img_gray.shape

            # 2. Synchronisation précise du masque pour CHAQUE image
            m_green_res = cv2.resize(m_green_base, (W, H), interpolation=cv2.INTER_NEAREST)
            m_black_res = cv2.resize(m_black_base, (W, H), interpolation=cv2.INTER_NEAREST)
            
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(m_green_res, M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(m_black_res, M, (W, H), flags=cv2.INTER_NEAREST)
            
            # Zone d'inspection = Vert SANS le Noir
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_total_px = np.sum(z_utile)

            # 3. IA
            features = engine.compute_features(img_gray)
            probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)

            # 4. Filtrage micro-bulles (fusion en Jaune si < 0.1%)
            void_mask = (pred_map == 0) & z_utile
            cnts, _ = cv2.findContours(void_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_voids = np.zeros((H, W), dtype=bool)
            for c in cnts:
                if (cv2.contourArea(c) / area_total_px * 100) >= 0.1:
                    cv2.drawContours(final_voids.view(np.uint8), [c], -1, 1, -1)

            final_solder = (z_utile) & (~final_voids)
            
            # 5. Void Majeur (Cyan)
            max_void_area = 0
            max_void_poly = None
            void_u8 = final_voids.astype(np.uint8)*255
            v_cnts, _ = cv2.findContours(void_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for vc in v_cnts:
                area = cv2.contourArea(vc)
                if area > max_void_area:
                    max_void_area = area
                    max_void_poly = vc

            # 6. Statistiques
            missing_pct = (np.sum(final_voids) / area_total_px * 100) if area_total_px > 0 else 0
            max_void_pct = (max_void_area / area_total_px * 100) if area_total_px > 0 else 0

            # 7. Overlay
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay[final_solder] = [255, 255, 0] # Jaune
            overlay[final_voids] = [255, 0, 0]    # Rouge
            if max_void_poly is not None:
                cv2.drawContours(overlay, [max_void_poly], -1, [0, 255, 255], 2)

            # 8. Archive
            _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            st.session_state.batch_history.append({
                "Fichier": rx_file.name,
                "Total_%": round(missing_pct, 2),
                "Void_Max_%": round(max_void_pct, 3),
                "img_bytes": img_jpg.tobytes()
            })
            progress_bar.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history).drop(columns=['img_bytes'])
    st.dataframe(df.style.apply(highlight_extremes, subset=['Total_%']), use_container_width=True)
    
    with st.expander("👁️ Résultats Visuels", expanded=True):
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            cols[i % 4].image(item['img_bytes'], caption=item['Fichier'])
