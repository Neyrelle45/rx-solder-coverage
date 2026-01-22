import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Miroir", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

st.title("📦 Analyse de Série (Batch)")

# --- SIDEBAR IDENTIQUE ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement commun")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

# --- BOUTON CLEAR ---
if st.sidebar.button("🗑️ Vider l'historique des résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

# --- UPLOADS ---
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque Unique", type=["png", "jpg"])

if model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    
    if st.button("🚀 Lancer l'analyse complète", use_container_width=True):
        st.session_state.batch_history = []
        
        # Lecture brute du masque (méthode app.py)
        mask_bytes = mask_file.read()
        mask_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), 1)
        b_mask, g_mask, r_mask = cv2.split(mask_raw)
        
        progress_bar = st.progress(0)
        
        for idx, rx_file in enumerate(uploaded_rx):
            # 1. Image Grise + CLAHE
            img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
            H, W = img_gray.shape

            # 2. Masques (Stricte répétition de la logique app.py)
            m_green_res = cv2.resize((g_mask > 100).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            m_black_res = cv2.resize(((b_mask < 50) & (g_mask < 50) & (r_mask < 50)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(m_green_res, M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(m_black_res, M, (W, H), flags=cv2.INTER_NEAREST)
            
            # Exclusion forcée des vias
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_total_px = np.sum(z_utile)

            # 3. IA
            features = engine.compute_features(img_gray)
            probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)

            # 4. Filtrage micro-bulles (en jaune si < 0.1%)
            void_u8 = ((pred_map == 0) & z_utile).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(void_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_voids = np.zeros((H, W), dtype=bool)
            for c in cnts:
                area = cv2.contourArea(c)
                if area_total_px > 0 and (area / area_total_px * 100) >= 0.1:
                    cv2.drawContours(final_voids.view(np.uint8), [c], -1, 1, -1)

            # 5. Définition des couleurs
            # Soudure = (Zones marquées Solder par IA OU Micro-bulles) DANS zone utile
            display_solder = z_utile & (~final_voids)
            display_voids = final_voids

            # 6. Void Majeur (Cyan) - Logique Enclavement
            max_void_area = 0
            max_void_poly = None
            v_cnts, _ = cv2.findContours(display_voids.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for vc in v_cnts:
                area = cv2.contourArea(vc)
                if area > max_void_area:
                    max_void_area = area
                    max_void_poly = vc

            # 7. Stats
            missing_pct = (np.sum(display_voids) / area_total_px * 100) if area_total_px > 0 else 0
            v_max_pct = (max_void_area / area_total_px * 100) if area_total_px > 0 else 0

            # 8. Overlay
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay[display_solder] = [255, 255, 0] # Jaune
            overlay[display_voids] = [255, 0, 0]    # Rouge
            if max_void_poly is not None:
                cv2.drawContours(overlay, [max_void_poly], -1, [0, 255, 255], 2)

            # 9. Stockage
            _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            st.session_state.batch_history.append({
                "Fichier": rx_file.name,
                "Total_%": round(missing_pct, 2),
                "Void_Max_%": round(v_max_pct, 3),
                "img_bytes": img_jpg.tobytes()
            })
            progress_bar.progress((idx + 1) / len(uploaded_rx))

# --- SORTIE ET EXPORTS ---
if st.session_state.batch_history:
    st.divider()
    df = pd.DataFrame(st.session_state.batch_history).drop(columns=['img_bytes'])
    st.dataframe(df.style.apply(highlight_extremes, subset=['Total_%']), use_container_width=True)

    # RE-CRÉATION DU ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("resultats.csv", df.to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    
    col1, col2 = st.columns(2)
    col1.download_button("📥 ZIP (Images + CSV)", zip_buf.getvalue(), "batch.zip", "application/zip", use_container_width=True)
    col2.download_button("📄 CSV seul", df.to_csv(index=False), "rapport.csv", "text/csv", use_container_width=True)

    with st.expander("👁️ Résultats Visuels", expanded=True):
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            cols[i % 4].image(item['img_bytes'], caption=item['Fichier'])
