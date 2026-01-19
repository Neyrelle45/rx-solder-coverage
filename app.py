import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os
from io import BytesIO

st.set_page_config(page_title="Analyse RX Soudure", layout="wide")

@st.cache_resource
def load_trained_model(file_upload):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

st.sidebar.title("🔍 Configuration")
model_file = st.sidebar.file_uploader("1. Charger le modèle (.joblib)", type=["joblib"])
mode_selection = st.sidebar.radio("2. Mode d'analyse", ["Unitaire (Ajustement)", "Batch (Série)"])

if model_file:
    clf = load_trained_model(model_file)
    
    if mode_selection == "Unitaire (Ajustement)":
        st.header("🎯 Analyse Unitaire & Réglage")
        
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg", "bmp"])
        with col_u2:
            mask_upload = st.file_uploader("Masque", type=["png", "jpg", "jpeg"])

        if rx_upload and mask_upload:
            # Sauvegardes temporaires
            with open("temp_rx.png", "wb") as f: f.write(rx_upload.getbuffer())
            with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())

            img_gray = engine.load_gray("temp_rx.png")
            H, W = img_gray.shape

            # --- Réglages Step 1.0 ---
            st.sidebar.subheader("🕹️ Alignement")
            tx = st.sidebar.number_input("Translation X (px)", value=0.0, step=1.0)
            ty = st.sidebar.number_input("Translation Y (px)", value=0.0, step=1.0)
            rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0, step=0.5)
            scale = st.sidebar.slider("Échelle", 0.5, 1.5, 1.0, step=0.005)

            zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")
            if zone_base.shape != (H, W):
                zone_base = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.where(zone_base > 0)
            cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
            M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
            zone_adj = cv2.warpAffine(zone_base, M, (W, H), flags=cv2.INTER_NEAREST)

            with st.spinner("Analyse..."):
                feats = engine.compute_features(img_gray)
                pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                pred_bin = (pred == 1).astype(np.uint8) * 255

            # --- Calculs ---
            total_zone_px = int(np.sum(zone_adj > 0))
            solder_found_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
            void_px = total_zone_px - solder_found_px
            missing_pct = (void_px / total_zone_px * 100) if total_zone_px > 0 else 0

            # --- Création de l'image de résultat (Format RGB pour Streamlit) ---
            overlay_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            
            # JAUNE en RGB = [255, 255, 0]
            overlay_rgb[(zone_adj > 0) & (pred_bin > 0)] = [255, 255, 0]
            # ROUGE en RGB = [255, 0, 0]
            overlay_rgb[(zone_adj > 0) & (pred_bin == 0)] = [255, 0, 0]

            st.divider()
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                st.image(overlay_rgb, caption="Jaune = Soudure | Rouge = Manque", use_container_width=True)
                
                # --- Bouton de téléchargement de l'image ---
                # On reconvertit en BGR juste pour l'encodage de l'image de sortie
                result_img_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.png', result_img_bgr)
                
                base_name = os.path.splitext(rx_upload.name)[0]
                st.download_button(
                    label="💾 Télécharger l'image d'analyse",
                    data=buffer.tobytes(),
                    file_name=f"{base_name}_measured.png",
                    mime="image/png"
                )

            with res_col2:
                st.subheader("📊 Résultats")
                st.metric("Taux de Manque", f"{missing_pct:.2f} %")
                st.write(f"Pixels utiles : `{total_zone_px}`")
                st.write(f"Pixels soudure : `{solder_found_px}`")
                st.write(f"Pixels vides : `{void_px}`")

    else:
        st.header("📦 Mode Batch")
        rx_uploads = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
        mask_upload = st.file_uploader("Masque commun", type=["png", "jpg", "jpeg"])

        if rx_uploads and mask_upload:
            if st.button("Lancer l'analyse"):
                with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())
                all_results = []
                progress_bar = st.progress(0)
                zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")

                for i, rx_up in enumerate(rx_uploads):
                    with open("temp_batch.png", "wb") as f: f.write(rx_up.getbuffer())
                    img_gray = engine.load_gray("temp_batch.png")
                    H, W = img_gray.shape
                    zone_curr = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)
                    feats = engine.compute_features(img_gray)
                    pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                    pred_bin = (pred == 1).astype(np.uint8) * 255
                    
                    t_px = int(np.sum(zone_curr > 0))
                    s_px = int(np.sum((pred_bin > 0) & (zone_curr > 0)))
                    m_pct = (1 - (s_px / t_px)) * 100 if t_px > 0 else 0
                    
                    all_results.append({"Fichier": rx_up.name, "Manque (%)": round(m_pct, 2)})
                    progress_bar.progress((i + 1) / len(rx_uploads))

                df = pd.DataFrame(all_results)
                st.dataframe(df)
                st.download_button("📥 Télécharger CSV", df.to_csv(index=False), "resultats.csv")
else:
    st.info("Veuillez charger votre modèle .joblib pour commencer.")
