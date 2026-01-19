import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os

st.set_page_config(page_title="Analyse RX Soudure", layout="wide")

@st.cache_resource
def load_trained_model(file_upload):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

st.sidebar.title("🛠 Configuration")
model_file = st.sidebar.file_uploader("1. Charger le modèle (.joblib)", type=["joblib"])
mode_selection = st.sidebar.radio("2. Choisir le mode", ["Mode Unitaire (Réglage)", "Mode Batch (Série)"])

if model_file:
    clf = load_trained_model(model_file)
    
    if mode_selection == "Mode Unitaire (Réglage)":
        st.header("🎯 Mode Unitaire - Ajustement")
        col1, col2 = st.columns(2)
        with col1:
            rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg"])
        with col2:
            mask_upload = st.file_uploader("Masque Unique", type=["png", "jpg", "jpeg"])

        if rx_upload and mask_upload:
            # On utilise une sauvegarde temporaire pour éviter les erreurs de flux binaires
            with open("temp_rx.png", "wb") as f:
                f.write(rx_upload.getbuffer())
            with open("temp_mask.png", "wb") as f:
                f.write(mask_upload.getbuffer())

            img_gray = engine.load_gray("temp_rx.png")
            H, W = img_gray.shape

            st.sidebar.subheader("Alignement")
            tx = st.sidebar.number_input("Trans X", value=0.0)
            ty = st.sidebar.number_input("Trans Y", value=0.0)
            rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
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

            zone_px = np.sum(zone_adj > 0)
            solder_px = np.sum((pred_bin > 0) & (zone_adj > 0))
            missing_pct = (1 - (solder_px / zone_px)) * 100 if zone_px > 0 else 0

            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            overlay[(zone_adj > 0) & (pred_bin > 0)] = [0, 255, 255]
            overlay[(zone_adj > 0) & (pred_bin == 0)] = [0, 0, 255]
            
            st.image(overlay, caption=f"Manque : {missing_pct:.2f}%", use_container_width=True)
            st.metric("Taux de vide", f"{missing_pct:.2f} %")

    else: # C'est ici que l'erreur d'indentation se trouvait
        st.header("📦 Mode Batch - Analyse en série")
        rx_uploads = st.file_uploader("Images RX", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        mask_upload = st.file_uploader("Masque commun", type=["png", "jpg", "jpeg"])

        if rx_uploads and mask_upload:
            if st.button("Lancer l'analyse"):
                all_results = []
                progress_bar = st.progress(0)
                with open("temp_mask.png", "wb") as f:
                    f.write(mask_upload.getbuffer())
                zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")

                for i, rx_up in enumerate(rx_uploads):
                    file_bytes = np.asarray(bytearray(rx_up.read()), dtype=np.uint8)
                    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    H, W = img_gray.shape
                    zone_curr = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    feats = engine.compute_features(img_gray)
                    pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                    pred_bin = (pred == 1).astype(np.uint8) * 255
                    
                    zone_px = np.sum(zone_curr > 0)
                    solder_px = np.sum((pred_bin > 0) & (zone_curr > 0))
                    missing_pct = (1 - (solder_px / zone_px)) * 100 if zone_px > 0 else 0
                    
                    all_results.append({"Fichier": rx_up.name, "Manque (%)": round(missing_pct, 2)})
                    progress_bar.progress((i + 1) / len(rx_uploads))

                df = pd.DataFrame(all_results)
                st.table(df)
                st.download_button("Télécharger CSV", df.to_csv(index=False), "resultats.csv")
else:
    st.info("Veuillez charger votre modèle pour commencer.")
