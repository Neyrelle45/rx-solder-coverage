import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
from PIL import Image

st.set_page_config(page_title="Analyse RX Soudure - Multi-Mode", layout="wide")

# Cache pour le modèle afin d'éviter de le recharger à chaque clic
@st.cache_resource
def load_trained_model(file_content):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_content.getbuffer())
    return joblib.load("temp_model.joblib")

# --- Interface Latérale ---
st.sidebar.title("🛠 Configuration")
model_file = st.sidebar.file_uploader("1. Charger le modèle (.joblib)", type=["joblib"])
mode_selection = st.sidebar.radio("2. Choisir le mode", ["Mode Unitaire (Réglage)", "Mode Batch (Série)"])

if model_file:
    clf = load_trained_model(model_file)
    
    if mode_selection == "Mode Unitaire (Réglage)":
        st.header("🎯 Mode Unitaire - Ajustement Précis")
        
        col1, col2 = st.columns(2)
        with col1:
            rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg"])
        with col2:
            mask_upload = st.file_uploader("Masque Unique", type=["png", "jpg", "jpeg"])

        if rx_upload and mask_upload:
            # Traitement image
               with open("temp_file.png", "wb") as f:
            f.write(rx_upload.getbuffer())

            img_gray = engine.load_gray("temp_file.png")
            H, W = img_gray.shape
            
            # Paramètres d'alignement manuel
            st.sidebar.subheader("Alignement du masque")
            tx = st.sidebar.number_input("Translation X (px)", value=0.0)
            ty = st.sidebar.number_input("Translation Y (px)", value=0.0)
            rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
            scale = st.sidebar.slider("Échelle", 0.5, 1.5, 1.0, step=0.005)

            # Calcul du masque
            with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())
            zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")
            if zone_base.shape != (H, W):
                zone_base = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)

            # Application de la transformation
            ys, xs = np.where(zone_base > 0)
            cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
            M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
            zone_adj = cv2.warpAffine(zone_base, M, (W, H), flags=cv2.INTER_NEAREST)

            # Inférence
            with st.spinner("Analyse..."):
                feats = engine.compute_features(img_gray)
                pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                pred_bin = (pred == 1).astype(np.uint8) * 255

            # Métriques
            zone_px = np.sum(zone_adj > 0)
            solder_px = np.sum((pred_bin > 0) & (zone_adj > 0))
            missing_pct = (1 - (solder_px / zone_px)) * 100 if zone_px > 0 else 0

            # Affichage
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            overlay[(zone_adj > 0) & (pred_bin > 0)] = [0, 255, 255] # Jaune
            overlay[(zone_adj > 0) & (pred_bin == 0)] = [0, 0, 255]   # Rouge
            
            st.image(overlay, caption=f"Manque : {missing_pct:.2f}%", use_container_width=True)
            st.metric("Taux de vide", f"{missing_pct:.2f} %")

    else:
        st.header("📦 Mode Batch - Analyse en série")
        rx_uploads = st.file_uploader("Sélectionnez vos images RX", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        mask_upload = st.file_uploader("Masque d'inspection (unique pour la série)", type=["png", "jpg", "jpeg"])

        if rx_uploads and mask_upload:
            if st.button("Lancer l'analyse de la série"):
                all_results = []
                progress_bar = st.progress(0)
                
                # Chargement du masque une seule fois
                with open("temp_mask.png", "wb") as f: f.write(mask_upload.getbuffer())
                zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")

                for i, rx_up in enumerate(rx_uploads):
                    # Lecture image
                    file_bytes = np.asarray(bytearray(rx_up.read()), dtype=np.uint8)
                    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    H, W = img_gray.shape
                    
                    # Redimensionnement du masque si nécessaire
                    zone_curr = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    # Inférence IA
                    feats = engine.compute_features(img_gray)
                    pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                    pred_bin = (pred == 1).astype(np.uint8) * 255
                    
                    # Calculs
                    zone_px = np.sum(zone_curr > 0)
                    solder_px = np.sum((pred_bin > 0) & (zone_curr > 0))
                    missing_pct = (1 - (solder_px / zone_px)) * 100 if zone_px > 0 else 0
                    
                    all_results.append({
                        "Fichier": rx_up.name,
                        "Surface Zone (px)": zone_px,
                        "Taux Manque (%)": round(missing_pct, 2)
                    })
                    progress_bar.progress((i + 1) / len(rx_uploads))

                # Affichage des résultats
                df = pd.DataFrame(all_results)
                st.success(f"Analyse de {len(rx_uploads)} images terminée.")
                st.dataframe(df, use_container_width=True)
                
                # Export CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger les résultats (CSV)", csv, "resultats_rx.csv", "text/csv")
else:
    st.info("Veuillez charger un modèle .joblib dans la barre latérale pour commencer.")
