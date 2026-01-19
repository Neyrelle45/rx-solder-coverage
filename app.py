import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os

st.set_page_config(page_title="Analyse RX Soudure", layout="wide")

# --- Fonctions de Cache pour la performance ---
@st.cache_resource
def load_trained_model(file_upload):
    # On sauvegarde le modèle uploadé pour que joblib puisse le lire proprement
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

# --- Interface Latérale ---
st.sidebar.title("🔍 Contrôles")
model_file = st.sidebar.file_uploader("1. Charger le modèle (.joblib)", type=["joblib"])
mode_selection = st.sidebar.radio("2. Mode d'analyse", ["Unitaire (Ajustement)", "Batch (Série)"])

if model_file:
    clf = load_trained_model(model_file)
    
    if mode_selection == "Unitaire (Ajustement)":
        st.header("🎯 Analyse Unitaire & Réglage du Masque")
        
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            rx_upload = st.file_uploader("Image RX (Composant)", type=["png", "jpg", "jpeg", "bmp"])
        with col_u2:
            mask_upload = st.file_uploader("Masque d'inspection", type=["png", "jpg", "jpeg"])

        if rx_upload and mask_upload:
            # --- CORRECTION : Sauvegarde temporaire pour OpenCV ---
            with open("temp_rx.png", "wb") as f:
                f.write(rx_upload.getbuffer())
            with open("temp_mask.png", "wb") as f:
                f.write(mask_upload.getbuffer())

            # Lecture via le moteur (en utilisant les chemins des fichiers créés ci-dessus)
            img_gray = engine.load_gray("temp_rx.png")
            H, W = img_gray.shape

            # --- Réglages d'alignement avec pas de 1 pixel ---
            st.sidebar.subheader("🕹️ Alignement du Masque")
            tx = st.sidebar.number_input("Translation X (px)", value=0.0, step=1.0)
            ty = st.sidebar.number_input("Translation Y (px)", value=0.0, step=1.0)
            rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0, step=0.5)
            scale = st.sidebar.slider("Échelle", 0.5, 1.5, 1.0, step=0.005)

            # Calcul du masque de base
            zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")
            if zone_base.shape != (H, W):
                zone_base = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)

            # Application de la transformation (logique de votre script original)
            ys, xs = np.where(zone_base > 0)
            cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
            M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
            zone_adj = cv2.warpAffine(zone_base, M, (W, H), flags=cv2.INTER_NEAREST)

            # Prédiction IA
            with st.spinner("Analyse des pixels en cours..."):
                feats = engine.compute_features(img_gray)
                pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                pred_bin = (pred == 1).astype(np.uint8) * 255

            # --- Calculs détaillés ---
            total_zone_px = int(np.sum(zone_adj > 0))
            solder_found_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
            void_px = total_zone_px - solder_found_px
            missing_pct = (void_px / total_zone_px * 100) if total_zone_px > 0 else 0

            # --- Affichage des résultats ---
            st.divider()
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                # Création de l'overlay : Jaune (Soudure) / Rouge (Vide)
                overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                overlay[(zone_adj > 0) & (pred_bin > 0)] = [0, 255, 255] # Jaune
                overlay[(zone_adj > 0) & (pred_bin == 0)] = [0, 0, 255]   # Rouge
                st.image(overlay, caption="Résultat visuel (Jaune=Soudure, Rouge=Manque)", use_container_width=True)

            with res_col2:
                st.subheader("📊 Détails")
                st.metric("Taux de Manque", f"{missing_pct:.2f} %")
                with st.expander("Détail du calcul", expanded=True):
                    st.write(f"Pixels zone utile : `{total_zone_px}`")
                    st.write(f"Pixels soudure : `{solder_found_px}`")
                    st.write(f"Pixels vides : `{void_px}`")

    else:
        st.header("📦 Mode Batch - Analyse de série")
        rx_uploads = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
        mask_upload = st.file_uploader("Masque commun", type=["png", "jpg", "jpeg"])

        if rx_uploads and mask_upload:
            if st.button("Lancer l'analyse groupée"):
                # Sauvegarde du masque commun une seule fois
                with open("temp_mask.png", "wb") as f:
                    f.write(mask_upload.getbuffer())
                
                all_results = []
                progress_bar = st.progress(0)
                zone_base, _, _ = engine.compute_zone_and_holes("temp_mask.png")

                for i, rx_up in enumerate(rx_uploads):
                    # Sauvegarde temporaire de l'image courante
                    with open("temp_batch_rx.png", "wb") as f:
                        f.write(rx_up.getbuffer())
                    
                    img_gray = engine.load_gray("temp_batch_rx.png")
                    H, W = img_gray.shape
                    zone_curr = cv2.resize(zone_base, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    feats = engine.compute_features(img_gray)
                    pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                    pred_bin = (pred == 1).astype(np.uint8) * 255
                    
                    total_px = int(np.sum(zone_curr > 0))
                    solder_px = int(np.sum((pred_bin > 0) & (zone_curr > 0)))
                    void_pct = (1 - (solder_px / total_px)) * 100 if total_px > 0 else 0
                    
                    all_results.append({
                        "Fichier": rx_up.name,
                        "Surface (px)": total_px,
                        "Soudure (px)": solder_px,
                        "Manque (%)": round(void_pct, 2)
                    })
                    progress_bar.progress((i + 1) / len(rx_uploads))

                df = pd.DataFrame(all_results)
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Télécharger les résultats (CSV)", df.to_csv(index=False), "resultats.csv", "text/csv")
else:
    st.info("Veuillez charger votre modèle .joblib pour commencer.")
