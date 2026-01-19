import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine
import os

st.set_page_config(page_title="Analyse RX Soudure & Voids", layout="wide")

@st.cache_resource
def load_trained_model(file_upload):
    with open("temp_model.joblib", "wb") as f:
        f.write(file_upload.getbuffer())
    return joblib.load("temp_model.joblib")

st.sidebar.title("🔍 Configuration")
model_file = st.sidebar.file_uploader("Charger le modèle (.joblib)", type=["joblib"])
mode_selection = st.sidebar.radio("Mode d'analyse", ["Unitaire (Ajustement)", "Batch (Série)"])

if model_file:
    clf = load_trained_model(model_file)
    
    if mode_selection == "Unitaire (Ajustement)":
        st.header("🎯 Analyse et Métriques de Voids")
        
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            rx_upload = st.file_uploader("Image RX", type=["png", "jpg", "jpeg", "bmp"])
        with col_u2:
            mask_upload = st.file_uploader("Masque d'inspection", type=["png", "jpg", "jpeg"])

        if rx_upload and mask_upload:
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

            with st.spinner("Analyse IA..."):
                feats = engine.compute_features(img_gray)
                pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(H, W)
                pred_bin = (pred == 1).astype(np.uint8) * 255

            # --- ANALYSE DES VOIDS INTERNES ---
            # Un void est un manque (pred_bin == 0) à l'intérieur de la zone (zone_adj > 0)
            lack_of_solder = ((zone_adj > 0) & (pred_bin == 0)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(lack_of_solder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            voids_found = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 5: continue 
                
                # CRITÈRE : Le void ne doit pas toucher le bord noir du masque (zone_adj == 0)
                test_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(test_mask, [cnt], -1, 255, 1) # Dessine le bord du manque
                if np.any((test_mask > 0) & (zone_adj == 0)):
                    continue # C'est un manque ouvert sur l'extérieur, on l'ignore
                
                voids_found.append({"area": area, "contour": cnt})

            voids_found = sorted(voids_found, key=lambda x: x['area'], reverse=True)

            # --- VISUALISATION ---
            overlay_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            # Couleurs de base : JAUNE = Soudure, ROUGE = Manque
            overlay_rgb[(zone_adj > 0) & (pred_bin > 0)] = [255, 255, 0]
            overlay_rgb[(zone_adj > 0) & (pred_bin == 0)] = [255, 0, 0]

            # Mise en avant des Voids (Cyan)
            for i, v in enumerate(voids_found[:5]):
                if i == 0:
                    cv2.drawContours(overlay_rgb, [v['contour']], -1, [0, 255, 255], 3) # Contour épais pour le #1
                else:
                    cv2.drawContours(overlay_rgb, [v['contour']], -1, [0, 255, 255], 1)

            # --- AFFICHAGE ET TABLEAUX ---
            st.divider()
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                st.image(overlay_rgb, caption="Analyse : Jaune=Soudure, Rouge=Manque, Cyan=Voids", use_container_width=True)
                
                result_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.png', result_bgr)
                st.download_button("📥 Télécharger l'image", buffer.tobytes(), f"{os.path.splitext(rx_upload.name)[0]}_measured.png")

            with res_col2:
                total_px = int(np.sum(zone_adj > 0))
                solder_px = int(np.sum((pred_bin > 0) & (zone_adj > 0)))
                missing_px = total_px - solder_px
                missing_pct = (missing_px / total_px * 100) if total_px > 0 else 0
                
                st.subheader("📊 Taux de Manque Global")
                metrics_df = pd.DataFrame({
                    "Description": ["Surface totale zone", "Surface soudure détectée", "Surface manque total"],
                    "Pixels": [total_px, solder_px, missing_px]
                })
                st.table(metrics_df)
                st.metric("Taux de Manque Total", f"{missing_pct:.2f} %")

                st.divider()
                st.subheader("🔝 Top 5 Voids Internes")
                if voids_found:
                    void_data = []
                    for i, v in enumerate(voids_found[:5]):
                        v_pct = (v['area'] / total_px * 100)
                        void_data.append({"Rang": i+1, "Surface (px)": int(v['area']), "% Zone": round(v_pct, 2)})
                    st.table(pd.DataFrame(void_data))
                else:
                    st.info("Aucun void interne fermé détecté.")
else:
    st.info("Veuillez charger votre modèle .joblib.")
