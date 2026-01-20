import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine # Ton script de 800 lignes

st.set_page_config(page_title="Station Analyse RX - ALL Circuits", layout="wide")

# --- Initialisation de l'historique ---
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🛠️ Paramètres")

# 1. SLIDER CONTRASTE
st.sidebar.markdown("### 👁️ Visibilité")
contrast_val = st.sidebar.slider("Ajustement Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.divider()

# 2. CHARGEMENT MODÈLE
st.sidebar.subheader("🧠 Modèle IA")
model_file = st.sidebar.file_uploader("Charger le fichier .joblib", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    st.sidebar.success("Modèle prêt")

    st.header("🔍 Analyse de Soudure & Voids")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("Étape 1 : Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("Étape 2 : Masque", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # Lecture avec contraste dynamique
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        
        if img_gray is not None:
            H, W = img_gray.shape
            
            # --- ALIGNEMENT ---
            st.sidebar.subheader("🕹️ Alignement manuel")
            tx = st.sidebar.number_input("Translation X (px)", value=0.0)
            ty = st.sidebar.number_input("Translation Y (px)", value=0.0)
            rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
            scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

            with open("temp_mask.png", "wb") as f:
                f.write(mask_upload.getbuffer())

            zone_utile, green, black = engine.compute_zone_and_holes("temp_mask.png")
            if zone_utile.shape != (H, W):
                zone_utile = cv2.resize(zone_utile, (W, H), interpolation=cv2.INTER_NEAREST)

            # Transformation
            ys, xs = np.where(green > 0)
            cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
            M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
            zone_adj = cv2.warpAffine(zone_utile, M, (W, H), flags=cv2.INTER_NEAREST)

            # --- ANALYSE IA ---
            with st.spinner("Analyse..."):
                feats = engine.compute_features(img_gray)
                flat_feats = feats.reshape(-1, feats.shape[-1])
                pred_flat = clf.predict(flat_feats)
                pred_map = pred_flat.reshape(H, W)

            # --- CALCULS VOIDS ---
            total_zone_px = np.sum(zone_adj > 0)
            solder_px = np.sum((pred_map == 1) & (zone_adj > 0))
            missing_pct = 100 - (solder_px / total_zone_px * 100) if total_zone_px > 0 else 0

            # Extraction des contours des vides
            void_mask = np.zeros((H, W), dtype=np.uint8)
            void_mask[(zone_adj > 0) & (pred_map == 0)] = 255
            
            cnts, _ = cv2.findContours(void_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # On stocke les contours avec leur surface pour pouvoir les trier
            void_data = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 1: # On ignore les bruits de 1 pixel
                    void_data.append({'area': area, 'contour': c})
            
            # Tri par surface décroissante
            void_data = sorted(void_data, key=lambda x: x['area'], reverse=True)
            top_5_voids = void_data[:5]

            # --- VISUALISATION ---
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay[(zone_adj > 0) & (pred_map == 0)] = [255, 0, 0]    # Rouge pour tous les manques
            overlay[(zone_adj > 0) & (pred_map == 1)] = [255, 255, 0] # Jaune pour la soudure
            
            # Dessin des contours des 5 plus gros en BLEU CIEL (Cyan)
            cyan_color = (0, 255, 255) # BGR: 255 bleu, 255 vert, 0 rouge
            for i, vd in enumerate(top_5_voids):
                cv2.drawContours(overlay, [vd['contour']], -1, cyan_color, 2)
            
            st.divider()
            st.subheader("🎭 Résultat Visuel")
            st.image(overlay, caption="Jaune: OK | Rouge: Manque | Bleu Ciel: Top 5 Voids", use_container_width=True)

            # --- STATISTIQUES ---
            st.write("### 📏 Détails des 5 plus gros Voids")
            void_cols = st.columns(5)
            void_results_for_table = {}
            for i in range(5):
                val = (top_5_voids[i]['area'] / total_zone_px * 100) if i < len(top_5_voids) else 0.0
                void_cols[i].metric(f"Void {i+1}", f"{val:.2f} %")
                void_results_for_table[f"Void {i+1} (%)"] = round(val, 2)

            # --- ENREGISTRER ---
            if st.button("📥 Enregistrer dans le rapport"):
                entry = {
                    "Heure": pd.Timestamp.now().strftime("%H:%M"),
                    "Fichier": rx_upload.name,
                    "Manque Total %": round(missing_pct, 2)
                }
                entry.update(void_results_for_table)
                st.session_state.history.append(entry)
                st.success("Données archivées.")

# --- TABLEAU RÉCAPITULATIF ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Rapport d'Analyse Session")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Exporter CSV", csv, "rapport_voids_rx.csv", "text/csv")
