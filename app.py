import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import analyse_rx_soudure as engine # Ton script de 800 lignes

st.set_page_config(page_title="Station Analyse RX - ALL Circuits", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🛠️ Paramètres")

# 1. SLIDER CONTRASTE (Envoie la valeur à engine.load_gray)
contrast_val = st.sidebar.slider("Ajustement Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)

# 2. CHARGEMENT MODÈLE
model_file = st.sidebar.file_uploader("Charger le fichier .joblib", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    st.sidebar.success("Modèle prêt")

    st.header("🔍 Analyse de Soudure (Strictement limitée au masque)")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # CHARGEMENT RX via ton moteur
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # --- ALIGNEMENT MANUEL DU MASQUE ---
        st.sidebar.subheader("🕹️ Alignement du Masque")
        tx = st.sidebar.number_input("Translation X (px)", value=0.0)
        ty = st.sidebar.number_input("Translation Y (px)", value=0.0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        # Calcul de la zone utile à partir du fichier uploadé
        with open("temp_app_mask.png", "wb") as f:
            f.write(mask_upload.getbuffer())
        
        # Extraction zone utile (Vert moins trous noirs)
        zone_utile, centers, radii = engine.compute_zone_and_holes("temp_app_mask.png")
        if zone_utile.shape != (H, W):
            zone_utile = cv2.resize(zone_utile, (W, H), interpolation=cv2.INTER_NEAREST)

        # Application de la transformation gérée par ton script
        cx, cy = (W/2, H/2) # Centre par défaut pour la rotation
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_utile, M, (W, H), flags=cv2.INTER_NEAREST)

        # --- PRÉDICTION IA (Seulement 3 filtres pour correspondre à ton train si besoin) ---
        with st.spinner("Analyse IA en cours..."):
            # On utilise ta fonction de features
            features = engine.compute_features(img_gray)
            flat_features = features.reshape(-1, features.shape[-1])
            pred_flat = clf.predict(flat_features)
            pred_map = pred_flat.reshape(H, W)

        # ==========================================
        # CALCULS ET FILTRAGE PAR LE MASQUE (STRICT)
        # ==========================================
        # On ne garde que les pixels où le masque est présent (zone_adj > 0)
        valid_solder = (pred_map == 1) & (zone_adj > 0)
        valid_voids  = (pred_map == 0) & (zone_adj > 0)

        area_total_px = np.sum(zone_adj > 0)
        area_solder_px = np.sum(valid_solder)
        
        if area_total_px > 0:
            missing_pct = (1.0 - (area_solder_px / area_total_px)) * 100.0
        else:
            missing_pct = 0.0

        # --- DÉTECTION DES 5 PLUS GROS VOIDS ---
        # On travaille uniquement sur le masque des vides filtré
        void_mask_u8 = (valid_voids.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(void_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcul des aires et stockage pour tri
        void_list = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a > 0.5:
                void_list.append({'area': a, 'poly': c})
        
        void_list = sorted(void_list, key=lambda x: x['area'], reverse=True)
        top_5 = void_list[:5]

        # --- GÉNÉRATION DE L'OVERLAY ---
        # Base image RX
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Coloration : JAUNE pour OK, ROUGE pour VOID (uniquement dans le masque)
        overlay[valid_solder] = [255, 255, 0] # Jaune (BGR: 0, 255, 255 ?) Non, OpenCV RGB
        # Note : st.image attend du RGB. 
        # Jaune RGB = [255, 255, 0] | Rouge RGB = [255, 0, 0]
        overlay[valid_voids] = [255, 0, 0]
        
        # Surlignage Bleu Ciel (Cyan) pour les 5 plus gros
        cyan_rgb = [0, 255, 255]
        for i, v in enumerate(top_5):
            cv2.drawContours(overlay, [v['poly']], -1, cyan_rgb, 2)

        # --- AFFICHAGE ---
        st.divider()
        st.subheader("📊 Résultats de l'analyse")
        c1, c2 = st.columns(2)
        c1.metric("Manque de soudure Total", f"{missing_pct:.2f} %")
        c2.write("📏 **Top 5 des Voids (en % du masque)**")
        
        void_results_table = {}
        for i in range(5):
            val = (top_5[i]['area'] / area_total_px * 100) if i < len(top_5) else 0.0
            st.sidebar.text(f"Void {i+1}: {val:.3f}%")
            void_results_table[f"Void {i+1} (%)"] = round(val, 3)

        st.image(overlay, caption="JAUNE: OK | ROUGE: Voids | CYAN: Top 5 Voids", use_container_width=True)

        if st.button("💾 Archiver l'analyse"):
            res = {"Fichier": rx_upload.name, "Manque Global %": round(missing_pct, 2)}
            res.update(void_results_table)
            st.session_state.history.append(res)
            st.success("Données ajoutées au tableau.")

# TABLEAU FINAL
if st.session_state.history:
    st.divider()
    st.write("### 📋 Historique")
    st.table(pd.DataFrame(st.session_state.history))
