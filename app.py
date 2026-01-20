import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import analyse_rx_soudure as engine  # Import de ton script de 800 lignes

st.set_page_config(page_title="Station Analyse RX - Expert Voids", layout="wide")

# --- Initialisation de l'historique de session ---
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🛠️ Configuration")

# 1. RÉGLAGES IMAGE
st.sidebar.markdown("### 👁️ Visibilité & Contraste")
contrast_val = st.sidebar.slider(
    "Contraste (CLAHE)", 
    0.0, 10.0, 2.0, 0.5,
    help="Ajuste la visibilité des détails. 2.0 est la valeur par défaut."
)

st.sidebar.divider()

# 2. CHARGEMENT DU MODÈLE IA
st.sidebar.subheader("🧠 Intelligence Artificielle")
model_file = st.sidebar.file_uploader("Charger modèle (.joblib)", type=["joblib"])

if model_file:
    clf = joblib.load(model_file)
    st.sidebar.success("Modèle opérationnel")

    st.header("🔍 Analyse de Soudure & Voids Internes")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("1. Image RX (JPG/PNG/TIF)", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- CHARGEMENT ET PRÉPARATION ---
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # --- ALIGNEMENT DU MASQUE ---
        st.sidebar.subheader("🕹️ Alignement du Masque")
        tx = st.sidebar.number_input("Translation X (px)", value=0.0)
        ty = st.sidebar.number_input("Translation Y (px)", value=0.0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        # Extraction de la zone utile via ton moteur (Vert - Trous Noirs)
        with open("temp_app_mask.png", "wb") as f:
            f.write(mask_upload.getbuffer())
        
        zone_utile, centers, radii = engine.compute_zone_and_holes("temp_app_mask.png")
        if zone_utile.shape != (H, W):
            zone_utile = cv2.resize(zone_utile, (W, H), interpolation=cv2.INTER_NEAREST)

        # Application de la transformation
        cx, cy = (W/2, H/2)
        M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
        zone_adj = cv2.warpAffine(zone_utile, M, (W, H), flags=cv2.INTER_NEAREST)

        # --- PRÉDICTION IA ---
        with st.spinner("Analyse des pixels..."):
            features = engine.compute_features(img_gray)
            flat_features = features.reshape(-1, features.shape[-1])
            pred_flat = clf.predict(flat_features)
            pred_map = pred_flat.reshape(H, W)

        # --- CALCULS GÉOMÉTRIQUES STRICTS ---
        # 1. Masques de base
        valid_solder = (pred_map == 1) & (zone_adj > 0)
        valid_voids_all = (pred_map == 0) & (zone_adj > 0)
        
        area_total_px = np.sum(zone_adj > 0)
        area_solder_px = np.sum(valid_solder)
        missing_pct_global = (1.0 - (area_solder_px / area_total_px)) * 100.0 if area_total_px > 0 else 0

        # 2. FILTRAGE DES VOIDS (INTERNES UNIQUEMENT)
        void_mask_u8 = (valid_voids_all.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(void_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        internal_voids = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 2.0: continue # Filtre anti-bruit
            
            # Test d'inclusion : le contour touche-t-il le bord du masque ?
            c_mask = np.zeros(void_mask_u8.shape, dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, 1)
            # Dilatation pour vérifier les pixels adjacents au contour
            dilated = cv2.dilate(c_mask, np.ones((3,3), np.uint8), iterations=1)
            
            # Si le contour dilaté touche une zone hors-masque (zone_adj == 0), il n'est pas interne
            touches_edge = np.any((dilated > 0) & (zone_adj == 0))
            
            if not touches_edge:
                internal_voids.append({'area': area, 'poly': c})
        
        # Tri par taille
        internal_voids = sorted(internal_voids, key=lambda x: x['area'], reverse=True)
        top_5 = internal_voids[:5]

        # --- GÉNÉRATION DE L'OVERLAY ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Jaune : Soudure présente
        overlay[valid_solder] = [255, 255, 0] 
        # Rouge : Manque (périphérique + interne)
        overlay[valid_voids_all] = [255, 0, 0]
        
        # Bleu Ciel (Cyan) : Surlignage des 5 plus gros Voids internes
        cyan_color = [0, 255, 255]
        for v in top_5:
            cv2.drawContours(overlay, [v['poly']], -1, cyan_color, 2)

        # --- INTERFACE RÉSULTATS ---
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric("Taux de Manque Global", f"{missing_pct_global:.2f} %")
            st.write("🔍 **Top 5 Voids (Internes)**")
            void_stats = {}
            for i in range(5):
                val = (top_5[i]['area'] / area_total_px * 100) if i < len(top_5) else 0.0
                st.caption(f"Void {i+1} : {val:.3f} %")
                void_stats[f"Void_{i+1}_pct"] = round(val, 3)
            
            if st.button("📥 Archiver l'analyse"):
                data = {
                    "Fichier": rx_upload.name,
                    "Total_Manque_%": round(missing_pct_global, 2)
                }
                data.update(void_stats)
                st.session_state.history.append(data)
                st.success("C'est enregistré.")

        with col_res2:
            st.image(overlay, caption="Légende - Jaune: Soudure | Rouge: Manque | Cyan: Voids Internes", use_container_width=True)

# --- TABLEAU RÉCAPITULATIF ---
if st.session_state.history:
    st.divider()
    st.subheader("📊 Rapport de Session")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Télécharger Rapport CSV", csv, "rapport_rx.csv", "text/csv")
