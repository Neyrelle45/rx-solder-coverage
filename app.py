import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import datetime
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Analyse Unitaire", layout="wide")

# Initialisation historique
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🛠️ Configuration")
model_file = st.sidebar.file_uploader("1. Charger modèle (.joblib)", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

if model_file:
    @st.cache_resource
    def load_my_model(file): return joblib.load(file)
    clf = load_my_model(model_file)

    st.header("🔍 Analyse Comparative")
    
    # 1. Zone d'Upload
    c_u, c_m = st.columns(2)
    with c_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with c_m: mask_upload = st.file_uploader("2. Masque de référence", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # 2. Réglages Alignement (Sidebar)
        st.sidebar.divider()
        tx = st.sidebar.number_input("Trans X", value=0)
        ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

        # 3. Chargement des images
        rx_upload.seek(0)
        file_bytes = np.frombuffer(rx_upload.read(), np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img_raw is None:
            st.error("Erreur de lecture de l'image.")
            st.stop()

        # Application du contraste
        if contrast_val > 0:
            clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8))
            img_gray = clahe.apply(img_raw)
        else:
            img_gray = img_raw

        H, W = img_gray.shape

        # 4. Traitement Masque
        mask_upload.seek(0)
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), 1)
        # On sépare les canaux RGB pour extraire Vert et Noir
        insp_rgb = cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB)
        r_r, g_r, b_r = cv2.split(insp_rgb)
        
        m_green = (g_r > 100).astype(np.uint8)
        m_black = ((r_r < 100) & (g_r < 100) & (b_r < 100) & (m_green > 0)).astype(np.uint8)
        
        # Alignement Matrix
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj > 0) & (hol_adj == 0)

        # 5. Analyse IA
        features = engine.compute_features(img_gray)
        probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
        pred_map = np.argmax(probs, axis=1).reshape(H, W)
        
        # Extraction Manques (Classe 0 dans tes labels)
        void_raw = ((pred_map == 0) & (z_utile)).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        clean_voids = cv2.morphologyEx(void_raw, cv2.MORPH_OPEN, kernel)
        
        # Extraction Soudure (Classe 1 dans tes labels)
        clean_solder = (z_utile) & (clean_voids == 0)

        # 6. Recherche du Void Majeur (Cyan)
        v_max_area, v_max_poly = 0, None
        z_stricte = cv2.erode(z_utile.astype(np.uint8), kernel, iterations=1)
        cnts, _ = cv2.findContours((clean_voids * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10: continue
            c_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_m, [c], -1, 255, -1)
            # Exclusion bords/vias
            if not np.any((c_m > 0) & (hol_adj > 0)) and not np.any((c_m > 0) & (z_stricte == 0)):
                if area > v_max_area:
                    v_max_area, v_max_poly = area, c

        # 7. Création de l'Overlay (COULEURS CORRIGÉES)
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[clean_solder] = [0, 50, 150]    # SOUDURE = BLEU FONCÉ
        overlay[clean_voids > 0] = [255, 0, 0]  # MANQUES = ROUGE
        if v_max_poly is not None:
            cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 3) # CYAN

        # 8. Affichage comparatif (UN SEUL BLOC DE COLONNES)
        st.divider()
        col_ref, col_ia = st.columns(2)
        
        with col_ref:
            st.subheader("🖼️ Image Originale (Réf)")
            st.image(img_gray, use_container_width=True)
            
            # Métriques
            area_tot = np.sum(z_utile)
            m_total = (np.sum(clean_voids)/area_tot*100) if area_tot > 0 else 0
            m_void = (v_max_area/area_tot*100) if area_tot > 0 else 0
            
            st.metric("Manque Total (Rouge)", f"{m_total:.2f} %")
            st.metric("Void Majeur (Cyan)", f"{m_void:.3f} %")

        with col_ia:
            st.subheader("🤖 Analyse IA")
            st.image(overlay, use_container_width=True)
            
            # Bouton avec clé unique pour éviter l'erreur de duplication
            if st.button("📥 Archiver le résultat", key="btn_archive", use_container_width=True):
                st.session_state.history.append({
                    "Fichier": rx_upload.name, 
                    "Total_%": round(m_total, 2),
                    "Void_Max_%": round(m_void, 3), 
                    "Heure": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.toast("Résultat archivé !")

# 9. Historique en bas de page
if st.session_state.history:
    with st.expander("📜 Historique des analyses"):
        st.table(pd.DataFrame(st.session_state.history))
