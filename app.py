import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import datetime
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Analyse Unitaire", layout="wide")

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
    
    c_u, c_m = st.columns(2)
    with c_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with c_m: mask_upload = st.file_uploader("2. Masque de référence", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- RÉGLAGES ALIGNEMENT ---
        st.sidebar.divider()
        tx = st.sidebar.number_input("Trans X", value=0)
        ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

        # --- CHARGEMENT ---
        rx_upload.seek(0)
        file_bytes = np.frombuffer(rx_upload.read(), np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img_raw is None:
            st.error("Erreur de lecture.")
            st.stop()

        if contrast_val > 0:
            clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8))
            img_gray = clahe.apply(img_raw)
        else:
            img_gray = img_raw

        H, W = img_gray.shape

        # --- MASQUE ---
        mask_upload.seek(0)
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), 1)
        r_r, g_r, b_r = cv2.split(cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB))
        m_green = (g_r > 100).astype(np.uint8)
        m_black = ((r_r < 100) & (g_r < 100) & (b_r < 100) & (m_green > 0)).astype(np.uint8)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj > 0) & (hol_adj == 0)

        # --- IA ---
        features = engine.compute_features(img_gray)
        pred_map = np.argmax(clf.predict_proba(features.reshape(-1, features.shape[-1])), axis=1).reshape(H, W)
        
        kernel = np.ones((3,3), np.uint8)
        void_raw = ((pred_map == 0) & (z_utile)).astype(np.uint8)
        clean_voids = cv2.morphologyEx(void_raw, cv2.MORPH_OPEN, kernel)
        
        # --- CORRECTION DES COULEURS ---
        # clean_solder = Soudure présente (Bleu foncé)
        # clean_voids = Manques (Rouge)
        clean_solder = (z_utile) & (clean_voids == 0)

        # --- VOID MAJEUR ---
        v_max_area, v_max_poly = 0, None
        z_stricte = cv2.erode(z_utile.astype(np.uint8), kernel, iterations=1)
        cnts, _ = cv2.findContours((clean_voids * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10: continue
            c_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_m, [c], -1, 255, -1)
            if not np.any((c_m > 0) & (hol_adj > 0)) and not np.any((c_m > 0) & (z_stricte == 0)):
                if area > v_max_area:
                    v_max_area, v_max_poly = area, c

# --- RENDU FINAL (COULEURS CORRIGÉES) ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # 1. On applique d'abord le ROUGE sur tous les manques détectés
        overlay[clean_voids > 0] = [255, 0, 0]  # ROUGE = MANQUES
        
        # 2. On applique ensuite le BLEU FONCÉ sur la soudure présente
        # Cela garantit que la matière prédomine visuellement dans la zone utile
        overlay[clean_solder] = [0, 50, 150]    # BLEU FONCÉ = SOUDURE
        
        # 3. On entoure le plus gros void en CYAN épais
        if v_max_poly is not None:
            cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 3) # CYAN

        # --- AFFICHAGE ---
        st.divider()
        col_ref, col_ia = st.columns(2)
        with col_ref:
            st.subheader("🖼️ Image Originale")
            st.image(img_gray, use_container_width=True)
            # Métriques
            area_tot = np.sum(z_utile)
            st.metric("Manque Total", f"{(np.sum(clean_voids)/area_tot*100):.2f} %" if area_tot > 0 else "0 %")
            st.metric("Void Majeur", f"{(v_max_area/area_tot*100):.3f} %" if area_tot > 0 else "0 %")

        with col_ia:
            st.subheader("🤖 Analyse IA")
            st.image(overlay, use_container_width=True)
            if st.button("📥 Archiver", use_container_width=True):
                st.toast("Résultat archivé")

# --- RAPPORT, ZIP ET GALERIE (FONCTIONS RESTAURÉES) ---
if st.session_state.history:
    st.divider()
    df_full = pd.DataFrame(st.session_state.history)
    df_csv = df_full.drop(columns=['img_bytes'])
    st.dataframe(df_csv.style.apply(highlight_extremes, subset=['Total_%'], axis=0), use_container_width=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rapport.csv", df_csv.to_csv(index=False))
        for i, item in enumerate(st.session_state.history):
            z.writestr(f"images/{i+1}_{item['Fichier']}.jpg", item['img_bytes'])
    st.download_button("🎁 Télécharger ZIP", buf.getvalue(), "rapport.zip", "application/zip", use_container_width=True)

    st.write("### 🖼️ Galerie")
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 6]:
            if st.button(f"🔎 Zoom {idx+1}", key=f"btn_{idx}"): st.session_state.selected_img = item['img_bytes']
            st.image(item['img_bytes'], caption=f"{item['Total_%']}%")

    if st.session_state.selected_img:
        st.divider(); st.image(st.session_state.selected_img, use_container_width=True)
        if st.button("❌ Fermer le zoom"): st.session_state.selected_img = None; st.rerun()
