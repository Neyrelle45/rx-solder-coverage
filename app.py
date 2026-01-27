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

    st.header("🔍 Analyse du Void Majeur (Soudure Uniquement)")
    c_u, c_m = st.columns(2)
    with c_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with c_m: mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # Contrôles d'alignement
        st.sidebar.divider()
        tx = st.sidebar.number_input("Trans X", value=0)
        ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
        sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

        # Chargement et préparation
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # Traitement Masque
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), 1)
        insp_rgb = cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB)
        r_r, g_r, b_r = cv2.split(insp_rgb)
        
        m_green_orig = (g_r > 100).astype(np.uint8)
        m_black_orig = ((r_r < 100) & (g_r < 100) & (b_r < 100) & (m_green_orig > 0)).astype(np.uint8)
        
        # Alignement
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green_orig, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black_orig, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj > 0) & (hol_adj == 0)

        # --- ANALYSE IA ---
        features = engine.compute_features(img_gray)
        probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
        pred_map = np.argmax(probs, axis=1).reshape(H, W)
        
        # Confiance
        conf_map = np.max(probs, axis=1).reshape(H, W)
        mean_conf = np.mean(conf_map[z_utile]) * 100 if np.any(z_utile) else 0

        # Nettoyage Bruit (Opening 3x3)
        kernel = np.ones((3,3), np.uint8)
        void_raw = ((pred_map == 0) & (z_utile)).astype(np.uint8)
        clean_voids = cv2.morphologyEx(void_raw, cv2.MORPH_OPEN, kernel)
        
        # Logique Binaire : Tout ce qui est utile et non rouge est jaune
        clean_solder = (z_utile) & (clean_voids == 0)

        # --- CALCUL VOID MAJEUR ENCLAVÉ ---
        v_max_area, v_max_poly = 0, None
        red_u8 = (clean_voids * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(red_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Zone de bordure pour exclusion (Erosion de 3px)
        z_interne_stricte = cv2.erode(z_utile.astype(np.uint8), kernel, iterations=1)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10: continue
            
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            
            # Exclusion si touche Via (Noir) ou Bord (Hors zone interne)
            touches_via = np.any((c_mask > 0) & (hol_adj > 0))
            touches_bord = np.any((c_mask > 0) & (z_interne_stricte == 0))
            
            if not touches_via and not touches_bord:
                if area > v_max_area:
                    v_max_area = area
                    v_max_poly = c

        # Stats
        area_total_px = np.sum(z_utile)
        missing_pct = (np.sum(clean_voids) / area_total_px * 100) if area_total_px > 0 else 0
        max_void_pct = (v_max_area / area_total_px * 100) if area_total_px > 0 else 0

        # Overlay
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[clean_solder] = [255, 255, 0]
        overlay[clean_voids > 0] = [255, 0, 0]
        if v_max_poly is not None:
            cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 2)

        # Affichage
        st.divider()
        col_res, col_img = st.columns([1, 2])
        with col_res:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            st.metric("Void Majeur (Enclavé)", f"{max_void_pct:.3f} %")
            st.metric("Confiance IA", f"{mean_conf:.1f} %")
            if st.button("📥 Archiver", use_container_width=True):
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                st.session_state.history.append({
                    "Fichier": rx_upload.name, "Total_%": round(missing_pct, 2),
                    "Void_Max_%": round(max_void_pct, 3), "Heure": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.toast("Ajouté à l'historique")
        with col_img:
            st.image(overlay, use_container_width=True)

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
