import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import gc
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Void Majeur Enclavé", layout="wide")

# --- INITIALISATION ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = None

def highlight_extremes(s):
    if len(s) < 2: return [''] * len(s)
    is_max = s == s.max(); is_min = s == s.min()
    return ['background-color: #ffcccc' if v else 'background-color: #ccf2ff' if m else '' for v, m in zip(is_max, is_min)]

st.sidebar.title("🛠️ Configuration")
if st.sidebar.button("🗑️ Vider l'historique", use_container_width=True):
    st.session_state.history = []; st.session_state.selected_img = None

contrast_val = st.sidebar.slider("Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)
model_file = st.sidebar.file_uploader("Charger modèle (.joblib)", type=["joblib"])

if model_file:
    @st.cache_resource
    def load_my_model(file): return joblib.load(file)
    clf = load_my_model(model_file)

    st.header("🔍 Analyse du Void Majeur (Soudure Uniquement)")
    c_u, c_m = st.columns(2)
    with c_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with c_m: mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        tx = st.sidebar.number_input("Trans X", value=0); ty = st.sidebar.number_input("Trans Y", value=0)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0); sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # Traitement Masque
        insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), 1)
        b_r, g_r, r_r = cv2.split(insp_raw)
        m_green_orig = (g_r > 100).astype(np.uint8)
        m_black_orig = ((b_r < 50) & (g_r < 50) & (r_r < 50) & (m_green_orig > 0)).astype(np.uint8)
        
        m_green_res = cv2.resize(m_green_orig, (W, H), interpolation=cv2.INTER_NEAREST)
        m_black_res = cv2.resize(m_black_orig, (W, H), interpolation=cv2.INTER_NEAREST)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(m_green_res, M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(m_black_res, M, (W, H), flags=cv2.INTER_NEAREST)
        z_utile = (env_adj & ~hol_adj)





        
# 1. Analyse IA avec seuil de confiance ajustable
        features = engine.compute_features(img_gray)
        features_flat = features.reshape(-1, features.shape[-1])
        probs = clf.predict_proba(features_flat)
        
        # On extrait la probabilité de la classe "Manque" (Classe 1 selon vos labels)
        # Un seuil de 0.4 permet de détecter les petits points jaunes plus finement
        void_probs = probs[:, 1].reshape(H, W)
        raw_voids = np.where((z_utile > 0) & (void_probs > 0.4), 255, 0).astype(np.uint8)

        # 2. RAFFINEMENT DES MANQUES (Basé sur vos labels)
        # On utilise un noyau plus petit (5x5) pour garder la précision des petits points
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        voids_cleaned = cv2.morphologyEx(raw_voids, cv2.MORPH_OPEN, kernel) 
        voids_refined = cv2.morphologyEx(voids_cleaned, cv2.MORPH_CLOSE, kernel)

        # 3. CRÉATION DES MASQUES D'AFFICHAGE
        mask_bin = np.where(z_utile > 0, 255, 0).astype(np.uint8)
        
        # final_voids_mask = Les manques (sera affiché en ROUGE pour l'alerte)
        final_voids_mask = cv2.bitwise_and(voids_refined, voids_refined, mask=mask_bin)
        
        # solder_mask = La soudure (sera affichée en JAUNE pour la conformité)
        solder_mask = cv2.bitwise_and(mask_bin, cv2.bitwise_not(final_voids_mask))

        # 4. CALCULS PRÉCIS
        area_total_px = np.count_nonzero(mask_bin)
        area_voids_px = np.count_nonzero(final_voids_mask)
        missing_pct = (area_voids_px / area_total_px * 100) if area_total_px > 0 else 0
        
        # 5. OVERLAY STYLE "LABEL"
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Couleur Jaune pour la soudure (comme vos fonds de labels)
        overlay[solder_mask > 0] = [255, 255, 0] 
        # Couleur Rouge pour les manques (pour que ça saute aux yeux)
        overlay[final_voids_mask > 0] = [255, 0, 0]

        # 6. VOID MAJEUR (Calcul du plus gros point rouge)
        max_void_area = 0
        max_void_poly = None
        v_cnts, _ = cv2.findContours(final_voids_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for v_c in v_cnts:
            a = cv2.contourArea(v_c)
            if a > max_void_area:
                max_void_area = a
                max_void_poly = v_c
        max_void_pct = (max_void_area / area_total_px * 100) if area_total_px > 0 else 0




        
        # Affichage et Archivage
        st.divider()
        c_res, c_img = st.columns([1, 2])
        with c_res:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            st.metric("Void Majeur (Enclavé)", f"{max_void_pct:.3f} %")
            st.metric("Confiance IA", f"{mean_conf:.1f} %")
            if st.button("📥 Archiver", use_container_width=True):
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                st.session_state.history.append({
                    "Fichier": rx_upload.name, "Total_%": round(missing_pct, 2),
                    "Void_Max_%": round(max_void_pct, 3), "Confiance_%": round(mean_conf, 1),
                    "img_bytes": img_jpg.tobytes(), "Heure": datetime.datetime.now().strftime("%H:%M:%S")
                }); st.toast("Archivé")
        with c_img: st.image(overlay, use_container_width=True)

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
