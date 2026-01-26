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

        # --- ANALYSE IA INITIALE ---
        features = engine.compute_features(img_gray)
        probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
        pred_map = np.argmax(probs, axis=1).reshape(H, W)
        
        # CALCUL CONFIANCE IA
        conf_map = np.max(probs, axis=1).reshape(H, W)
        mean_conf = np.mean(conf_map[z_utile > 0]) * 100 if np.any(z_utile) else 0

        # --- LOGIQUE DE RECLASSIFICATION DES PISTES ---
        raw_voids = (pred_map == 0) & (z_utile > 0)
        void_cnts, _ = cv2.findContours(raw_voids.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        corrected_solder_mask = (pred_map == 1) & (z_utile > 0)
        
        for v_cnt in void_cnts:
            area = cv2.contourArea(v_cnt)
            if area < 20: continue
            
            perimeter = cv2.arcLength(v_cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            rect = cv2.minAreaRect(v_cnt)
            (w_r, h_r) = rect[1]
            aspect_ratio = max(w_r, h_r) / (min(w_r, h_r) + 1e-5)
            
            # Si c'est une piste (allongée), on l'ajoute au masque de soudure (Jaune)
            if circularity < 0.25 or aspect_ratio > 3.0:
                cv2.drawContours(corrected_solder_mask.astype(np.uint8), [v_cnt], -1, 1, -1)

        # --- DÉTECTION DES VOIDS DANS LA SOUDURE CORRIGÉE ---
        final_voids_mask = np.zeros((H, W), dtype=np.uint8)
        max_void_area = 0
        max_void_poly = None

        solder_u8 = (corrected_solder_mask > 0).astype(np.uint8) * 255
        solder_cnts, _ = cv2.findContours(solder_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for s_cnt in solder_cnts:
            s_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(s_mask, [s_cnt], -1, 255, -1)
            
            inverted_s_mask = cv2.bitwise_not(solder_u8)
            holes_in_island = cv2.bitwise_and(s_mask, inverted_s_mask)
            h_cnts, _ = cv2.findContours(holes_in_island, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for h_cnt in h_cnts:
                h_area = cv2.contourArea(h_cnt)
                if h_area < 10: continue
                
                h_m = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(h_m, [h_cnt], -1, 255, -1)
                if np.any((h_m > 0) & (hol_adj > 0)): continue
                
                final_voids_mask[h_m > 0] = 255
                if h_area > max_void_area:
                    max_void_area = h_area
                    max_void_poly = h_cnt

        # --- CALCULS FINAUX ---
        area_total_px = np.sum(z_utile > 0)
        missing_pct = (np.sum(final_voids_mask > 0) / area_total_px * 100.0) if area_total_px > 0 else 0
        max_void_pct = (max_void_area / area_total_px * 100) if area_total_px > 0 else 0

        # --- AFFICHAGE ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[corrected_solder_mask > 0] = [255, 255, 0] # Soudure + Pistes en Jaune
        overlay[final_voids_mask > 0] = [255, 0, 0]        # Voids réels en Rouge
        if max_void_poly is not None:
            cv2.drawContours(overlay, [max_void_poly], -1, [0, 255, 255], 3)

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
                })
                st.toast("Archivé")
        with c_img: st.image(overlay, use_container_width=True)

# --- RAPPORT ET GALERIE ---
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
        st.divider()
        st.image(st.session_state.selected_img, use_container_width=True)
        if st.button("❌ Fermer le zoom"): 
            st.session_state.selected_img = None
            st.rerun()
