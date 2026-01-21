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

st.set_page_config(page_title="RX Expert - Analyse du Void Majeur", layout="wide")

# --- INITIALISATION ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = None

# --- FONCTION DE COLORATION (Rouge=Pire, Bleu=Meilleur) ---
def highlight_extremes(s):
    if len(s) < 2: return [''] * len(s)
    is_max = s == s.max()
    is_min = s == s.min()
    return ['background-color: #ffcccc' if v else 'background-color: #ccf2ff' if m else '' for v, m in zip(is_max, is_min)]

st.sidebar.title("🛠️ Configuration")

if st.sidebar.button("🗑️ Vider l'historique", use_container_width=True):
    st.session_state.history = []
    st.session_state.selected_img = None
    st.sidebar.success("Historique vidé !")

st.sidebar.divider()

contrast_val = st.sidebar.slider("Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)
model_file = st.sidebar.file_uploader("Charger modèle (.joblib)", type=["joblib"])

if model_file:
    @st.cache_resource
    def load_my_model(file):
        return joblib.load(file)
    
    clf = load_my_model(model_file)
    st.sidebar.success("IA Opérationnelle")

    st.header("🔍 Analyse du Void Maximum Interne")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- ALIGNEMENT MANUEL ---
        st.sidebar.subheader("🕹️ Alignement")
        tx = st.sidebar.number_input("Translation X", value=0, step=1)
        ty = st.sidebar.number_input("Translation Y", value=0, step=1)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0, 0.5)
        sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # --- TRAITEMENT DU MASQUE ---
        mask_bytes = mask_upload.getvalue()
        insp_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), 1)
        b_r, g_r, r_r = cv2.split(insp_raw)
        
        m_green_orig = (g_r > 100).astype(np.uint8)
        m_black_orig = ((b_r < 50) & (g_r < 50) & (r_r < 50) & (m_green_orig > 0)).astype(np.uint8)
        
        m_green_res = cv2.resize(m_green_orig, (W, H), interpolation=cv2.INTER_NEAREST)
        m_black_res = cv2.resize(m_black_orig, (W, H), interpolation=cv2.INTER_NEAREST)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        z_utile = cv2.warpAffine((m_green_res & ~m_black_res), M, (W, H), flags=cv2.INTER_NEAREST)
        env_adj = cv2.warpAffine(m_green_res, M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(m_black_res, M, (W, H), flags=cv2.INTER_NEAREST)

        # --- ANALYSE IA ---
        with st.spinner("Calcul en cours..."):
            features = engine.compute_features(img_gray)
            probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)
            conf_map = np.max(probs, axis=1).reshape(H, W)
            mean_conf = np.mean(conf_map[z_utile > 0]) * 100 if np.any(z_utile) else 0

        # --- CALCULS ---
        valid_solder = (pred_map == 1) & (z_utile > 0)
        valid_voids = (pred_map == 0) & (z_utile > 0)
        area_total_px = np.sum(z_utile > 0)
        missing_pct = (1.0 - (np.sum(valid_solder) / area_total_px)) * 100.0 if area_total_px > 0 else 0

        # --- RECHERCHE DU VOID MAXIMUM STRICT ---
        v_mask_u8 = (valid_voids.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(v_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_void_area = 0
        max_void_poly = None

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 5.0: continue
            
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            
            # CONDITION 1 : Le void ne doit pas toucher les trous noirs (Vias)
            if np.any((c_mask > 0) & (hol_adj > 0)): continue
            
            # CONDITION 2 : Le void doit être entouré de soudure (pas toucher le bord vert)
            border_mask = cv2.dilate(c_mask, np.ones((3,3), np.uint8)) - c_mask
            if np.any((border_mask > 0) & (env_adj == 0)): continue

            if area > max_void_area:
                max_void_area = area
                max_void_poly = c

        max_void_pct = (max_void_area / area_total_px * 100) if area_total_px > 0 else 0

        # --- OVERLAY ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[valid_solder] = [255, 255, 0] # Jaune
        overlay[valid_voids] = [255, 0, 0]    # Rouge
        if max_void_poly is not None:
            cv2.drawContours(overlay, [max_void_poly], -1, [0, 255, 255], 3) # Cyan

        # --- AFFICHAGE ---
        st.divider()
        c_res, c_img = st.columns([1, 2])
        with c_res:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            st.metric("Void Majeur Interne", f"{max_void_pct:.3f} %")
            st.metric("Confiance IA", f"{mean_conf:.1f} %")
            
            if st.button("📥 Archiver l'analyse", use_container_width=True):
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                st.session_state.history.append({
                    "Fichier": rx_upload.name,
                    "Total_%": round(missing_pct, 2),
                    "Void_Max_%": round(max_void_pct, 3),
                    "Confiance_%": round(mean_conf, 1),
                    "img_bytes": img_jpg.tobytes(),
                    "Heure": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.toast("Archivé")

        with c_img:
            st.image(overlay, caption="Cyan = Plus gros void interne détecté", use_container_width=True)

# --- RAPPORT ET GALERIE ---
if st.session_state.history:
    st.divider()
    st.subheader("📊 Rapport Consolidé")
    
    df_full = pd.DataFrame(st.session_state.history)
    df_csv = df_full.drop(columns=['img_bytes'])
    
    # Tableau Stylisé
    styled_df = df_csv.style.apply(highlight_extremes, subset=['Total_%'], axis=0)
    st.dataframe(styled_df, use_container_width=True)

    # --- TÉLÉCHARGEMENT ZIP ---
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rapport_session.csv", df_csv.to_csv(index=False))
        for i, item in enumerate(st.session_state.history):
            z.writestr(f"images/{i+1}_{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("🎁 Télécharger ZIP (Images + CSV)", buf.getvalue(), "session_export.zip", "application/zip", use_container_width=True)

    # --- GALERIE DE VIGNETTES ---
    st.write("### 🖼️ Galerie")
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 6]:
            # Petit bouton pour zoomer
            if st.button(f"🔎 Zoom {idx+1}", key=f"btn_{idx}"):
                st.session_state.selected_img = item['img_bytes']
            st.image(item['img_bytes'], caption=f"{item['Total_%']}%")

    # --- AFFICHAGE DU ZOOM ---
    if st.session_state.selected_img:
        st.divider()
        st.subheader("🧐 Vue Détaillée")
        st.image(st.session_state.selected_img, use_container_width=True)
        if st.button("❌ Fermer la vue détaillée"):
            st.session_state.selected_img = None
            st.rerun()
