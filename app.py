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

st.set_page_config(page_title="RX Expert - Analyse Unitaire Pro", layout="wide")

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

# BOUTON DE VIDAGE (Soft Reset pour garder l'interface)
if st.sidebar.button("🗑️ Vider l'historique session", use_container_width=True):
    st.session_state.history = []
    st.session_state.selected_img = None
    st.sidebar.success("Historique vidé !")

st.sidebar.divider()

# 1. RÉGLAGES IA & IMAGE
contrast_val = st.sidebar.slider("Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.5)
model_file = st.sidebar.file_uploader("Charger modèle (.joblib)", type=["joblib"])

if model_file:
    @st.cache_resource
    def load_my_model(file):
        return joblib.load(file)
    
    clf = load_my_model(model_file)
    st.sidebar.success("IA Opérationnelle")

    st.header("🔍 Analyse de Soudure (Unitaire)")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("2. Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- ALIGNEMENT MANUEL ---
        st.sidebar.subheader("🕹️ Alignement manuel")
        tx = st.sidebar.number_input("Translation X", value=0, step=1)
        ty = st.sidebar.number_input("Translation Y", value=0, step=1)
        rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0, 0.5)
        scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

        # Chargement Radio
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # --- TRAITEMENT DU MASQUE (CORRECTION DIMENSIONS) ---
        mask_bytes = mask_upload.getvalue()
        insp_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Extraction sur taille originale du masque
        b_raw, g_raw, r_raw = cv2.split(insp_raw)
        m_green_orig = (g_raw > 100).astype(np.uint8)
        # Détection noir : b,g,r bas ET dans la zone verte
        m_black_orig = ((b_raw < 50) & (g_raw < 50) & (r_raw < 50) & (m_green_orig > 0)).astype(np.uint8)
        
        # Redimensionnement à la taille RX
        m_green_res = cv2.resize(m_green_orig, (W, H), interpolation=cv2.INTER_NEAREST)
        m_black_res = cv2.resize(m_black_orig, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Application de l'alignement
        M = engine.compose_similarity(scale, rot, float(tx), float(ty), W/2, H/2)
        z_adj = cv2.warpAffine((m_green_res & ~m_black_res), M, (W, H), flags=cv2.INTER_NEAREST)
        env_adj = cv2.warpAffine(m_green_res, M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(m_black_res, M, (W, H), flags=cv2.INTER_NEAREST)

        # --- ANALYSE IA ---
        with st.spinner("Analyse IA et calcul de confiance..."):
            features = engine.compute_features(img_gray)
            n_f = features.shape[-1]
            probs = clf.predict_proba(features.reshape(-1, n_f))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)
            conf_map = np.max(probs, axis=1).reshape(H, W)
            
            # Confiance moyenne uniquement sur la zone utile
            mean_conf = np.mean(conf_map[z_adj > 0]) * 100 if np.sum(z_adj > 0) > 0 else 0

        # --- CALCULS VOIDS ---
        valid_solder = (pred_map == 1) & (z_adj > 0)
        valid_voids = (pred_map == 0) & (z_adj > 0)
        area_total_px = np.sum(z_adj > 0)
        missing_pct = (1.0 - (np.sum(valid_solder) / area_total_px)) * 100.0 if area_total_px > 0 else 0

        # Top 5 Voids (Filtrage bulles internes)
        v_mask_u8 = (valid_voids.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(v_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        internals = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 3.0: continue
            c_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_m, [c], -1, 255, -1)
            
            # Exclusion si contient un trou noir (via) ou touche le bord vert
            if not np.any((c_m > 0) & (hol_adj > 0)):
                b_m = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(b_m, [c], -1, 255, 1)
                if not np.any((cv2.dilate(b_m, np.ones((3,3))) > 0) & (env_adj == 0)):
                    internals.append({'area': area, 'poly': c})
        
        top_5 = sorted(internals, key=lambda x: x['area'], reverse=True)[:5]

        # --- OVERLAY VISUEL ---
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[valid_solder] = [255, 255, 0] # Jaune (Soudure OK)
        overlay[valid_voids] = [255, 0, 0]    # Rouge (Vides)
        for v in top_5:
            cv2.drawContours(overlay, [v['poly']], -1, [0, 255, 255], 2) # Cyan (Bulles critiques)

        # --- AFFICHAGE RÉSULTATS ---
        st.divider()
        c_res, c_img = st.columns([1, 2])
        with c_res:
            st.metric("Manque Total", f"{missing_pct:.2f} %")
            st.metric("Confiance IA", f"{mean_conf:.1f} %")
            
            st.write("📏 **Top 5 Bulles Internes**")
            v_stats = {}
            for i in range(5):
                val = (top_5[i]['area'] / area_total_px * 100) if (area_total_px > 0 and i < len(top_5)) else 0.0
                st.caption(f"Bulle {i+1} : {val:.3f} %")
                v_stats[f"V{i+1}_%"] = round(val, 3)
            
            if st.button("📥 Archiver l'analyse", use_container_width=True):
                # OPTIMISATION RAM : Compression JPEG
                _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                entry = {
                    "Fichier": rx_upload.name,
                    "Total_%": round(missing_pct, 2),
                    "Confiance_%": round(mean_conf, 1),
                    "img_bytes": img_jpg.tobytes(),
                    "Heure": datetime.datetime.now().strftime("%H:%M:%S")
                }
                entry.update(v_stats)
                st.session_state.history.append(entry)
                st.toast("Analyse ajoutée au rapport")

        with c_img:
            st.image(overlay, caption="Jaune: OK | Rouge: Manque | Cyan: Bulles Critiques", use_container_width=True)

# --- SECTION RAPPORT & HISTORIQUE ---
if st.session_state.history:
    st.divider()
    st.subheader("📊 Rapport Consolidé")
    
    df_full = pd.DataFrame(st.session_state.history)
    df_csv = df_full.drop(columns=['img_bytes'])
    
    # Application du style coloré (Max=Rouge, Min=Bleu)
    styled_df = df_csv.style.apply(highlight_extremes, subset=['Total_%'], axis=0)
    st.dataframe(styled_df, use_container_width=True)

    # Export ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rapport_session.csv", df_csv.to_csv(index=False))
        for i, item in enumerate(st.session_state.history):
            z.writestr(f"images/{i+1}_{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("🎁 Télécharger ZIP (Images + CSV)", buf.getvalue(), "export_expert_rx.zip", "application/zip")

    # Galerie des miniatures
    st.write("### 🖼️ Galerie")
    cols = st.columns(6)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 6]:
            if st.button(f"🔎 {idx+1}", key=f"btn_{idx}"):
                st.session_state.selected_img = item['img_bytes']
            st.image(item['img_bytes'])

    if st.session_state.selected_img:
        st.image(st.session_state.selected_img, use_container_width=True)
        if st.button("Fermer la vue détaillée"): st.session_state.selected_img = None
