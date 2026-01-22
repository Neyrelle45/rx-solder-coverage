import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Final", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

# --- STYLE DU TABLEAU : ROUGE (MAX) ET BLEU (MIN) ---
def apply_table_style(df):
    """Met en rouge le pire cas et en bleu le meilleur cas (colonne Total_%)."""
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_min(subset=['Total_%'], color='#cce5ff', axis=0)

st.title("📦 Analyse de Série - Rapport Final")

# --- SIDEBAR & INPUTS ---
st.sidebar.title("⚙️ Configuration")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)
st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider les résultats"):
    st.session_state.batch_history = []
    st.rerun()

col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

st.divider()
trigger = st.button("🚀 Lancer l'analyse batch", use_container_width=True, type="primary")

if trigger and model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    st.session_state.batch_history = [] 
    
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    m_green = (g_s > 100).astype(np.uint8)
    m_black = ((r_s < 100) & (g_s < 100) & (b_s < 100) & (m_green > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_ref = np.sum(z_utile)

        # IA
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # Manques (Rouge)
        mask_red_raw = (raw_pred == 0) & z_utile
        cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_ref > 0 and (cv2.contourArea(c) / area_ref * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # --- LOGIQUE FILTRAGE VOID (NOUVELLE) ---
        v_max_area, v_max_poly = 0, None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for rc in red_cnts:
            # Création d'une bordure de vérification pour ce contour précis
            temp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(temp_mask, [rc], -1, 255, 1) # On dessine juste le contour (périmètre)
            
            # Vérification : le contour touche-t-il un via (hol_adj) ou l'extérieur (env_adj == 0) ?
            touches_via = np.any((temp_mask > 0) & (hol_adj > 0))
            touches_bord = np.any((temp_mask > 0) & (env_adj == 0))
            
            if not touches_via and not touches_bord:
                area = cv2.contourArea(rc)
                if area > v_max_area:
                    v_max_area, v_max_poly = area, rc

        # --- RENDU VISUEL ---
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay = res_rgb.copy()
        overlay[env_adj > 0] = [255, 255, 0]
        overlay[mask_red] = [255, 0, 0]
        res_rgb = np.where(env_adj[:,:,None] > 0, overlay, res_rgb)
        
        # Perçage des vias (Image RX brute)
        rx_original_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb[hol_adj > 0] = rx_original_rgb[hol_adj > 0]

        # Tracer du Void Max (Cyan)
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # Archivage
        total_val = round((np.sum(mask_red) / area_ref * 100), 2) if area_ref > 0 else 0
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": total_val,
            "Void_Max_%": round((v_max_area / area_ref * 100), 3) if area_ref > 0 else 0,
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.subheader("📊 Résultats Batch")
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    # Export ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport.csv", df.drop(columns=['img_bytes']).to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    st.download_button("📥 Télécharger ZIP Complet", zip_buf.getvalue(), "batch_results.zip", use_container_width=True)

    with st.expander("👁️ Galerie des analyses", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
