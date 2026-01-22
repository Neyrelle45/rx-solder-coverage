import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Stable", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

# --- STYLE DU TABLEAU ---
def apply_table_style(df):
    """Met en rouge la valeur MAX du manque total uniquement."""
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0)

st.title("📦 Analyse de Série")

# --- PARAMÈTRES (SIDEBAR) ---
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

# --- CHARGEMENT FICHIERS ---
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
    
    # 1. Préparation du Masque Maître
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    m_green = (g_s > 100).astype(np.uint8)
    # Détection des vias dans le masque (zones sombres dans le vert)
    m_black = ((r_s < 100) & (g_s < 100) & (b_s < 100) & (m_green > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 2. Image RX et Alignement
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        # SURFACE UTILE : La zone verte moins les vias (dénominateur de référence)
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_ref = np.sum(z_utile)

        # 3. Prédiction IA
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # 4. Détection des Manques (Rouge)
        # On ne garde la prédiction 'vide' que si elle est dans la zone de cuivre utile
        mask_red_raw = (raw_pred == 0) & z_utile
        
        # Filtrage micro-bulles (Seuil 0.1%)
        cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_ref > 0 and (cv2.contourArea(c) / area_ref * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # CALCUL DU TAUX (Garanti <= 100% car numérateur est un sous-ensemble du dénominateur)
        total_pixels_rouge = np.sum(mask_red)
        final_percentage = (total_pixels_rouge / area_ref * 100) if area_ref > 0 else 0

        # 5. Rendu Visuel (Méthode Transparence Vias)
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Overlay de couleur
        overlay = res_rgb.copy()
        overlay[env_adj > 0] = [255, 255, 0] # Fond jaune sur tout le cuivre (vias inclus au départ)
        overlay[mask_red] = [255, 0, 0]      # Manques en rouge
        
        # Fusion : On n'applique l'overlay QUE sur le cuivre, pas sur le fond
        res_rgb = np.where(env_adj[:,:,None] > 0, overlay, res_rgb)
        
        # PERÇAGE FINAL : On remet l'image RX brute exactement là où sont les vias
        rx_original_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb[hol_adj > 0] = rx_original_rgb[hol_adj > 0]

        # Macro-void (Contour Cyan)
        v_max_area, v_max_poly = 0, None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for rc in red_cnts:
            area = cv2.contourArea(rc)
            if area > v_max_area:
                v_max_area, v_max_poly = area, rc
        
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 6. Archivage
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round(final_percentage, 2),
            "Void_Max_%": round((v_max_area / area_ref * 100), 3) if area_ref > 0 else 0,
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE FINAL ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.subheader("📊 Tableau des Résultats")
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    with st.expander("👁️ Galerie des analyses", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
