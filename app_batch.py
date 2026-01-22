import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Mirror", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

# --- STYLE DU TABLEAU ---
def apply_table_style(df):
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc') \
                   .highlight_max(subset=['Void_Max_%'], color='#cce5ff')

st.title("📦 Analyse de Série - Mode Synchronisé")

# --- PARAMÈTRES ---
st.sidebar.title("⚙️ Configuration")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement Fixe")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider l'historique"):
    st.session_state.batch_history = []
    st.rerun()

# --- CHARGEMENT ---
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
    
    # 1. Pré-chargement du masque maître
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    # On définit les zones une fois pour toutes
    m_green = (g_s > 100).astype(np.uint8)
    m_black = ((r_s < 80) & (g_s < 80) & (b_s < 80) & (m_green > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 2. Chargement image RX
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        
        # 3. Synchronisation GÉOMÉTRIQUE (Identique à app.py)
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        # ZONE UTILE STRICTE
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_ref = np.sum(z_utile)

        # 4. PRÉDICTION IA FILTRÉE
        # On calcule les features sur toute l'image
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # ICI EST LA CLÉ : On force l'IA à être "Soudure" (1) partout où le masque l'exige,
        # sauf si elle est vraiment sûre qu'il y a un manque (0).
        # On ignore totalement les prédictions hors zone utile.
        final_pred = np.ones((H, W), dtype=np.uint8) # Par défaut : Soudure partout
        final_pred[z_utile] = raw_pred[z_utile] # On injecte l'IA uniquement dans le vert

        # 5. CALCUL DES MANQUES (ROUGE)
        mask_red_raw = (final_pred == 0) & z_utile
        
        # Filtrage micro-bulles
        cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_ref > 0 and (cv2.contourArea(c) / area_ref * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # 6. MACRO-VOID (CYAN)
        v_max_area, v_max_poly = 0, None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for rc in red_cnts:
            area = cv2.contourArea(rc)
            t_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(t_m, [rc], -1, 255, 2)
            # Ne doit toucher aucun via ni aucun bord du masque vert
            if not np.any((t_m > 0) & (hol_adj > 0)) and not np.any((t_m > 0) & (env_adj == 0)):
                if area > v_max_area:
                    v_max_area = area
                    v_max_poly = rc

        # 7. RENDU FINAL (Fusion d'images)
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Jaune pour la soudure prédite dans la zone utile
        res_rgb[z_utile & (~mask_red)] = [255, 255, 0]
        # Rouge pour les manques validés
        res_rgb[mask_red] = [255, 0, 0]
        # On remet l'image RX originale dans les vias (Transparence)
        res_rgb[hol_adj > 0] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)[hol_adj > 0]
        
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 8. Sauvegarde
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round((np.sum(mask_red)/area_ref*100), 2) if area_ref > 0 else 0,
            "Void_Max_%": round((v_max_area/area_ref*100), 3) if area_ref > 0 else 0,
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    with st.expander("👁️ Résultats Visuels", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=item['Fichier'])
