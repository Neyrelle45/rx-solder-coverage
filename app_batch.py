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

st.title("📦 Analyse de Série (Stabilité Renforcée)")

# --- PARAMÈTRES ---
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

# --- CHARGEMENT ---
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

st.divider()
trigger = st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary")

if trigger and model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    st.session_state.batch_history = [] 
    
    # CHARGEMENT DU MASQUE SOURCE
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_src, g_src, b_src = cv2.split(m_rgb)
    
    # Extraction des formes de base une seule fois
    base_green_mask = (g_src > 100).astype(np.uint8)
    base_black_mask = ((r_src < 60) & (g_src < 60) & (b_src < 60) & (base_green_mask > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # --- ISOLATION ET NETTOYAGE ---
        # On force la lecture de l'image RX sans aucun cache
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # RE-GÉNÉRATION LOCALE DU MASQUE (Pour éviter les décalages aléatoires)
        # On redimensionne le masque source aux dimensions EXACTES de l'image actuelle
        m_g_local = cv2.resize(base_green_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        m_b_local = cv2.resize(base_black_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Application de la transformation
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(m_g_local, M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(m_b_local, M, (W, H), flags=cv2.INTER_NEAREST)
        
        # Zone utile (Soudure possible)
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_px = np.sum(z_utile)

        # --- IA ---
        feats = engine.compute_features(img_gray)
        probs = clf.predict_proba(feats.reshape(-1, feats.shape[-1]))
        pred = np.argmax(probs, axis=1).reshape(H, W)

        # --- FILTRAGE ET COULEURS ---
        # Uniquement ce qui est dans z_utile
        void_map = (pred == 0) & z_utile
        cnts, _ = cv2.findContours(void_map.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            # Seuil de 0.1% pour être considéré comme un manque (Rouge)
            if area_px > 0 and (cv2.contourArea(c) / area_px * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        disp_solder = z_utile & (~mask_red)
        
        # --- IDENTIFICATION DU MACRO-VOID (CYAN) ---
        # Correction : Un macro-void est une bulle rouge QUI NE TOUCHE PAS UN VIA
        v_max_area = 0
        v_max_poly = None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for rc in red_cnts:
            area = cv2.contourArea(rc)
            # Test d'enclavement : on vérifie si le contour rouge touche la zone noire (via)
            test_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(test_mask, [rc], -1, 255, 1) # On teste juste le bord
            # Si le bord du vide rouge touche un via noir, ce n'est pas un macro-void à encercler
            if not np.any((test_mask > 0) & (hol_adj > 0)):
                if area > v_max_area:
                    v_max_area = area
                    v_max_poly = rc

        # --- SORTIE VISUELLE ---
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb[disp_solder] = [255, 255, 0] # Jaune
        res_rgb[mask_red] = [255, 0, 0]      # Rouge
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2) # Cyan

        # Stockage
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round((np.sum(mask_red)/area_px*100), 2) if area_px > 0 else 0,
            "Void_Max_%": round((v_max_area/area_px*100), 3) if area_px > 0 else 0,
            "img": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE FINAL ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history).drop(columns=['img'])
    st.dataframe(df, use_container_width=True)
    
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        for item in st.session_state.batch_history:
            z.writestr(f"{item['Fichier']}.jpg", item['img'])
    st.download_button("📥 ZIP Résultats", zip_buf.getvalue(), "batch_export.zip")

    with st.expander("👁️ Galerie", expanded=True):
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            cols[i % 4].image(item['img'], caption=item['Fichier'])
