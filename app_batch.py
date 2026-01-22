import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Correction Radicale", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def apply_table_style(df):
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_min(subset=['Total_%'], color='#cce5ff', axis=0)

# --- INTERFACE ---
st.title("📦 Analyse de Série (Correctif Géométrique Strict)")

st.sidebar.title("⚙️ Contrôles")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

st.sidebar.divider()
if st.sidebar.button("🗑️ Vider les résultats", use_container_width=True):
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
    
    # 1. PRÉPARATION DES MASQUES MAITRES
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    m_green_master = (g_s > 100).astype(np.uint8)
    # Les vias sont noirs (R,G,B < 100) ET à l'intérieur de la zone verte
    m_black_master = ((r_s < 100) & (g_s < 100) & (b_s < 100) & (m_green_master > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 2. ALIGNEMENT
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        
        env_adj = cv2.warpAffine(cv2.resize(m_green_master, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black_master, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        # --- VERROU GÉOMÉTRIQUE ---
        # Zone inspectée = Vert MOINS Noir
        z_inspectee = ((env_adj > 0) & (hol_adj == 0)).astype(np.uint8)
        area_ref = np.sum(z_inspectee)

        # 3. IA & SEGMENTATION ROUGE
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # On force : Rouge = IA dit Rouge ET on est dans z_inspectee
        mask_red = ((raw_pred == 0) & (z_inspectee > 0)).astype(np.uint8)
        
        # 4. LOGIQUE VOID MAX (AUCUN CONTACT VIA/BORD)
        # On réduit la zone autorisée d'un pixel pour garantir qu'on ne touche pas le bord
        kernel = np.ones((3,3), np.uint8)
        z_autorisee_interne = cv2.erode(z_inspectee, kernel, iterations=1)
        
        v_max_area, v_max_poly = 0, None
        cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            c_area = cv2.contourArea(c)
            if c_area < (area_ref * 0.001): continue # Filtre micro-bulles
            
            # Vérification : tous les points du contour doivent être dans z_autorisee_interne
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            
            # Si le void déborde sur un via ou un bord (pixel dans z_autorisee_interne == 0)
            if not np.any((c_mask > 0) & (z_autorisee_interne == 0)):
                if c_area > v_max_area:
                    v_max_area = c_area
                    v_max_poly = c

        # 5. RENDU VISUEL (SUPERPOSITION FINALE)
        img_rx_col = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb = img_rx_col.copy()
        
        # Calque Jaune/Rouge
        overlay = res_rgb.copy()
        overlay[env_adj > 0] = [255, 255, 0] # Jaune
        overlay[mask_red > 0] = [255, 0, 0]   # Rouge
        # Application du calque uniquement sur la zone de soudure
        res_rgb = np.where(env_adj[:,:,None] > 0, overlay, res_rgb)
        
        # RÉTABLISSEMENT DES VIAS (TRANSPARENCE)
        # On remet l'image RX brute là où il y a des trous noirs
        res_rgb[hol_adj > 0] = img_rx_col[hol_adj > 0]
        
        # CONTOUR CYAN (Void Max Interne)
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 6. CALCULS & HISTORISATION
        total_pct = (np.sum(mask_red) / area_ref * 100) if area_ref > 0 else 0
        void_max_pct = (v_max_area / area_ref * 100) if area_ref > 0 else 0

        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round(total_pct, 2),
            "Void_Max_%": round(void_max_pct, 3),
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE FINAL ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.subheader("📊 Résultats Statistiques")
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    # Bouton ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport.csv", df.drop(columns=['img_bytes']).to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    st.download_button("📥 Télécharger l'archive ZIP", zip_buf.getvalue(), "analyse_rx.zip", use_container_width=True)

    # Vignettes
    st.subheader("👁️ Galerie des résultats")
    grid = st.columns(4)
    for i, item in enumerate(st.session_state.batch_history):
        grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
