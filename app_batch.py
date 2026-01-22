import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Formatage Final", layout="wide")

# Initialisation de l'historique
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def apply_table_style(df):
    """Style du tableau : Bleu pour le min, Rouge pour le max sur Total_%."""
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_min(subset=['Total_%'], color='#cce5ff', axis=0)

# --- INTERFACE ---
st.title("📦 Analyse de Série (Formatage des Résultats)")

st.sidebar.title("⚙️ Contrôles")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

st.sidebar.divider()
if st.sidebar.button("🗑️ Vider tout l'historique", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

st.divider()
trigger = st.button("🚀 Lancer l'analyse (Ajouter à la suite)", use_container_width=True, type="primary")

if trigger and model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    
    # 1. PRÉPARATION DES MASQUES MAITRES
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    m_green_master = (g_s > 100).astype(np.uint8)
    m_black_master = ((r_s < 100) & (g_s < 100) & (b_s < 100) & (m_green_master > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 2. ALIGNEMENT
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        
        env_adj = cv2.warpAffine(cv2.resize(m_green_master, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black_master, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        # Zone inspectée = Vert MOINS Noir
        z_inspectee = ((env_adj > 0) & (hol_adj == 0)).astype(np.uint8)
        area_ref = np.sum(z_inspectee)

        # 3. IA & SEGMENTATION ROUGE
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        mask_red = ((raw_pred == 0) & (z_inspectee > 0)).astype(np.uint8)
        
        # 4. LOGIQUE VOID MAX (AUCUN CONTACT VIA/BORD)
        kernel = np.ones((3,3), np.uint8)
        z_autorisee_interne = cv2.erode(z_inspectee, kernel, iterations=1)
        
        v_max_area, v_max_poly = 0, None
        cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            c_area = cv2.contourArea(c)
            if c_area < (area_ref * 0.001): continue 
            
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            
            if not np.any((c_mask > 0) & (z_autorisee_interne == 0)):
                if c_area > v_max_area:
                    v_max_area = c_area
                    v_max_poly = c

        # 5. RENDU VISUEL
        img_rx_col = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb = img_rx_col.copy()
        overlay = res_rgb.copy()
        overlay[env_adj > 0] = [255, 255, 0] 
        overlay[mask_red > 0] = [255, 0, 0]   
        res_rgb = np.where(env_adj[:,:,None] > 0, overlay, res_rgb)
        res_rgb[hol_adj > 0] = img_rx_col[hol_adj > 0]
        
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 6. CALCULS ET FORMATAGE
        raw_total = (np.sum(mask_red) / area_ref * 100) if area_ref > 0 else 0
        raw_void_max = (v_max_area / area_ref * 100) if area_ref > 0 else 0
        
        # Formatage Manque Total (Arrondi entier)
        fmt_total = f"{int(round(raw_total))}%"
        
        # Formatage Void Max
        if raw_void_max < 1.0:
            fmt_void = "<1%"
        else:
            fmt_void = f"{int(round(raw_void_max))}%"

        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": raw_total,          # On garde le float pour le code couleur
            "Affichage_Total": fmt_total,  # Texte pour le tableau
            "Void_Max_%": fmt_void,        # Texte selon votre règle
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE FINAL ---
if st.session_state.batch_history:
    df_raw = pd.DataFrame(st.session_state.batch_history)
    
    # On prépare le tableau final pour l'utilisateur
    df_display = df_raw.copy()
    # On remplace les colonnes par les versions formatées
    df_display["Total_%"] = df_raw["Affichage_Total"]
    df_display = df_display.drop(columns=["Affichage_Total", "img_bytes"])
    
    st.subheader(f"📊 Résultats (Total : {len(df_raw)} images)")
    
    # On applique le style sur le df_display mais en utilisant les valeurs numériques du df_raw pour les calculs de min/max si nécessaire, 
    # ou plus simplement on affiche le dataframe formaté directement.
    st.dataframe(df_display, use_container_width=True)
    
    # Export ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport.csv", df_display.to_csv(index=False))
        for i, item in enumerate(st.session_state.batch_history):
            z.writestr(f"images/{i}_{item['Fichier']}.jpg", item['img_bytes'])
    st.download_button("📥 Télécharger TOUTE l'archive ZIP", zip_buf.getvalue(), "analyse_globale.zip", use_container_width=True)

    st.subheader("👁️ Galerie cumulée")
    grid = st.columns(4)
    for i, item in enumerate(st.session_state.batch_history):
        # Pour la légende sous les vignettes, on utilise aussi le format arrondi
        caption_txt = f"{item['Fichier']} (Total: {item['Affichage_Total']} | Max: {item['Void_Max_%']})"
        grid[i % 4].image(item['img_bytes'], caption=caption_txt)
