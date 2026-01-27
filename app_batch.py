import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Analyse de Série", layout="wide")

# Initialisation de l'historique
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

st.title("📦 Analyse de Série (Batch Mode)")
st.info("Traitez plusieurs images RX simultanément avec un alignement manuel sur masque maître.")

# --- SIDEBAR : CONFIGURATION ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Modèle IA (.joblib)", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("Alignement")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider l'historique", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

# --- CHARGEMENT DES FICHIERS ---
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence (Vert/Noir)", type=["png", "jpg"])

# --- TRAITEMENT ---
if st.button("🚀 Lancer l'analyse de série", type="primary", use_container_width=True):
    if not model_file or not uploaded_rx or not mask_file:
        st.error("Veuillez charger le modèle, les images et le masque.")
    else:
        # Chargement du modèle
        clf = joblib.load(model_file)
        
        # Préparation du Masque Maître (Lecture robuste)
        m_bytes = np.frombuffer(mask_file.read(), np.uint8)
        m_raw = cv2.imdecode(m_bytes, cv2.IMREAD_COLOR)
        m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
        r_m, g_m, b_m = cv2.split(m_rgb)
        
        # Extraction des zones Maîtres
        m_green_master = (g_m > 100).astype(np.uint8)
        m_black_master = ((r_m < 100) & (g_m < 100) & (b_m < 100) & (m_green_master > 0)).astype(np.uint8)

        progress_bar = st.progress(0)
        
        for idx, rx_f in enumerate(uploaded_rx):
            # 1. Lecture sécurisée de l'image RX
            rx_f.seek(0)
            file_bytes = np.frombuffer(rx_f.read(), np.uint8)
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            if img_raw is None: continue
            
            # Application du contraste (CLAHE)
            if contrast_val > 0:
                clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8))
                img_gray = clahe.apply(img_raw)
            else:
                img_gray = img_raw
            
            H, W = img_gray.shape

            # 2. Alignement du masque
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(cv2.resize(m_green_master, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(cv2.resize(m_black_master, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
            
            # Zone d'inspection = Vert SANS Noir
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_total_px = np.sum(z_utile)

            # 3. Prédiction IA et Nettoyage
            feats = engine.compute_features(img_gray)
            probs = clf.predict_proba(feats.reshape(-1, feats.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)
            
            # Suppression du bruit (Opening)
            kernel = np.ones((3,3), np.uint8)
            void_raw = ((pred_map == 0) & z_utile).astype(np.uint8)
            clean_voids = cv2.morphologyEx(void_raw, cv2.MORPH_OPEN, kernel)
            
            # Logique Binaire (Jaune = Reste du pad)
            clean_solder = (z_utile) & (clean_voids == 0)

            # 4. Calcul du Void Majeur (Strictement enclavé)
            v_max_area, v_max_poly = 0, None
            z_stricte = cv2.erode(z_utile.astype(np.uint8), kernel, iterations=1)
            
            red_u8 = (clean_voids * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(red_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 10: continue
                
                c_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                
                # Exclusion si touche un Via (Noir) ou la Bordure du pad
                touches_via = np.any((c_mask > 0) & (hol_adj > 0))
                touches_bord = np.any((c_mask > 0) & (z_stricte == 0))
                
                if not touches_via and not touches_bord:
                    if area > v_max_area:
                        v_max_area = area
                        v_max_poly = c

            # 5. Calculs des ratios
            missing_pct = (np.sum(clean_voids) / area_total_px * 100.0) if area_total_px > 0 else 0
            max_void_pct = (v_max_area / area_total_px * 100.0) if area_total_px > 0 else 0

            # 6. Rendu Visuel
            res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            res_rgb[clean_solder] = [255, 255, 0] # Jaune
            res_rgb[clean_voids > 0] = [255, 0, 0] # Rouge
            if v_max_poly is not None:
                cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2) # Cyan

            # Archivage
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
            st.session_state.batch_history.append({
                "Fichier": rx_f.name,
                "Total_Manque_%": round(missing_pct, 2),
                "Void_Max_Enclave_%": round(max_void_pct, 3),
                "img_bytes": enc.tobytes(),
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    
    st.divider()
    st.subheader(f"📊 Rapport d'analyse ({len(df)} images)")
    
    # Tableau récapitulatif
    st.dataframe(df.drop(columns=['img_bytes']), use_container_width=True)

    # Téléchargement ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport_complet.csv", df.drop(columns=['img_bytes']).to_csv(index=False))
        for i, item in enumerate(st.session_state.batch_history):
            z.writestr(f"images/{i}_{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("📥 Télécharger l'archive complète (CSV + Images)", 
                       zip_buf.getvalue(), "analyse_rx_batch.zip", 
                       use_container_width=True)

    # Galerie
    st.subheader("👁️ Galerie des résultats")
    grid = st.columns(4)
    for i, item in enumerate(st.session_state.batch_history):
        grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_Manque_%']}%)")
