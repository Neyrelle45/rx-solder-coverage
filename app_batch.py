import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Precision", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

st.title("📦 Analyse de Série (Batch - Stabilité Haute)")

# --- CONFIGURATION ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement Masque")
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
trigger = st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary")

if trigger and model_file and uploaded_rx and mask_file:
    clf = joblib.load(model_file)
    st.session_state.batch_history = [] 
    
    # CHARGEMENT DU MASQUE (Source de vérité)
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    
    # Vias = Noir (Seuil plus large pour capturer tous les cercles d'exclusion)
    base_green = (g_s > 100).astype(np.uint8)
    base_black = ((r_s < 80) & (g_s < 80) & (b_s < 80) & (base_green > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 1. Image Grise
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape

        # 2. Synchronisation STRICTE (Chaque image est traitée isolément)
        m_g_local = cv2.resize(base_green, (W, H), interpolation=cv2.INTER_NEAREST)
        m_b_local = cv2.resize(base_black, (W, H), interpolation=cv2.INTER_NEAREST)
        
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        env_adj = cv2.warpAffine(m_g_local, M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(m_b_local, M, (W, H), flags=cv2.INTER_NEAREST)
        
        # Zone utile = Dans le masque vert MAIS pas dans les vias noirs
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_px = np.sum(z_utile)

        # 3. Prédiction IA
        feats = engine.compute_features(img_gray)
        probs = clf.predict_proba(feats.reshape(-1, feats.shape[-1]))
        pred = np.argmax(probs, axis=1).reshape(H, W)

        # 4. Identification des Manques (Rouge)
        # On ne garde que les manques qui sont dans la zone utile (exclut de fait les vias)
        raw_voids = (pred == 0) & (env_adj > 0) # On regarde tout ce qui est dans le carré vert
        
        # Filtrage : On retire ce qui touche ou est dans un via (hol_adj)
        final_voids_mask = (raw_voids > 0) & (hol_adj == 0)
        
        # Filtrage Micro-bulles (Seuil 0.1%)
        cnts, _ = cv2.findContours(final_voids_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_px > 0 and (cv2.contourArea(c) / area_px * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # 5. TEST DE COLLISION MACRO-VOID (Cyan)
        # Une bulle n'est "Macro-void" que si elle est TOTALEMENT entourée de jaune (soudure)
        v_max_area = 0
        v_max_poly = None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for rc in red_cnts:
            area = cv2.contourArea(rc)
            test_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(test_m, [rc], -1, 255, 2) # On teste une bordure un peu épaisse
            
            # CRITÈRES D'EXCLUSION :
            # - Ne doit pas toucher un via (hol_adj)
            # - Ne doit pas toucher le bord du masque vert (env_adj)
            touches_via = np.any((test_m > 0) & (hol_adj > 0))
            touches_bord = np.any((test_m > 0) & (env_adj == 0))
            
            if not touches_via and not touches_bord:
                if area > v_max_area:
                    v_max_area = area
                    v_max_poly = rc

        # 6. Rendu Couleur
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # Fond soudure IA = Jaune
        res_rgb[z_utile & (~mask_red)] = [255, 255, 0]
        # Manques validés = Rouge
        res_rgb[mask_red] = [255, 0, 0]
        # Macro-void validé = Cercle Cyan
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 7. Archivage
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round((np.sum(mask_red)/area_px*100), 2) if area_px > 0 else 0,
            "Void_Max_%": round((v_max_area/area_px*100), 3) if area_px > 0 else 0,
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.dataframe(df.drop(columns=['img_bytes']), use_container_width=True)
    
    with st.expander("👁️ Galerie des résultats", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=item['Fichier'])
