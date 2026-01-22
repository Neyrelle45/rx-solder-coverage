import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Precision Final", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def apply_table_style(df):
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_max(subset=['Void_Max_%'], color='#cce5ff', axis=0)

st.title("📦 Analyse de Série - Calculs Certifiés")

# --- SIDEBAR & INPUTS ---
# (Gardez votre configuration sidebar et uploads identique)
# ... [Code Sidebar et Uploads] ...

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
        
        # --- LOGIQUE DE SURFACE (CRITIQUE) ---
        # La zone utile est le cuivre théorique (vert) MOINS les vias (noir)
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_ref = np.sum(z_utile) # C'est notre 100%

        # IA
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # --- CALCUL DU MANQUE ---
        # On ne garde la prédiction "manque" (0) QUE si elle est dans z_utile
        mask_red_raw = (raw_pred == 0) & z_utile
        
        # Filtrage micro-bulles (Seuil 0.1% de area_ref)
        cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_ref > 0 and (cv2.contourArea(c) / area_ref * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # Calcul final du % (Maintenant garanti <= 100%)
        total_pixels_rouge = np.sum(mask_red)
        final_percentage = (total_pixels_rouge / area_ref * 100) if area_ref > 0 else 0

        # --- MACRO-VOID (CYAN) ---
        v_max_area, v_max_poly = 0, None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for rc in red_cnts:
            area = cv2.contourArea(rc)
            t_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(t_m, [rc], -1, 255, 2)
            if not np.any((t_m > 0) & (hol_adj > 0)) and not np.any((t_m > 0) & (env_adj == 0)):
                if area > v_max_area:
                    v_max_area, v_max_poly = area, rc

        # --- RENDU VISUEL ---
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # 1. On peint la soudure (jaune) sur toute la zone utile
        res_rgb[z_utile] = [255, 255, 0]
        # 2. On peint le manque (rouge) par dessus
        res_rgb[mask_red] = [255, 0, 0]
        # 3. On "troue" le tout avec l'image originale pour les vias
        res_rgb[hol_adj > 0] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)[hol_adj > 0]
        
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round(min(final_percentage, 100.0), 2),
            "Void_Max_%": round((v_max_area / area_ref * 100), 3) if area_ref > 0 else 0,
            "img_bytes": cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    # ... [Galerie et Export ZIP] ...
