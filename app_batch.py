import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Version Cumulée", layout="wide")

# Initialisation de l'historique
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def apply_table_style(df):
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_min(subset=['Total_%'], color='#cce5ff', axis=0)

# --- INTERFACE ---
st.title("📦 Analyse de Série (Optimisée)")

st.sidebar.title("⚙️ Contrôles")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider tout l'historique", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

st.divider()
trigger = st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary")

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
        
        # Zone utile (Soudure théorique)
        z_inspectee = ((env_adj > 0) & (hol_adj == 0)).astype(np.uint8)
        area_ref = np.sum(z_inspectee > 0)

        # 3. IA & NETTOYAGE DU BRUIT
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # On ne garde que la classe 0 (void) dans la zone utile
        void_raw = ((raw_pred == 0) & (z_inspectee > 0)).astype(np.uint8)
        
        # Morphologie : Ouverture pour supprimer les petits points isolés
        kernel_clean = np.ones((3,3), np.uint8)
        mask_red = cv2.morphologyEx(void_raw, cv2.MORPH_OPEN, kernel_clean)
        
        # LOGIQUE BINAIRE : La soudure (jaune) est tout ce qui n'est pas rouge dans z_inspectee
        mask_yellow = (z_inspectee > 0) & (mask_red == 0)

C'est un point critique : on doit distinguer le manque de soudure structurel (qui peut être en bordure de pad) du void sphérique/bulle (qui est piégé à l'intérieur).

Le problème est que si un "manque" touche le bord du masque, il n'est plus "enclavé". Dans ta définition métier, un Void Majeur doit être une bulle interne, tandis que les bords grignotés sont des Manques.

Voici comment verrouiller la logique dans app_batch.py pour que le Void Max (Cyan) ne soit calculé que sur des formes "fermées", tout en comptant tout le rouge dans le Manque Total.

Correction de la Logique de Distinction
On va utiliser une érosion du masque d'inspection pour créer une "zone de sécurité". Si un défaut rouge touche cette zone de sécurité (la bordure), il est déclassé du titre de "Void Max".

Python
        # --- 4. LOGIQUE VOID MAX (STRICTEMENT ENCLAVÉ) ---
        v_max_area, v_max_poly = 0, None
        
        # On crée une bordure interne de sécurité (3 pixels)
        # Tout ce qui touche cette bordure est considéré comme "Manque de bord" et non "Void"
        kernel_border = np.ones((3,3), np.uint8)
        z_interne_stricte = cv2.erode(z_inspectee, kernel_border, iterations=1)
        
        red_u8 = mask_red.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(red_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            c_area = cv2.contourArea(c)
            if c_area < 10: continue 
            
            # Masque du défaut actuel
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            
            # --- TESTS DE VALIDITÉ POUR "VOID MAJEUR" ---
            
            # 1. Exclusion des Vias (Noir)
            touches_via = np.any((c_mask > 0) & (hol_adj > 0))
            
            # 2. Exclusion des Bords (Si le défaut touche la zone hors 'interne_stricte')
            # Si un pixel du défaut est dans z_inspectee mais PAS dans z_interne_stricte, 
            # c'est qu'il touche le bord du masque vert.
            touches_bord = np.any((c_mask > 0) & (z_interne_stricte == 0))
            
            # 3. Validation
            if not touches_via and not touches_bord:
                if c_area > v_max_area:
                    v_max_area = c_area
                    v_max_poly = c

        # 5. RENDU VISUEL (Propre)
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb[mask_yellow] = [255, 255, 0] # Jaune plein
        res_rgb[mask_red > 0] = [255, 0, 0]   # Rouge
        
        # Overlay des VIAS par-dessus pour la clarté
        res_rgb[hol_adj > 0] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)[hol_adj > 0]
        
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 6. CALCULS & HISTORIQUE
# Le "Total" comptabilise TOUT le rouge (bords + vias inclus ou non)
        total_pct = (np.sum(mask_red) / area_ref * 100) if area_ref > 0 else 0
        
        # Le "Void Max" est le plus gros des défauts "propres" (ni bord, ni via)
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
    st.subheader(f"📊 Résultats (Total : {len(df)} images)")
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport_complet.csv", df.drop(columns=['img_bytes']).to_csv(index=False))
        for i, item in enumerate(st.session_state.batch_history):
            z.writestr(f"images/{i}_{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("📥 Télécharger TOUTE l'archive ZIP", zip_buf.getvalue(), "analyse_globale.zip", use_container_width=True)

    st.subheader("👁️ Galerie cumulée")
    grid = st.columns(4)
    for i, item in enumerate(st.session_state.batch_history):
        grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
