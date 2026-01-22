import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Certifié V4", layout="wide")

# Initialisation de l'historique
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def apply_table_style(df):
    """Style du tableau : Bleu pour le minimum, Rouge pour le maximum."""
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_min(subset=['Total_%'], color='#cce5ff', axis=0)

st.title("📦 Analyse de Série (Version Correction Calcul & Voids)")

# --- SIDEBAR : CONFIGURATION & BOUTONS ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Modèle IA", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

st.sidebar.divider()
if st.sidebar.button("🗑️ Vider les résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

# --- CHARGEMENT DES FICHIERS ---
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
    
    # 1. PRÉPARATION DU MASQUE DE RÉFÉRENCE (Strict)
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    
    # Vert = Surface totale / Noir = Vias
    m_green = (g_s > 100).astype(np.uint8)
    m_black = ((r_s < 100) & (g_s < 100) & (b_s < 100) & (m_green > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 2. CHARGEMENT & ALIGNEMENT
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        
        env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        # --- LOGIQUE DE SURFACE DE RÉFÉRENCE (DENOMINATEUR) ---
        # La zone inspectée est UNIQUEMENT là où il y a du vert ET PAS de noir (vias)
        z_utile = (env_adj > 0) & (hol_adj == 0)
        area_ref = np.sum(z_utile)

        # 3. IA & MASQUE ROUGE (NUMÉRATEUR)
        feats = engine.compute_features(img_gray)
        raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
        
        # Le manque (rouge) est strictement contraint à la zone utile
        mask_red_raw = (raw_pred == 0) & z_utile
        
        # Filtrage micro-bulles (Seuil 0.1% de la surface utile)
        cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_ref > 0 and (cv2.contourArea(c) / area_ref * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # 4. IDENTIFICATION DU VOID MAX (STRICTE)
        v_max_area, v_max_poly = 0, None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for rc in red_cnts:
            # Création d'un masque de vérification pour ce contour précis
            c_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(c_mask, [rc], -1, 255, -1)
            
            # CONDITION 1 : Le void ne doit contenir aucun via (noir)
            contient_via = np.any((c_mask > 0) & (hol_adj > 0))
            
            # CONDITION 2 : Le pourtour du void ne doit pas toucher le bord du masque utile
            edge_check = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(edge_check, [rc], -1, 255, 1)
            touche_bord = np.any((edge_check > 0) & (z_utile == 0))
            
            if not contient_via and not touche_bord:
                area = cv2.contourArea(rc)
                if area > v_max_area:
                    v_max_area, v_max_poly = area, rc

        # 5. RENDU VISUEL (SUPERPOSITION DES COUCHES)
        img_rx_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        res_rgb = img_rx_rgb.copy()
        
        # A. Fond Jaune (Soudure) et Rouge (Manques)
        overlay = res_rgb.copy()
        overlay[env_adj > 0] = [255, 255, 0] # Jaune sur toute l'emprise
        overlay[mask_red] = [255, 0, 0]      # Rouge par dessus
        res_rgb = np.where(env_adj[:,:,None] > 0, overlay, res_rgb)
        
        # B. Transparence des VIAS (Restauration RX d'origine)
        # On force les pixels des vias à redevenir les pixels de la radio brute
        res_rgb[hol_adj > 0] = img_rx_rgb[hol_adj > 0]
        
        # C. Dessin du Void Max interne (Cyan)
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 6. CALCULS FINAUX & ARCHIVAGE
        final_total = (np.sum(mask_red) / area_ref * 100) if area_ref > 0 else 0
        final_void_max = (v_max_area / area_ref * 100) if area_ref > 0 else 0

        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round(final_total, 2),
            "Void_Max_%": round(final_void_max, 3),
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.subheader("📊 Résultats Statistiques")
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    # Export ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport_batch.csv", df.drop(columns=['img_bytes']).to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("📥 Télécharger ZIP Complet", zip_buf.getvalue(), "analyse_batch.zip", use_container_width=True)

    # Vignettes
    with st.expander("👁️ Vignettes des analyses", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
