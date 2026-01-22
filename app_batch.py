import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Final Fix", layout="wide")

if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

# --- FONCTIONS DE STYLE DU TABLEAU ---
def style_stats(df):
    """Applique les couleurs rouge (max total) et bleu (max void) au tableau"""
    styler = df.style
    if not df.empty:
        # Rouge pour le maximum de manque total
        styler = styler.highlight_max(subset=['Total_%'], color='#ffcccc')
        # Bleu pour le plus gros macro-void (bulle)
        styler = styler.highlight_max(subset=['Void_Max_%'], color='#cce5ff')
    return styler

st.title("📦 Analyse de Série (Priorité Masque & Vias)")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Paramètres")
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

# --- INPUTS ---
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
    
    # 1. Préparation du Masque Source
    m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
    r_s, g_s, b_s = cv2.split(m_rgb)
    base_green = (g_s > 100).astype(np.uint8)
    base_black = ((r_s < 80) & (g_s < 80) & (b_s < 80) & (base_green > 0)).astype(np.uint8)

    progress = st.progress(0)
    
    for idx, rx_f in enumerate(uploaded_rx):
        # 2. Image et Alignement
        img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
        H, W = img_gray.shape
        M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
        
        env_adj = cv2.warpAffine(cv2.resize(base_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        hol_adj = cv2.warpAffine(cv2.resize(base_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
        
        # 3. Prédiction IA
        feats = engine.compute_features(img_gray)
        pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)

        # 4. LOGIQUE DE CALCUL (SOUSTRACTION DES VIAS)
        # On définit le manque UNIQUEMENT là où l'IA dit "vide" ET où le masque dit "zone utile"
        mask_red_raw = (pred == 0) & (env_adj > 0) & (hol_adj == 0)
        
        # Filtrage micro-bulles (0.1%)
        area_utile = np.sum((env_adj > 0) & (hol_adj == 0))
        cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_red = np.zeros((H, W), dtype=bool)
        for c in cnts:
            if area_utile > 0 and (cv2.contourArea(c) / area_utile * 100) >= 0.1:
                cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

        # 5. MACRO-VOID (CYAN)
        v_max_area, v_max_poly = 0, None
        red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for rc in red_cnts:
            area = cv2.contourArea(rc)
            # Collision : ne doit pas toucher le bord du composant ni un via
            t_m = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(t_m, [rc], -1, 255, 2)
            if not np.any((t_m > 0) & (hol_adj > 0)) and not np.any((t_m > 0) & (env_adj == 0)):
                if area > v_max_area:
                    v_max_area = area
                    v_max_poly = rc

        # 6. RENDU COULEUR (L'ordre est crucial pour les vias)
        res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # a. On met tout le composant en JAUNE (Soudure par défaut)
        res_rgb[env_adj > 0] = [255, 255, 0]
        # b. On applique le ROUGE (Manques IA validés)
        res_rgb[mask_red] = [255, 0, 0]
        # c. On "NETTOIE" les VIAS (Retour à l'image originale ou blanc)
        # Ici on remet l'image RX originale dans les trous pour qu'ils soient visibles
        res_rgb[hol_adj > 0] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)[hol_adj > 0]
        
        if v_max_poly is not None:
            cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

        # 7. Sauvegarde
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        st.session_state.batch_history.append({
            "Fichier": rx_f.name,
            "Total_%": round((np.sum(mask_red)/area_utile*100), 2) if area_utile > 0 else 0,
            "Void_Max_%": round((v_max_area/area_utile*100), 3) if area_utile > 0 else 0,
            "img_bytes": enc.tobytes()
        })
        progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE FINAL ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    df_display = df.drop(columns=['img_bytes'])
    
    st.subheader("📊 Tableau Récapitulatif")
    # Retour du code couleur : Rouge = Max Manque / Bleu = Max Bulle
    st.dataframe(style_stats(df_display), use_container_width=True)
    
    # Export ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rapport.csv", df_display.to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    st.download_button("📥 Télécharger ZIP Complet", buf.getvalue(), "resultats_batch.zip", use_container_width=True)

    with st.expander("👁️ Galerie des pièces", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
