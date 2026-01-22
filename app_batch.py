import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Final", layout="wide")

# --- INITIALISATION ---
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def apply_table_style(df):
    """Applique les couleurs max indépendantes sur le tableau."""
    return df.style.highlight_max(subset=['Total_%'], color='#ffcccc', axis=0) \
                   .highlight_max(subset=['Void_Max_%'], color='#cce5ff', axis=0)

st.title("📦 Analyse de Série (Calculs Certifiés)")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Modèle IA (.joblib)", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (Réglage critique)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement Masque")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider les résultats"):
    st.session_state.batch_history = []
    st.rerun()

# --- INTERFACE PRINCIPALE ---
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence", type=["png", "jpg"])

st.divider()

# Définition du bouton AVANT le test de condition
trigger = st.button("🚀 Lancer l'analyse batch", use_container_width=True, type="primary")

# --- LOGIQUE D'ANALYSE ---
if trigger:
    if not model_file or not uploaded_rx or not mask_file:
        st.warning("⚠️ Veuillez charger le modèle, les images et le masque.")
    else:
        # Chargement du modèle
        clf = joblib.load(model_file)
        st.session_state.batch_history = [] 
        
        # Pré-traitement du masque maître
        m_raw = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
        m_rgb = cv2.cvtColor(m_raw, cv2.COLOR_BGR2RGB)
        r_s, g_s, b_s = cv2.split(m_rgb)
        
        # Identification des zones du masque
        m_green = (g_s > 100).astype(np.uint8)
        m_black = ((r_s < 100) & (g_s < 100) & (b_s < 100) & (m_green > 0)).astype(np.uint8)

        progress = st.progress(0)
        
        for idx, rx_f in enumerate(uploaded_rx):
            # 1. Chargement et Contraste
            img_gray = engine.load_gray(rx_f, contrast_limit=contrast_val)
            H, W = img_gray.shape
            
            # 2. Alignement Géométrique
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(cv2.resize(m_green, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(cv2.resize(m_black, (W, H), interpolation=cv2.INTER_NEAREST), M, (W, H), flags=cv2.INTER_NEAREST)
            
            # SURFACE DE RÉFÉRENCE (Dénominateur : Vert pur sans les trous noirs)
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_ref = np.sum(z_utile) 

            if area_ref == 0:
                st.error(f"Erreur d'alignement sur {rx_f.name} : Surface utile nulle.")
                continue

            # 3. Prédiction IA
            feats = engine.compute_features(img_gray)
            raw_pred = np.argmax(clf.predict_proba(feats.reshape(-1, feats.shape[-1])), axis=1).reshape(H, W)
            
            # 4. Calcul du Manque (ROUGE)
            # Intersection entre la prédiction "0" et la zone autorisée
            mask_red_raw = (raw_pred == 0) & z_utile
            
            # Filtrage micro-bulles (0.1% de la surface utile)
            cnts, _ = cv2.findContours(mask_red_raw.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_red = np.zeros((H, W), dtype=bool)
            for c in cnts:
                if (cv2.contourArea(c) / area_ref * 100) >= 0.1:
                    cv2.drawContours(mask_red.view(np.uint8), [c], -1, 1, -1)

            # CALCUL STATISTIQUE FINAL (Garanti <= 100%)
            total_perc = (np.sum(mask_red) / area_ref * 100)

            # 5. Macro-Void (CYAN)
            v_max_area, v_max_poly = 0, None
            red_cnts, _ = cv2.findContours(mask_red.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for rc in red_cnts:
                area = cv2.contourArea(rc)
                t_m = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(t_m, [rc], -1, 255, 2)
                # Exclusion si touche un bord ou un via
                if not np.any((t_m > 0) & (hol_adj > 0)) and not np.any((t_m > 0) & (env_adj == 0)):
                    if area > v_max_area:
                        v_max_area, v_max_poly = area, rc

            # 6. Rendu Visuel (Transparence des Vias)
            res_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            
            # Calque de couleur
            res_rgb[z_utile] = [255, 255, 0] # Jaune (Soudure)
            res_rgb[mask_red] = [255, 0, 0]  # Rouge (Manque)
            
            # Forçage transparence des vias (Image RX brute)
            res_rgb[hol_adj > 0] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)[hol_adj > 0]
            
            if v_max_poly is not None:
                cv2.drawContours(res_rgb, [v_max_poly], -1, [0, 255, 255], 2)

            # 7. Sauvegarde
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            st.session_state.batch_history.append({
                "Fichier": rx_f.name,
                "Total_%": round(min(total_perc, 100.0), 2),
                "Void_Max_%": round((v_max_area / area_ref * 100), 3),
                "img_bytes": enc.tobytes()
            })
            progress.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    st.subheader("📊 Résultats d'analyse")
    st.dataframe(apply_table_style(df.drop(columns=['img_bytes'])), use_container_width=True)
    
    # Export ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport_analyse.csv", df.drop(columns=['img_bytes']).to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    
    st.download_button("📥 Télécharger ZIP Complet", zip_buf.getvalue(), "batch_results.zip", use_container_width=True)

    with st.expander("👁️ Galerie des pièces (Vias transparents)", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
