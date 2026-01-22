import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import io
import zipfile
import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Expert - Batch Mode Pro", layout="wide")

# --- PERSISTENCE DE L'INTERFACE ---
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

def highlight_max(s):
    return ['background-color: #ffcccc' if v == s.max() and v > 0 else '' for v in s]

st.title("📦 Analyse de Série (Batch)")

# --- SIDEBAR FIXE ---
st.sidebar.title("⚙️ Paramètres")
model_file = st.sidebar.file_uploader("1. Charger modèle IA (.joblib)", type=["joblib"])
contrast_val = st.sidebar.slider("2. Contraste (CLAHE)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🕹️ Alignement commun")
tx = st.sidebar.number_input("Translation X", value=0)
ty = st.sidebar.number_input("Translation Y", value=0)
rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
sc = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0)

if st.sidebar.button("🗑️ Vider les résultats", use_container_width=True):
    st.session_state.batch_history = []
    st.rerun()

# --- ZONE DE CHARGEMENT ---
st.divider()
col_u, col_m = st.columns(2)
with col_u:
    uploaded_rx = st.file_uploader("Images RX (Série)", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
with col_m:
    mask_file = st.file_uploader("Masque de référence (Unique)", type=["png", "jpg"])

# --- BOUTON DE LANCEMENT (Sorti de toute condition pour ne jamais disparaître) ---
st.divider()
trigger_analysis = st.button("🚀 Lancer l'analyse de la série", use_container_width=True, type="primary")

if trigger_analysis:
    if not model_file or not uploaded_rx or not mask_file:
        st.error("Veuillez charger le modèle, les images et le masque avant de lancer.")
    else:
        clf = joblib.load(model_file)
        st.session_state.batch_history = [] 
        
        # LECTURE IDENTIQUE A APP.PY (Correction du bug des vias)
        mask_bytes = mask_file.getvalue()
        mask_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_COLOR)
        b_m, g_m, r_m = cv2.split(mask_raw)
        
        # Définition stricte des masques
        m_green_base = (g_m > 100).astype(np.uint8)
        # Vias = Noir DANS le vert (Exclusion)
        m_black_base = ((b_m < 50) & (g_m < 50) & (r_m < 50) & (m_green_base > 0)).astype(np.uint8)

        progress_bar = st.progress(0)
        
        for idx, rx_file in enumerate(uploaded_rx):
            # 1. Image Grise
            img_gray = engine.load_gray(rx_file, contrast_limit=contrast_val)
            H, W = img_gray.shape

            # 2. Resynchronisation des masques
            m_g_res = cv2.resize(m_green_base, (W, H), interpolation=cv2.INTER_NEAREST)
            m_b_res = cv2.resize(m_black_base, (W, H), interpolation=cv2.INTER_NEAREST)
            
            M = engine.compose_similarity(sc, rot, float(tx), float(ty), W/2, H/2)
            env_adj = cv2.warpAffine(m_g_res, M, (W, H), flags=cv2.INTER_NEAREST)
            hol_adj = cv2.warpAffine(m_b_res, M, (W, H), flags=cv2.INTER_NEAREST)
            
            # ZONE UTILE = VERT sans NOIR
            z_utile = (env_adj > 0) & (hol_adj == 0)
            area_total_px = np.sum(z_utile)

            # 3. IA
            features = engine.compute_features(img_gray)
            probs = clf.predict_proba(features.reshape(-1, features.shape[-1]))
            pred_map = np.argmax(probs, axis=1).reshape(H, W)

            # 4. FILTRAGE MICRO-VOIDS (Bulles < 0.1% -> Jaune)
            void_raw = ((pred_map == 0) & z_utile).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(void_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_voids_mask = np.zeros((H, W), dtype=bool)
            for c in cnts:
                area = cv2.contourArea(c)
                if area_total_px > 0 and (area / area_total_px * 100) >= 0.1:
                    cv2.drawContours(valid_voids_mask.view(np.uint8), [c], -1, 1, -1)

            # Affichage : Soudure IA + Micro-bulles = Jaune
            display_solder = z_utile & (~valid_voids_mask)
            display_voids = valid_voids_mask
            
            # 5. VOID MAX (Cercle Cyan) - Uniquement dans la soudure
            max_v_area = 0
            max_v_poly = None
            # On cherche les contours du rouge
            v_cnts, _ = cv2.findContours(display_voids.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for vc in v_cnts:
                area = cv2.contourArea(vc)
                # On vérifie si ce vide est entouré de jaune (void bulle) et ne touche pas l'exclusion (noir)
                temp_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(temp_mask, [vc], -1, 255, -1)
                # Si le vide touche une zone d'exclusion (vias), on ne le prend pas comme Void Max
                if not np.any((temp_mask > 0) & (hol_adj > 0)):
                    if area > max_v_area:
                        max_v_area = area
                        max_v_poly = vc

            # 6. Statistiques
            missing_pct = (np.sum(display_voids) / area_total_px * 100) if area_total_px > 0 else 0
            v_max_pct = (max_v_area / area_total_px * 100) if area_total_px > 0 else 0

            # 7. Overlay
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay[display_solder] = [255, 255, 0] # Jaune
            overlay[display_voids] = [255, 0, 0]    # Rouge
            if max_v_poly is not None:
                cv2.drawContours(overlay, [max_v_poly], -1, [0, 255, 255], 2) # Cyan

            # 8. Sauvegarde
            _, img_jpg = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            st.session_state.batch_history.append({
                "Fichier": rx_file.name,
                "Total_%": round(missing_pct, 2),
                "Void_Max_%": round(v_max_pct, 3),
                "img_bytes": img_jpg.tobytes()
            })
            progress_bar.progress((idx + 1) / len(uploaded_rx))

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.batch_history:
    df = pd.DataFrame(st.session_state.batch_history)
    df_display = df.drop(columns=['img_bytes'])
    
    st.subheader("📊 Tableau récapitulatif")
    st.dataframe(df_display.style.apply(highlight_max, subset=['Total_%']), use_container_width=True)

    # EXPORT ZIP (Restauré)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("rapport_analyse.csv", df_display.to_csv(index=False))
        for item in st.session_state.batch_history:
            z.writestr(f"images/{item['Fichier']}.jpg", item['img_bytes'])
    
    col_z1, col_z2 = st.columns(2)
    col_z1.download_button("📥 Télécharger ZIP Complet", zip_buf.getvalue(), "batch_results.zip", "application/zip", use_container_width=True)
    col_z2.download_button("📄 Télécharger CSV seul", df_display.to_csv(index=False), "rapport.csv", "text/csv", use_container_width=True)

    st.subheader("👁️ Galerie des résultats")
    with st.expander("Ouvrir la galerie", expanded=True):
        grid = st.columns(4)
        for i, item in enumerate(st.session_state.batch_history):
            grid[i % 4].image(item['img_bytes'], caption=f"{item['Fichier']} ({item['Total_%']}%)")
