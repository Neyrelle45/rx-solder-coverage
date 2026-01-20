import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import analyse_rx_soudure as engine  # Ton script de 800 lignes

st.set_page_config(page_title="Station Analyse RX - ALL Circuits", layout="wide")

# --- Initialisation de l'historique ---
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🛠️ Paramètres & Contraste")

# 1. AJOUT DU SLIDER DE CONTRASTE
# Ce paramètre sera envoyé à ta fonction load_gray
st.sidebar.markdown("### 👁️ Visibilité")
contrast_val = st.sidebar.slider(
    "Ajustement Contraste (CLAHE)", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.0, 
    step=0.5,
    help="Augmentez si l'image est trop sombre ou 'plate'. 2.0 est la valeur standard."
)

st.sidebar.divider()

# 2. CHARGEMENT DU MODÈLE
st.sidebar.subheader("🧠 Modèle IA")
model_file = st.sidebar.file_uploader("Charger le fichier .joblib", type=["joblib"])

if model_file:
    # Chargement du classifieur
    clf = joblib.load(model_file)
    st.sidebar.success("Modèle chargé !")

    st.header("🔍 Analyse de Soudure")
    
    col_u, col_m = st.columns(2)
    with col_u:
        rx_upload = st.file_uploader("Étape 1 : Image RX", type=["png", "jpg", "jpeg", "tif"])
    with col_m:
        mask_upload = st.file_uploader("Étape 2 : Masque (Vert/Noir)", type=["png", "jpg"])

    if rx_upload and mask_upload:
        # --- TRAITEMENT DE L'IMAGE AVEC LE CURSEUR ---
        # On utilise ta fonction load_gray modifiée (ou on applique le filtre ici)
        img_gray = engine.load_gray(rx_upload, contrast_limit=contrast_val)
        
        if img_gray is not None:
            H, W = img_gray.shape
            
            # Affichage de l'image traitée pour vérification du contraste
            st.subheader("🖼️ Prévisualisation (Contraste)")
            st.image(img_gray, caption=f"Image RX avec CLAHE réglé sur {contrast_val}", use_container_width=True)

            # --- ALIGNEMENT (Paramètres pour tes fonctions d'origine) ---
            st.sidebar.subheader("🕹️ Alignement du Masque")
            tx = st.sidebar.number_input("Translation X (px)", value=0.0)
            ty = st.sidebar.number_input("Translation Y (px)", value=0.0)
            rot = st.sidebar.slider("Rotation (°)", -180.0, 180.0, 0.0)
            scale = st.sidebar.slider("Échelle", 0.8, 1.2, 1.0, 0.001)

            # Sauvegarde temporaire du masque pour que ton script puisse le lire
            with open("temp_mask.png", "wb") as f:
                f.write(mask_upload.getbuffer())

            # Utilisation de ta fonction d'origine pour extraire la zone
            # (Adapté de ta logique VERT - NOIR)
            zone_utile, green, black = engine.compute_zone_and_holes("temp_mask.png")
            
            # Redimensionnement si nécessaire
            if zone_utile.shape != (H, W):
                zone_utile = cv2.resize(zone_utile, (W, H), interpolation=cv2.INTER_NEAREST)

            # Application de la transformation (Similarity)
            # On cherche le centre pour la rotation comme dans ton script
            ys, xs = np.where(green > 0)
            cx, cy = (xs.mean(), ys.mean()) if ys.size > 0 else (W/2, H/2)
            M = engine.compose_similarity(scale, rot, tx, ty, cx, cy)
            zone_adj = cv2.warpAffine(zone_utile, M, (W, H), flags=cv2.INTER_NEAREST)

            # --- ANALYSE IA ---
            with st.spinner("Analyse des pixels par l'IA..."):
                # Extraction des caractéristiques (tes 3 filtres d'origine)
                feats = engine.compute_features(img_gray)
                # Prédiction
                flat_feats = feats.reshape(-1, feats.shape[-1])
                pred_flat = clf.predict(flat_feats)
                pred_map = pred_flat.reshape(H, W)

            # --- CALCULS DE RÉSULTATS ---
            # Uniquement dans la zone utile alignée
            solder_in_zone = np.sum((pred_map == 1) & (zone_adj > 0))
            total_zone_px = np.sum(zone_adj > 0)
            
            if total_zone_px > 0:
                coverage_pct = (solder_in_zone / total_zone_px) * 100
                missing_pct = 100 - coverage_pct
            else:
                coverage_pct = missing_pct = 0

            # --- AFFICHAGE ---
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Couverture de Soudure", f"{coverage_pct:.2f} %")
            with c2:
                # Couleur dynamique pour le taux de manque
                color = "normal" if missing_pct < 25 else "inverse"
                st.metric("Taux de Manque (VIDES)", f"{missing_pct:.2f} %", delta=f"{missing_pct:.1f}%", delta_color=color)

            # Overlay visuel
            # JAUNE = Brasure présente, ROUGE = Manque dans la zone utile
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            mask_red = (zone_adj > 0) & (pred_map == 0)
            mask_yellow = (zone_adj > 0) & (pred_map == 1)
            
            overlay[mask_red] = [255, 0, 0]    # ROUGE
            overlay[mask_yellow] = [255, 255, 0] # JAUNE

            st.subheader("🎭 Résultat Visuel")
            st.image(overlay, caption="Jaune : Soudure OK | Rouge : Vide/Manque", use_container_width=True)

            # --- BOUTON ENREGISTRER ---
            if st.button("📥 Enregistrer dans l'historique"):
                st.session_state.history.append({
                    "Horodatage": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Fichier": rx_upload.name,
                    "Couverture %": round(coverage_pct, 2),
                    "Manque %": round(missing_pct, 2),
                    "Contraste utilisé": contrast_val
                })
                st.success("Résultat ajouté au tableau ci-dessous.")

# --- TABLEAU RÉCAPITULATIF ---
if st.session_state.history:
    st.divider()
    st.subheader("📊 Historique de la session")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)
    
    # Export CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Télécharger le rapport CSV", csv, "rapport_analyse.csv", "text/csv")
else:
    st.info("Chargez une image et un masque pour commencer l'analyse.")
