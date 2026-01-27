import streamlit as st
import cv2
import numpy as np
import analyse_rx_soudure as engine 

st.set_page_config(page_title="RX Expert - Analyse Robuste", layout="wide")

st.sidebar.title("🛠️ Paramètres d'Analyse")
# Le curseur devient un seuil de sensibilité pour isoler les bulles du bruit
sensibilite = st.sidebar.slider("Sensibilité détection (Bruit)", 10, 100, 40)
contrast_val = st.sidebar.slider("Contraste (CLAHE)", 0.0, 5.0, 2.0, 0.1)

st.header("🔍 Analyse par Segmentation de Contours")

c_u, c_m = st.columns(2)
with c_u: rx_upload = st.file_uploader("1. Image RX", type=["png", "jpg", "jpeg", "tif"])
with c_m: mask_upload = st.file_uploader("2. Masque de référence", type=["png", "jpg"])

if rx_upload and mask_upload:
    # --- CHARGEMENT & PRÉ-TRAITEMENT ---
    img_raw = cv2.imdecode(np.frombuffer(rx_upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # On lisse l'image pour supprimer le bruit granuleux qui trompait l'IA
    img_blurred = cv2.GaussianBlur(img_raw, (5, 5), 0)
    
    if contrast_val > 0:
        clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8,8))
        img_proc = clahe.apply(img_blurred)
    else:
        img_proc = img_blurred

    H, W = img_proc.shape

    # --- ALIGNEMENT DU MASQUE ---
    mask_upload.seek(0)
    insp_raw = cv2.imdecode(np.frombuffer(mask_upload.read(), np.uint8), cv2.IMREAD_COLOR)
    r_r, g_r, b_r = cv2.split(cv2.cvtColor(insp_raw, cv2.COLOR_BGR2RGB))
    # On récupère la zone verte (zone de soudure théorique)
    z_utile = cv2.resize((g_r > 100).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # --- NOUVELLE LOGIQUE : DÉTECTION DES BULLES ---
    # Au lieu de l'IA pixel par pixel, on cherche les zones significativement plus claires 
    # que la moyenne locale dans la zone utile.
    thresh = cv2.adaptiveThreshold(img_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, -sensibilite/10)
    
    # On ne garde que ce qui est DANS le masque vert
    mask_manque = cv2.bitwise_and(thresh, thresh, mask=z_utile)
    
    # Nettoyage des petits points de bruit restants
    kernel = np.ones((3,3), np.uint8)
    mask_manque = cv2.morphologyEx(mask_manque, cv2.MORPH_OPEN, kernel)

    # --- RENDU VISUEL ---
    overlay = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
    
    # SOUDURE = BLEU FONCÉ (Fond de la zone utile)
    overlay[z_utile > 0] = [0, 50, 150]
    
    # MANQUES = ROUGE (Par-dessus)
    overlay[mask_manque > 0] = [255, 0, 0]

    # RECHERCHE DU VOID MAJEUR
    cnts, _ = cv2.findContours(mask_manque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_max_area = 0
    v_max_poly = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area > v_max_area:
            v_max_area = area
            v_max_poly = c

    if v_max_poly is not None:
        cv2.drawContours(overlay, [v_max_poly], -1, [0, 255, 255], 2)

    # --- AFFICHAGE ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Image Originale")
        st.image(img_raw, use_container_width=True)
        area_tot = np.sum(z_utile)
        ratio = (np.sum(mask_manque > 0) / area_tot * 100) if area_tot > 0 else 0
        st.metric("Taux de Manque (Rouge)", f"{ratio:.2f} %")
    
    with col2:
        st.subheader("🔬 Analyse Filtrée")
        st.image(overlay, use_container_width=True)
