import streamlit as st
import os
import joblib
import numpy as np
import cv2
import io
from datetime import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Trainer - Nomenclature Pro", layout="wide")

st.title("🧠 Entraînement & Nomenclature des Modèles")

# --- CONFIGURATION DU MODÈLE ET NOMENCLATURE ---
with st.sidebar:
    st.header("📋 Informations Modèle")
    
    # 1. Nom du produit
    nom_produit = st.text_input("1. Nom du Produit", value="PRODUIT").upper().replace(" ", "_")
    
    # 2. Repère Topo
    repere_topo = st.text_input("2. Repère Topo (ex: U12)", value="REPERE").upper().replace(" ", "_")
    
    # 3. Choix de la machine
    type_machine = st.radio("3. Source des images", ["LABOMSL", "AXI"])
    
    st.divider()
    st.header("⚙️ Paramètres IA")
    n_estimators = st.slider("Nombre d'arbres", 10, 150, 50, 10)

# Génération dynamique du nom de fichier
date_heure = datetime.now().strftime("%Y%m%d_%H%M%S")
nom_final_modele = f"{nom_produit}_{repere_topo}_{type_machine}_{date_heure}.joblib"

st.info(f"📂 Format de sortie : `{nom_final_modele}`")

# --- CHARGEMENT DES DONNÉES ---
col1, col2 = st.columns(2)
with col1:
    uploaded_images = st.file_uploader("🖼️ Images RX d'entraînement", accept_multiple_files=True)
with col2:
    uploaded_labels = st.file_uploader("🏷️ Masques Labels correspondants", accept_multiple_files=True)

# --- LOGIQUE D'ENTRAÎNEMENT ---
if st.button("🚀 Lancer l'Apprentissage", use_container_width=True, type="primary"):
    if not uploaded_images or not uploaded_labels:
        st.error("❌ Veuillez charger les images et les labels.")
    elif len(uploaded_images) != len(uploaded_labels):
        st.error(f"❌ Déséquilibre : {len(uploaded_images)} images vs {len(uploaded_labels)} labels.")
    else:
        try:
            # On s'assure que les fichiers sont triés pour correspondre
            images_sorted = sorted(uploaded_images, key=lambda x: x.name)
            labels_sorted = sorted(uploaded_labels, key=lambda x: x.name)
            
            all_features = []
            all_labels = []
            
            status_msg = st.empty()
            progress_bar = st.progress(0)
            
            for i, (img_file, lbl_file) in enumerate(zip(images_sorted, labels_sorted)):
                status_msg.text(f"Traitement de {img_file.name}...")
                
                # Chargement Image via votre moteur
                img = engine.load_gray(img_file)
                
                # Chargement Label
                lbl_raw = cv2.imdecode(np.frombuffer(lbl_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                lbl_resized = cv2.resize(lbl_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Extraction des caractéristiques
                feats = engine.compute_features(img)
                
                # Cible : On considère que les zones sombres (<128) dans le label sont les manques (classe 0)
                # Note: Ajustez cette logique si vos labels utilisent une autre convention
                target = (lbl_resized < 128).astype(int).flatten()
                
                all_features.append(feats.reshape(-1, feats.shape[-1]))
                all_labels.append(target)
                
                progress_bar.progress((i + 1) / len(images_sorted))

            status_msg.text("🌳 Construction du Random Forest... (Patientez)")
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)

            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            clf.fit(X, y)

            # --- SAUVEGARDE EN MÉMOIRE ---
            model_buffer = io.BytesIO()
            joblib.dump(clf, model_buffer)
            
            st.success(f"🎉 Entraînement terminé !")
            st.balloons()
            
            # Bouton de téléchargement avec le nom formaté
            st.download_button(
                label=f"📥 Télécharger {nom_final_modele}",
                data=model_buffer.getvalue(),
                file_name=nom_final_modele,
                mime="application/octet-stream",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Erreur critique : {e}")
