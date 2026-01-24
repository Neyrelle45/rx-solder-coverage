import streamlit as st
import os
import joblib
import numpy as np
import cv2
from datetime import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Trainer Cloud", layout="wide")

st.title("🧠 Entraîneur de Modèle RX (Mode Upload)")
st.info("Déposez vos paires (Image + Label) pour générer votre modèle .joblib")

# --- PARAMÈTRES ---
with st.sidebar:
    st.header("⚙️ Configuration")
    n_estimators = st.slider("Nombre d'arbres", 10, 150, 50, 10)
    model_name = st.text_input("Nom du modèle", f"modele_rx_{datetime.now().strftime('%Y%m%d')}.joblib")

# --- CHARGEMENT DES DONNÉES ---
col1, col2 = st.columns(2)

with col1:
    uploaded_images = st.file_uploader("🖼️ Images RX d'entraînement", accept_multiple_files=True)
with col2:
    uploaded_labels = st.file_uploader("🏷️ Masques Labels correspondants", accept_multiple_files=True)

# --- LOGIQUE D'ENTRAÎNEMENT ---
if st.button("🚀 Lancer l'Apprentissage", use_container_width=True, type="primary"):
    if not uploaded_images or not uploaded_labels:
        st.error("❌ Vous devez charger des images ET des labels.")
    elif len(uploaded_images) != len(uploaded_labels):
        st.error(f"❌ Déséquilibre : {len(uploaded_images)} images vs {len(uploaded_labels)} labels.")
    else:
        try:
            st.write("⏳ Préparation des données et extraction des caractéristiques...")
            
            # On trie pour s'assurer que image_1 correspond à label_1
            uploaded_images.sort(key=lambda x: x.name)
            uploaded_labels.sort(key=lambda x: x.name)
            
            # --- SIMULATION DE LA LOGIQUE DE TRAIN ---
            # Note : On adapte ici pour traiter les fichiers en mémoire
            all_features = []
            all_labels = []
            
            progress_bar = st.progress(0)
            
            for i, (img_file, lbl_file) in enumerate(zip(uploaded_images, uploaded_labels)):
                # Chargement
                img = engine.load_gray(img_file)
                # Chargement du label (on suppose que c'est une image où le noir=manque)
                lbl_raw = cv2.imdecode(np.frombuffer(lbl_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                lbl_resized = cv2.resize(lbl_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Extraction des caractéristiques via votre moteur
                feats = engine.compute_features(img)
                
                # Mise à plat pour Scikit-Learn
                # On définit ici la cible (ex: pixel noir dans label = classe 0)
                # Ajustez selon la logique de votre moteur analyse_rx_soudure.py
                target = (lbl_resized < 128).astype(int).flatten()
                
                all_features.append(feats.reshape(-1, feats.shape[-1]))
                all_labels.append(target)
                
                progress_bar.progress((i + 1) / len(uploaded_images))

            # Concaténation
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)

            st.write(f"🌳 Entraînement du Random Forest ({n_estimators} arbres)...")
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            clf.fit(X, y)

            # --- SAUVEGARDE ET TÉLÉCHARGEMENT ---
            model_buffer = io.BytesIO()
            joblib.dump(clf, model_buffer)
            
            st.success("🎉 Modèle entraîné avec succès !")
            st.download_button(
                label="📥 Télécharger le modèle .joblib",
                data=model_buffer.getvalue(),
                file_name=model_name,
                mime="application/octet-stream"
            )

        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement : {e}")
