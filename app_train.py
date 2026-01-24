import streamlit as st
import os
import joblib
import numpy as np
import cv2
import io
import tempfile
from datetime import datetime
import analyse_rx_soudure as engine

st.set_page_config(page_title="RX Trainer - Stable", layout="wide")

st.title("🧠 Entraînement du Modèle IA")

# --- SIDEBAR : Nomenclature ---
with st.sidebar:
    st.header("📋 Identification")
    prod = st.text_input("Produit", value="PRODUIT").upper().replace(" ", "_")
    topo = st.text_input("Repère Topo", value="U1").upper().replace(" ", "_")
    source = st.radio("Source", ["LABOMSL", "AXI"])
    st.divider()
    n_trees = st.slider("Nombre d'arbres", 10, 100, 50)

# Nom de fichier dynamique
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{prod}_{topo}_{source}_{date_str}.joblib"

# --- CHARGEMENT ---
c1, c2 = st.columns(2)
with c1:
    imgs = st.file_uploader("🖼️ Images RX", accept_multiple_files=True)
with c2:
    lbls = st.file_uploader("🏷️ Labels", accept_multiple_files=True)

if st.button("🚀 Lancer l'Apprentissage", use_container_width=True, type="primary"):
    if not imgs or not lbls or len(imgs) != len(lbls):
        st.error("❌ Erreur : Chargez un nombre identique d'images et de labels.")
    else:
        try:
            # Tri pour correspondance parfaite
            imgs.sort(key=lambda x: x.name)
            lbls.sort(key=lambda x: x.name)
            
            X_list, y_list = [], []
            prog = st.progress(0)
            status = st.empty()

            for i, (f_img, f_lbl) in enumerate(zip(imgs, lbls)):
                status.text(f"Analyse : {f_img.name}...")
                
                # Image
                img = engine.load_gray(f_img)
                # Label (Soudure = Blanc, Manque/Void = Noir)
                l_raw = cv2.imdecode(np.frombuffer(f_lbl.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                l_res = cv2.resize(l_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Features
                feats = engine.compute_features(img)
                
                # Cible : On apprend à détecter le "Noir" (<128) comme étant la classe 'Manque'
                target = (l_res < 128).astype(int).flatten()
                
                X_list.append(feats.reshape(-1, feats.shape[-1]))
                y_list.append(target)
                prog.progress((i + 1) / len(imgs))

            status.text("🌳 Calcul du Random Forest...")
            X = np.vstack(X_list)
            y = np.concatenate(y_list)

            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
            clf.fit(X, y)

            # --- SAUVEGARDE SÉCURISÉE (Via fichier temporaire pour éviter le crash) ---
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                joblib.dump(clf, tmp.name)
                with open(tmp.name, "rb") as f:
                    model_bytes = f.read()
            os.unlink(tmp.name) # Nettoyage

            st.success(f"🎉 Modèle prêt : {filename}")
            st.download_button(
                label=f"📥 Télécharger {filename}",
                data=model_bytes,
                file_name=filename,
                mime="application/octet-stream"
            )

        except Exception as e:
            st.error(f"❌ Erreur critique : {str(e)}")
            st.info("Astuce : Vérifiez que vos images et labels ont bien la même taille.")
