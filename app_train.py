import streamlit as st
import os
import datetime
import joblib
import analyse_rx_soudure as engine
from glob import glob

st.set_page_config(page_title="RX Expert - Training Studio", layout="centered")

st.title("🎓 Entraînement de Modèle RX")
st.info("Cette interface permet de créer un modèle IA spécifique à un produit et un repère topo.")

# --- 1. SAISIE DES MÉTADONNÉES ---
with st.expander("📝 Informations Modèle", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        nom_produit = st.text_input("Nom du Produit", placeholder="ex: OBC_Mainboard").upper().replace(" ", "_")
        repere_topo = st.text_input("Repère Topo", placeholder="ex: U12").upper().replace(" ", "_")
    with col2:
        origine = st.radio("Origine des images", ["LABO", "AXI"])
        n_estimators = st.slider("Nombre d'estimateurs (Précision)", 50, 300, 150)

# --- 2. CONFIGURATION DES DOSSIERS ---
with st.expander("📂 Sources des données", expanded=True):
    img_dir = st.text_input("Dossier des images RX", "./MyDrive/OBC_mainboard/rx_images")
    lbl_dir = st.text_input("Dossier des labels (_label.png)", "./MyDrive/OBC_mainboard/labels")
    models_dir = st.text_input("Dossier de destination des modèles", "./models")

# --- 3. GÉNÉRATION DU NOM DE FICHIER ---
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Format: NOMDUPRODUIT_REPERETOPO_LABOouAXI_YYYYMMDD_HHMMSS
model_name = f"{nom_produit}_{repere_topo}_{origine}_{now}.joblib"

st.subheader("📦 Futur nom du modèle :")
st.code(model_name)

# --- 4. BOUTON D'ENTRAÎNEMENT ---
if st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True):
    if not nom_produit or not repere_topo:
        st.error("Veuillez saisir le nom du produit et le repère topo.")
    else:
        # Vérification de l'existence des dossiers
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            st.error("Les dossiers sources sont introuvables. Vérifiez les chemins.")
        else:
            with st.status("Entraînement en cours...", expanded=True) as status:
                st.write("🔍 Analyse des paires Images/Labels...")
                
                # On utilise la logique de ton moteur corrigé
                # Mais on surcharge la sauvegarde pour utiliser notre nom personnalisé
                try:
                    # On appelle la fonction de ton moteur
                    # Note: on passe models_dir mais on gérera le renommage après
                    engine.train_model(
                        img_dir=img_dir,
                        lbl_dir=lbl_dir,
                        out_dir=models_dir,
                        n_estimators=n_estimators,
                        max_samples=5000
                    )
                    
                    # Le moteur crée par défaut 'model_rx.joblib', on le renomme
                    default_path = os.path.join(models_dir, "model_rx.joblib")
                    final_path = os.path.join(models_dir, model_name)
                    
                    if os.path.exists(default_path):
                        if os.path.exists(final_path): os.remove(final_path)
                        os.rename(default_path, final_path)
                        
                        status.update(label="✅ Entraînement terminé !", state="complete")
                        st.success(f"Modèle sauvegardé : {final_path}")
                        
                        # --- 5. BOUTON DE TÉLÉCHARGEMENT ---
                        with open(final_path, "rb") as f:
                            st.download_button(
                                label="📥 Télécharger le modèle (.joblib)",
                                data=f,
                                file_name=model_name,
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                    else:
                        st.error("Le moteur n'a pas généré de fichier. Vérifiez les logs console.")
                
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")

# --- 6. RÉCAPITULATIF DES DONNÉES DISPONIBLES ---
st.divider()
st.write("### 📊 État du dataset")
rx_count = len(glob(os.path.join(img_dir, "*.*")))
lbl_count = len(glob(os.path.join(lbl_dir, "*_label.*")))

c1, c2 = st.columns(2)
c1.metric("Images RX", rx_count)
c2.metric("Labels trouvés", lbl_count)
