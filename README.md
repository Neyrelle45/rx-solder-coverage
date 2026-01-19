
# RX Solder Coverage – Analyse brasure sous radios X

Application Streamlit pour :
- aligner un masque unique (manuel) ou des masques individuels (automatique),
- prédire la présence de brasure pixel par pixel (RandomForest),
- générer des overlays (JAUNE = brasure / ROUGE = manque),
- produire un CSV global avec les taux de manque.

## 🚀 Utilisation en ligne (Streamlit Cloud)

1. Crée un dépôt GitHub (bouton "New Repository").
2. Ajoute les fichiers :
   - streamlit_app.py
   - src/analyse_rx_soudure.py
   - requirements.txt
   - README.md
3. Sur https://streamlit.io/cloud :
   - "New App"
   - Choisir le dépôt
   - Choisir `streamlit_app.py`
   - Déployer

## 🧭 Démarrer l’application localement (optionnel)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
