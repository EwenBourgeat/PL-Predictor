# PL Match Predictor

Prédiction des résultats de Premier League (victoire domicile / nul / victoire extérieur) à partir de données historiques.

## Stack

Python · pandas · scikit-learn · XGBoost · Streamlit

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Entraîner les modèles
python train.py

# Lancer l'application
streamlit run app.py
```

## Données

Fichiers CSV issus de [football-data.co.uk](https://www.football-data.co.uk), saisons 2018-19 à 2024-25, à placer dans le dossier `data/`.
