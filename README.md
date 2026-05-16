# Machine Learning Final Project

## Resume
Ce projet analyse des donnees de crimes (Chicago) et d'interventions (Seattle) avec des notebooks Jupyter. Il inclut EDA, feature engineering, clustering et modeles supervises.

## Contenu
- Notebooks : analyses et entrainements
- Donnees : fichiers CSV a la racine
- Artifacts : modeles et sorties generes (dossier artifacts)

## Notebooks
- `notebooks/chicago_crime_ML_v3.ipynb` : notebook source complet
- `notebooks/01_chicago_crime_eda.ipynb` : EDA dediee
- `notebooks/02_chicago_crime_predictions.ipynb` : pipeline predictions et sauvegarde modeles

## Equipe
- Alexis Arnaud
- Khadidja Addi
- Thomas Le Bourdon
- Marion Gomes De Sousa

## Execution rapide
1) Creer et synchroniser l'environnement Python avec uv

```bash
uv sync --group dev
```

2) Lancer JupyterLab

```bash
uv run jupyter lab
```

3) Lancer le simulateur Streamlit

```bash
uv run streamlit run src/simulator/app.py
```

4) Ouvrir un notebook et executer les cellules

## Structure
- Notebooks : *.ipynb
- Donnees : *.csv
- Artifacts : artifacts/
