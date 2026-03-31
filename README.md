# ecommerce-dynamic-pricing-optimizer
## 1. L'objectif : 
Est de trouver juste le prix de produits en temps réel pour maximiser le gain de l'entreprise.
# 2. Le concept clé : L'Élasticité-Prix
Comment le client réagit à la variation du prix ?
Produit Élastique : le prix augemente de 1€, les ventes chutent de 50% (ex: une marque de pâtes générique).

Produit Inélastique : le prix augmente de 5€, les gens achètent toujours autant (ex: l'essence ou un médicament précis).
<img width="2000" height="2000" alt="image" src="https://github.com/user-attachments/assets/85c86f8f-12b6-49f7-a93f-a50e09b86193" />

## 3. La problématique:
Si je change mon prix de X, combien vais-je vendre de Y, et combien de profit va-t-il me rester à la fin ?????????


# Retail Demand Forecasting & Price Optimization

## Contexte du Projet
Dans un secteur du retail hautement concurrentiel, fixer le bon prix est crucial. Ce projet déploie un modèle de Machine Learning capable de prédire le volume des ventes (demande) d'un produit en fonction de son prix, de la saisonnalité et de la pression concurrentielle. 


## Fonctionnalités (L'Interface)
* **Simulation de Prix :** Entrez un prix cible et découvrez immédiatement l'impact estimé sur les volumes de ventes.
* **Analyse Concurrentielle :** Calcul automatique de votre positionnement par rapport à vos 3 concurrents principaux (Price Ratios & Differences).
* **Tableau de Bord Interactif :** Une interface web fluide développée avec [Streamlit / FastAPI] pour une utilisation métier simplifiée.

##  Architecture Machine Learning
Le cœur du projet repose sur un algorithme **LightGBM** optimisé, choisi pour sa performance sur les données tabulaires de taille moyenne.

* **Feature Engineering Avancé :** * Création de variables de décalage temporel (`lag_price`) pour capter l'historique.
  * Extraction de la saisonnalité (Mois, Année, Vacances, Week-ends).
  * Création de ratios psychologiques (Différence par rapport au prix moyen du marché).
* **Traitement des Valeurs Extrêmes :** Utilisation de la transformation logarithmique (`log1p`) ou de la distribution de Poisson pour stabiliser l'apprentissage face aux pics de ventes soudains.
* **Data Augmentation (Jittering) :** Injection contrôlée d'un bruit gaussien (3%) sur les variables continues d'entraînement pour doubler la taille du dataset et accroître la robustesse du modèle.
* **Optimisation :** Recherche exhaustive des hyperparamètres via `GridSearchCV`.

## Stack Technique
* **Langage :** Python 3.9+
* **Data Manipulation :** Pandas, NumPy
* **Machine Learning :** Scikit-Learn, LightGBM, XGBoost
* **Déploiement / API :** FastAPI & Uvicorn
* **Interface Web :** Streamlit

## Comment lancer le projet en local

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/diabateyoussouf/ecommerce-dynamic-pricing-optimizer.git
