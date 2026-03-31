# 📈 E-Commerce Dynamic Pricing Optimizer

## 1. 🎯 L'Objectif
L'objectif principal de ce projet est de trouver le **juste prix** des produits en temps réel afin de maximiser les profits de l'entreprise. 

Dans un secteur du retail hautement concurrentiel, fixer le bon prix est crucial. Ce projet déploie un modèle de Machine Learning capable de prédire le volume des ventes (la demande) d'un produit en fonction de son prix, de la saisonnalité, et de la pression concurrentielle.

## 2. 🧠 Le Concept Clé : L'Élasticité-Prix
Toute la logique repose sur une question fondamentale : **Comment le client réagit-il à la variation du prix ?**

* **Produit Élastique :** Si le prix augmente de 1€, les ventes chutent drastiquement, par exemple de 50% (ex: une marque de pâtes générique, où le client ira chez le concurrent).
* **Produit Inélastique :** Si le prix augmente de 5€, les clients achètent toujours autant (ex: l'essence ou un médicament précis, car c'est un besoin essentiel).

#### Graphe de l'élasticité (dQ / dP)
<img width="600" alt="Graphe Élasticité" src="https://github.com/user-attachments/assets/85c86f8f-12b6-49f7-a93f-a50e09b86193" />

## 3. ❓ La Problématique
Le simulateur répond mathématiquement à la question que tout e-commerçant se pose :
> *"Si je modifie mon prix de X €, combien vais-je vendre d'unités Y, et quel sera mon bénéfice net final ?"*

---

## 💻 Fonctionnalités (L'Interface)
* **Simulation de Prix :** Entrez un prix cible et découvrez immédiatement l'impact estimé sur les volumes de ventes.
* **Analyse Concurrentielle :** Calcul automatique de votre positionnement par rapport à vos 3 concurrents principaux (ratios de prix et différences directes).
* **Tableau de Bord Interactif :** Une interface web fluide développée avec Streamlit pour une utilisation métier simplifiée et visuelle.
  <img width="1859" height="967" alt="Capture d’écran du 2026-03-31 16-10-33" src="https://github.com/user-attachments/assets/9ad7b4c4-b9a0-4a85-b40e-ccdc15f9daa9" />

  ### lien : https://ecommerce-dynamic-pricing-optimizer.streamlit.app/


## ⚙️ Architecture Machine Learning
Le cœur du projet repose sur un algorithme **LightGBM** optimisé, choisi pour sa performance exceptionnelle sur les données tabulaires de taille moyenne.

* **Feature Engineering Avancé :** * Création de variables de décalage temporel (`lag_price`) pour capter l'historique de tarification.
  * Extraction de la saisonnalité (Mois, Année, Vacances, Week-ends).
  * Création de ratios psychologiques (Différence par rapport au prix moyen du marché).
* **Traitement des Valeurs Extrêmes :** Utilisation de la transformation logarithmique (`log1p`) pour stabiliser l'apprentissage de l'IA face aux pics de ventes soudains.
* **Data Augmentation (Jittering) :** Injection contrôlée d'un bruit gaussien (3%) sur les variables continues d'entraînement pour simuler de nouveaux scénarios et accroître la robustesse du modèle.
* **Optimisation :** Recherche exhaustive des meilleurs hyperparamètres via `GridSearchCV`.

## 🛠️ Stack Technique
* **Langage :** Python 3.9+
* **Data Manipulation :** Pandas, NumPy
* **Machine Learning :** Scikit-Learn, LightGBM
* **Interface Web :** Streamlit

---

## 🚀 Comment lancer le projet en local

1. Clonez ce dépôt :
   ```bash
   git clone [https://github.com/diabateyoussouf/ecommerce-dynamic-pricing-optimizer.git](https://github.com/diabateyoussouf/ecommerce-dynamic-pricing-optimizer.git)
