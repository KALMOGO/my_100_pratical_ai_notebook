# 🧠 100 Notebooks d'Ingénieur IA pour la Santé

> Programme complet d'apprentissage en Intelligence Artificielle appliquée à la santé, de la régression linéaire aux transformers, avec PyTorch.

## 📖 À propos du projet

Ce repository contient 100 notebooks Jupyter progressifs couvrant l'ensemble des compétences requises pour devenir ingénieur en IA appliquée à la santé. Chaque notebook est conçu pour être **riche en contenu**, **pratique** et **directement applicable** à des problèmes réels.

**Créé par :** Étudiant en Master 2 Intelligence Artificielle appliquée à la Santé  
**Objectif :** Consolider mes connaissances théoriques et pratiques tout en constituant un portfolio professionnel  
**Durée estimée :** 12-18 mois (1-2 notebooks par semaine)

## 🎯 Public cible

- Étudiants en IA/Data Science souhaitant se spécialiser en santé
- Data Scientists en reconversion vers le médical
- Professionnels de santé s'intéressant à l'IA
- Toute personne motivée pour apprendre l'IA de manière structurée

## 📚 Structure du programme

### Bloc 1 : Fondations Python & Data Science (1-15)
- NumPy, Pandas, Matplotlib, Seaborn
- Web scraping (BeautifulSoup, Selenium)
- Feature engineering et pipelines ML
- Optimisation hyperparamètres

### Bloc 2 : Machine Learning Classique (16-30)
- Régression (linéaire, Ridge, Lasso, Logistique)
- SVM, Decision Trees, Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Clustering (K-Means, DBSCAN, GMM)
- Réduction dimensionnalité (PCA, t-SNE)

### Bloc 3 : Deep Learning avec PyTorch (31-45)
- Fondamentaux PyTorch (tenseurs, autograd)
- Architecture réseaux neurones
- Optimizers, régularisation, normalisation
- Transfer learning et fine-tuning
- Mixed precision training

### Bloc 4 : Computer Vision Médical (46-60)
- CNN classiques (AlexNet, VGG, ResNet, EfficientNet)
- Vision Transformers (ViT)
- Segmentation (U-Net, nnU-Net)
- Object Detection (YOLO, Faster R-CNN)
- GANs pour génération d'images médicales
- Imagerie 3D (CT scans, IRM)

### Bloc 5 : NLP pour Données Médicales (61-72)
- Preprocessing texte clinique
- Word embeddings, LSTM, Transformers
- BERT, BioBERT, ClinicalBERT
- NER médical, extraction relations
- Question Answering, résumés cliniques
- Chatbot médical

### Bloc 6 : Séries Temporelles Médicales (73-80)
- ARIMA, LSTM pour time series
- TCN, WaveNet
- Transformers pour séries temporelles
- Détection anomalies (ECG, EEG)
- Prédiction événements critiques

### Bloc 7 : Interprétabilité & Explainability (81-88)
- SHAP, LIME
- Grad-CAM, Integrated Gradients
- Attention visualization
- Counterfactual explanations
- Dashboard d'explainability

### Bloc 8 : Sujets Avancés & Production (89-100)
- Federated Learning
- Differential Privacy
- Quantization, ONNX
- Docker, MLflow, FastAPI
- CI/CD pour ML
- Monitoring en production
- Active Learning, AutoML

## 🛠️ Technologies utilisées

**Langages & Frameworks :**
- Python 3.9+
- PyTorch (framework principal)
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn

**Bibliothèques spécialisées :**
- Hugging Face Transformers
- Albumentations (augmentation)
- XGBoost, LightGBM, CatBoost
- SHAP, LIME (explainability)
- FastAPI (déploiement)
- MLflow (tracking)

**Outils :**
- Jupyter Notebook / JupyterLab
- Git & GitHub
- Docker
- Optuna (hyperparameter tuning)

## 📊 Datasets utilisés

Le programme couvre plus de 50 datasets différents, notamment :

**Datasets médicaux publics :**
- MIMIC-III (notes cliniques, signaux)
- NIH Chest X-ray14
- ISIC Skin Lesion
- PhysioNet (ECG, EEG)
- Diabetic Retinopathy Detection
- PubMed articles

**Datasets génériques :**
- MNIST, Fashion-MNIST, CIFAR-10
- ImageNet (subset)
- IMDB Reviews

**Sources :**
- UCI Machine Learning Repository
- Kaggle
- PhysioNet
- OpenML

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/100-notebooks-ia-sante.git
cd 100-notebooks-ia-sante

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter lab
```

## 📋 Prérequis

**Connaissances :**
- Python de base
- Mathématiques niveau licence (algèbre linéaire, probabilités)
- Motivation et curiosité !

**Matériel :**
- Ordinateur avec minimum 8 GB RAM
- GPU recommandé (mais pas obligatoire, Colab disponible)
- 50 GB d'espace disque

## 📖 Comment utiliser ce repository

1. **Progression linéaire recommandée** : Suivre l'ordre des notebooks (1 → 100)
2. **Chaque notebook contient :**
   - Introduction théorique
   - Code commenté et expliqué
   - Visualisations
   - Exercices pratiques
   - Ressources pour aller plus loin
3. **Personnalisation encouragée** : Adaptez les notebooks à vos besoins
4. **Partagez vos résultats** : Pull requests bienvenues !

## 📚 Ressources complémentaires

**Livres recommandés :**
- "Deep Learning" - Ian Goodfellow
- "Hands-On Machine Learning" - Aurélien Géron
- "Deep Learning with PyTorch" - Eli Stevens
- "Pattern Recognition and Machine Learning" - Christopher Bishop

**Cours en ligne :**
- Fast.ai Practical Deep Learning
- Stanford CS231n (Computer Vision)
- Stanford CS224n (NLP)
- PyTorch Official Tutorials

**Communautés :**
- PyTorch Forums
- Kaggle Discussions
- Reddit r/MachineLearning
- Papers with Code

## 🤝 Contribution

Les contributions sont les bienvenues ! Si vous souhaitez :
- Corriger des erreurs
- Améliorer du code
- Ajouter des explications
- Proposer de nouveaux notebooks

Merci de :
1. Fork le projet
2. Créer une branche (`git checkout -b amelioration/nouveau-notebook`)
3. Commit vos changements (`git commit -m 'Ajout notebook XYZ'`)
4. Push vers la branche (`git push origin amelioration/nouveau-notebook`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📧 Contact

Pour toute question ou suggestion :
- GitHub Issues : [Ouvrir une issue](https://github.com/votre-username/100-notebooks-ia-sante/issues)
- Email : votre.email@example.com
- LinkedIn : [Votre profil](https://linkedin.com/in/votre-profil)

## 🙏 Remerciements

- La communauté PyTorch pour leur excellent framework
- Kaggle et PhysioNet pour les datasets
- Tous les auteurs des papers et tutoriels utilisés
- L'IA (Claude) pour m'avoir aidé à structurer ce programme !

## ⭐ Soutenir le projet

Si ce repository vous aide dans votre apprentissage :
- ⭐ Mettez une étoile sur GitHub
- 🔄 Partagez avec vos collègues
- 💬 Laissez vos retours dans les issues
- 🤝 Contribuez avec vos propres notebooks

---

**"Un homme qui veut apprendre peut y arriver s'il se fait aider par l'IA."**

*Dernière mise à jour : Décembre 2025*
