# Projet Deep Learning - Vision par ordinateur

## 1. Description

Ce projet a été réalisé dans le cadre du module Intelligence Artificielle, séance 1 : Vision par ordinateur.

**Objectif :** Développer un système de classification automatique de radiographies pulmonaires pour détecter la COVID-19, distinguer les pneumonies virales et identifier les cas normaux, en utilisant le Transfer Learning avec l'architecture Xception.

---

## 2. Contexte

**Secteur :** Santé - Diagnostic médical - Imagerie médicale (Radiologie)

**Problématique :** La COVID-19 présente des symptômes radiologiques souvent similaires à ceux d'autres pneumonies virales sur les radiographies pulmonaires. Un diagnostic rapide et précis par imagerie est essentiel pour le triage des patients, le contrôle de l'infection et l'optimisation du traitement. Le projet vise à développer un outil d'aide à la décision basé sur le Deep Learning pour différencier automatiquement les cas de COVID-19 des autres pathologies pulmonaires et des cas sains.

**KPIs métiers :**
- **Recall COVID-19** : Minimiser les faux négatifs (patients COVID non détectés) - Métrique critique pour la santé publique
- **Precision globale** : Éviter les fausses alertes qui surchargent le système de santé
- **F1-Score** : Équilibre entre précision et rappel pour une performance globale robuste
- **AUC-ROC** : Capacité de discrimination entre les classes

---

## 3. Données

**Dataset :** COVID-19 Radiography Database  
**Lien :** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

**Taille :** 
- Total : ~18,000 images de radiographies pulmonaires
- COVID-19 : ~3,600 images
- Normal : ~10,200 images
- Viral Pneumonia : ~4,500 images
- Format : PNG/JPEG
- Résolution : Variable (redimensionnée à 299x299 pixels)

**Classes :**
1. COVID-19 (Pneumonie COVID)
2. Normal (Radiographies saines)
3. Viral Pneumonia (Pneumonie virale non-COVID)

**Prétraitements effectués :**
- Redimensionnement : 299x299 pixels (format d'entrée Xception)
- Normalisation : Préprocessing Xception (preprocess_input)
- Encodage : One-Hot Encoding des labels
- Split : 70% Train / 15% Validation / 15% Test (stratifié)
- Augmentation de données (train uniquement) :
  - Rotation : ±15°
  - Translation : 10% horizontal/vertical
  - Zoom : ±10%
  - Shear : 10%
  - Flip horizontal
- Gestion du déséquilibre : Class weights (inversement proportionnels aux fréquences)

---

## 4. Modèle

**Architecture :** Xception avec Transfer Learning

**Framework :** TensorFlow 2.15.0 / Keras

**Hyperparamètres :**

*Phase 1 (Base gelée) :*
- Epochs : 15
- Batch size : 32
- Learning rate : 0.001
- Optimizer : Adam
- Loss : Categorical Crossentropy

*Phase 2 (Fine-tuning) :*
- Epochs : 10
- Batch size : 32
- Learning rate : 1e-5 (0.00001)
- Optimizer : Adam
- Dégel : 30 dernières couches du modèle de base

**Callbacks :**
- EarlyStopping (patience=5)
- ReduceLROnPlateau (factor=0.5, patience=3)
- ModelCheckpoint (sauvegarde meilleur modèle)

**Structure du modèle :**
```
Input (299x299x3)
    ↓
Xception Base (pré-entraîné ImageNet, 22.9M params)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(3, softmax)
```

**Justification :** Xception a été choisi pour sa performance supérieure sur l'imagerie médicale grâce aux convolutions séparables en profondeur, son efficacité paramétrique et ses excellents résultats en Transfer Learning depuis ImageNet.

---

## 5. Résultats

### Métriques globales sur l'ensemble de test

**Accuracy :** 96.48%  
**Precision :** 96.57%  
**Recall :** 96.48%  
**F1-Score :** 96.52%  
**Loss :** 0.0981

### Métriques par classe

| Classe | Precision | Recall | F1-Score | Support | AUC-ROC |
|--------|-----------|--------|----------|---------|---------|
| **COVID-19** | 95.90% | 94.84% | 95.37% | 543 | 0.9954 |
| **Normal** | 97.39% | 97.45% | 97.42% | 1529 | 0.9929 |
| **Viral Pneumonia** | 91.30% | 93.56% | 92.42% | 202 | 0.9982 |
| **Weighted Avg** | 96.49% | 96.48% | 96.48% | 2274 | 0.9970 |

### Analyse des performances

**Points forts :**
- Recall COVID-19 à 94.84% : Seulement 28 cas COVID non détectés sur 543 (excellent pour la santé publique)
- AUC-ROC très élevés (>0.99) : Excellente capacité de discrimination
- Performance équilibrée sur les 3 classes
- Accuracy globale de 96.48% dépasse largement l'objectif

**Analyse des erreurs :**
- COVID → Normal : 27 confusions (cas limites ou images de faible qualité)
- COVID → Viral Pneumonia : 1 confusion (symptômes très similaires)
- Normal → COVID : 22 confusions (faux positifs acceptables car confirmés par PCR)
- Viral Pneumonia → Normal : 13 confusions (sous-estimation de la pathologie)

**KPIs métiers atteints :**
- Recall COVID-19 : 94.84% (objectif >90% atteint)
- Minimisation des faux négatifs COVID : 5.16% seulement
- F1-Score global : 96.52% (excellent équilibre)
- AUC-ROC micro-moyenne : 0.9970 (discrimination quasi-parfaite)

### Courbes d'entraînement

Les courbes montrent :
- Convergence progressive sans overfitting majeur
- Amélioration significative après le fine-tuning (Phase 2)
- Validation accuracy stable autour de 96-97%
- Réduction continue de la loss

### Matrice de confusion

```
Prédictions →        COVID    Normal    Viral Pneumonia
COVID                515      27        1              (543)
Normal               22       1490      17             (1529)
Viral Pneumonia      0        13        189            (202)
```

**Taux de vrais positifs :**
- COVID : 94.84%
- Normal : 97.45%
- Viral Pneumonia : 93.56%

---

## 6. Reproduction

### Environnement

**Python :** 3.10+

**Bibliothèques principales :**
```
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
pillow==10.0.0
kaggle==1.5.16
```

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/hadiatou4/covid19-xray-classification.git
cd covid19-xray-classification

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer Kaggle API
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Exécution

**Option 1 : Google Colab (Recommandé)**
1. Ouvrir le notebook `COVID19_Classification_Xception.ipynb` dans Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Exécuter toutes les cellules

**Option 2 : Local**
```bash
jupyter notebook
# Ouvrir COVID19_Classification_Xception.ipynb
# Exécuter les cellules séquentiellement
```

**Temps d'exécution :**
- Phase 1 (15 epochs) : ~30-40 minutes (GPU T4)
- Phase 2 (10 epochs) : ~20-30 minutes (GPU T4)
- Total : ~50-70 minutes

---

## 7. Auteurs

**Étudiant 1 :** Keita Hadiatou  
Email : haadikeita4@gmail.com  
Contribution : Développement du modèle, entraînement, évaluation

**Étudiant 2 :** Rouchdi Asmaa  
Email : asmaarouchdi72@gmail.com  
Contribution : Préparation des données, visualisations, analyse des résultats

**Date :** Novembre 2025  
**Module :** Intelligence Artificielle - Vision par Ordinateur

---

## 8. Licence

**Code :** MIT License

**Dataset :** Creative Commons Attribution 4.0 International (CC BY 4.0)  
Source : COVID-19 Radiography Database (Kaggle)

---

## Avertissement

Ce modèle est un outil d'aide à la décision à des fins éducatives et de recherche. Il ne doit pas remplacer l'expertise d'un professionnel de santé qualifié. Toute décision diagnostique finale doit être prise par un médecin. Une validation clinique approfondie est nécessaire avant tout déploiement en environnement réel.

---

## Structure du projet

```
covid19-xray-classification/
├── COVID19_Classification_Xception.ipynb   # Notebook principal
├── README.md                                # Ce fichier
├── requirements.txt                         # Dépendances
├── rapport_technique.pdf                    # Rapport détaillé
├── dataset/                                 # Dataset (non versionné)
├── results/                                 # Résultats (générés)
│   ├── xception_covid19_final.keras        # Modèle final
│   ├── results.json                         # Métriques
│   └── training_history.csv                # Historique
└── docs/                                    # Documentation
    ├── Fiche_de_cadrage.docx
    └── Fiche_de_justification_dataset.docx
```

---

## Références

1. Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. CVPR 2017.
2. Tawsifur Rahman et al. (2021). COVID-19 Radiography Database. Kaggle.
3. Keras Documentation : https://keras.io/api/applications/xception/
4. TensorFlow Transfer Learning Guide : https://www.tensorflow.org/tutorials/images/transfer_learning
