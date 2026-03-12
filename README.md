# PROJET___-2__-GROUPE__7__CODING__WEEK-1

## 1. Le dataset était-il équilibré ?

L’analyse de la variable cible **NObeyesdad** montre que le dataset contient **7 classes représentant différents niveaux d’obésité**.

L’étude de la distribution des classes montre que chaque catégorie représente environ **12 % à 15 % des observations**. Le dataset est donc **relativement équilibré**.

### Stratégie adoptée

Étant donné cet équilibre, aucune technique de rééchantillonnage comme :

- Oversampling
- Undersampling

n’a été appliquée.

Cependant, pour garantir un apprentissage équitable entre les classes, l’utilisation de **class weights** peut être envisagée lors de l’entraînement des modèles.

### Impact

Comme le dataset est déjà équilibré, les modèles de machine learning peuvent apprendre les différentes classes correctement sans biais important vers une classe dominante. Cela contribue à de bonnes performances globales des modèles.

---

# 2. Quel modèle de Machine Learning a obtenu les meilleures performances ?

Plusieurs modèles de machine learning ont été entraînés et comparés afin de prédire le niveau d’obésité :

- Random Forest
- XGBoost
- LightGBM

Les performances ont été évaluées à l’aide des métriques suivantes :

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Résultats des modèles

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|------|------|------|------|------|
| Random Forest | 0.9641 | 0.9658 | 0.9641 | 0.9643 | 0.9970 |
| XGBoost | 0.9737 | 0.9740 | 0.9737 | 0.9738 | 0.9986 |
| LightGBM | 0.9737 | 0.9735 | 0.9737 | 0.9735 | **0.9991** |

### Conclusion

Les modèles **XGBoost** et **LightGBM** obtiennent les meilleures performances globales.

Le modèle **LightGBM** présente le **ROC-AUC le plus élevé (0.9991)**, indiquant une excellente capacité de discrimination entre les différentes classes d’obésité.

Ces résultats montrent que les **modèles de gradient boosting** sont particulièrement efficaces pour ce problème de classification.

---

# 3. Quelles variables influencent le plus les prédictions ? (SHAP)

Afin d'interpréter le modèle et comprendre l’importance des variables, la méthode **SHAP (SHapley Additive exPlanations)** a été utilisée.

SHAP permet d’identifier quelles variables contribuent le plus aux prédictions du modèle.

### Variables les plus influentes

Les variables ayant le plus d’influence sur les prédictions sont :

- **Weight (Poids)**
- **Height (Taille)**
- **Age**
- **FAF (Fréquence d'activité physique)**
- **CH2O (Consommation d'eau)**
- **FCVC (Consommation de légumes)**

### Interprétation

Les résultats montrent que :

- **le poids (Weight)** est le facteur le plus déterminant pour prédire le niveau d’obésité
- **la taille (Height)** et **l’âge (Age)** influencent également les prédictions
- certaines habitudes de vie comme **l’activité physique (FAF)** et **la consommation d’eau (CH2O)** jouent un rôle important

Ces résultats confirment que **les facteurs physiologiques et les habitudes de vie sont fortement liés au niveau d’obésité**.

---

# 4. Quels insights le Prompt Engineering a-t-il apporté ?

Le **prompt engineering** a été utilisé pour améliorer l'interaction avec les modèles de langage et faciliter l’analyse du dataset.

Différentes formulations de prompts ont été testées afin d’obtenir des réponses plus précises et mieux structurées.

### Observations

Les expériences montrent que :

- les **prompts détaillés et structurés** produisent des réponses plus pertinentes
- l’ajout de **contexte spécifique au dataset** améliore la qualité des analyses
- les prompts bien formulés permettent d’obtenir **des explications plus interprétables et utiles**

### Conclusion

Le prompt engineering améliore significativement la qualité des analyses générées par les modèles de langage et facilite l’interprétation des résultats du projet.