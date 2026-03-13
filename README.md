# PROJET___-2__-GROUPE__7__CODING__WEEK-1

## 1. Le dataset était-il équilibré ?

L'analyse de la variable cible **NObeyesdad** montre que le dataset contient **7 classes représentant différents niveaux d'obésité**.

L'étude de la distribution des classes montre que chaque catégorie représente environ **12 % à 15 % des observations**. Le dataset est donc **relativement équilibré**.

### Stratégie adoptée

Étant donné cet équilibre, aucune technique de rééchantillonnage comme :

- Oversampling
- Undersampling

n'a été appliquée.

Cependant, pour garantir un apprentissage équitable entre les classes, l'utilisation de **class weights** peut être envisagée lors de l'entraînement des modèles.

### Impact

Comme le dataset est déjà équilibré, les modèles de machine learning peuvent apprendre les différentes classes correctement sans biais important vers une classe dominante. Cela contribue à de bonnes performances globales des modèles.

---

## 2. Quel modèle de Machine Learning a obtenu les meilleures performances ?

Plusieurs modèles de machine learning ont été entraînés et comparés afin de prédire le niveau d'obésité :

- Random Forest
- XGBoost
- LightGBM

Les performances ont été évaluées à l'aide des métriques suivantes :

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

Le modèle **LightGBM** présente le **ROC-AUC le plus élevé (0.9991)**, indiquant une excellente capacité de discrimination entre les différentes classes d'obésité.

Ces résultats montrent que les **modèles de gradient boosting** sont particulièrement efficaces pour ce problème de classification.

---

## 3. Quelles variables influencent le plus les prédictions ? (SHAP)

Afin d'interpréter le modèle et comprendre l'importance des variables, la méthode **SHAP (SHapley Additive exPlanations)** a été utilisée.

SHAP permet d'identifier quelles variables contribuent le plus aux prédictions du modèle.

### Variables les plus influentes

Les variables ayant le plus d'influence sur les prédictions sont :

- **Weight (Poids)**
- **Height (Taille)**
- **Age**
- **FAF (Fréquence d'activité physique)**
- **CH2O (Consommation d'eau)**
- **FCVC (Consommation de légumes)**

### Interprétation

Les résultats montrent que :

- **le poids (Weight)** est le facteur le plus déterminant pour prédire le niveau d'obésité
- **la taille (Height)** et **l'âge (Age)** influencent également les prédictions
- certaines habitudes de vie comme **l'activité physique (FAF)** et **la consommation d'eau (CH2O)** jouent un rôle important

Ces résultats confirment que **les facteurs physiologiques et les habitudes de vie sont fortement liés au niveau d'obésité**.

---

## 4. Quels insights le Prompt Engineering a-t-il apporté ?

### Contexte

Le **prompt engineering** a été appliqué dans ce projet pour améliorer la qualité des analyses générées par des modèles de langage (Copilot / ChatGPT), notamment lors de la phase de **prétraitement des données** et d'**évaluation des modèles**.

---

### Tâche sélectionnée : Évaluation des modèles de Machine Learning

La tâche choisie consiste à demander à un LLM d'interpréter et de commenter les résultats des métriques d'évaluation (Accuracy, F1-score, ROC-AUC) pour les trois modèles entraînés.

---

### Prompts utilisés et résultats obtenus

#### Prompt 1 — Version basique (peu efficace)

**Prompt :**
```
Donne-moi une analyse des résultats de mon modèle.
```

**Résultat obtenu :**
Le modèle a fourni une réponse générique et vague, sans référence aux métriques spécifiques ni aux classes du dataset. L'analyse manquait de profondeur et n'apportait pas de valeur ajoutée pour le projet.

**Problème identifié :** Le prompt est trop ouvert. Il ne fournit ni contexte, ni données, ni objectif précis.

---

#### Prompt 2 — Version améliorée avec contexte (efficace)

**Prompt :**
```
Voici les résultats d'évaluation de trois modèles de classification entraînés sur un dataset de prédiction du niveau d'obésité (7 classes) :

| Modèle        | Accuracy | F1-score | ROC-AUC |
|---------------|----------|----------|---------|
| Random Forest | 0.9641   | 0.9643   | 0.9970  |
| XGBoost       | 0.9737   | 0.9738   | 0.9986  |
| LightGBM      | 0.9737   | 0.9735   | 0.9991  |

Analyse ces résultats en comparant les modèles, en identifiant le meilleur modèle et en expliquant pourquoi les modèles de gradient boosting performent mieux dans ce contexte.
```

**Résultat obtenu :**
Le modèle a produit une analyse structurée et pertinente, comparant les trois modèles, expliquant la supériorité du gradient boosting (gestion des interactions non-linéaires, robustesse au surapprentissage), et soulignant que LightGBM se distingue par son ROC-AUC légèrement supérieur.

**Ce qui a fonctionné :** L'ajout des données réelles et d'une consigne précise a drastiquement amélioré la qualité de la réponse.

---

#### Prompt 3 — Version avancée avec rôle et format attendu (très efficace)

**Prompt :**
```
Tu es un data scientist expert en machine learning. Voici les résultats d'évaluation de trois modèles
entraînés sur un dataset de prédiction du niveau d'obésité (7 classes équilibrées) :

| Modèle        | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------|----------|-----------|--------|----------|---------|
| Random Forest | 0.9641   | 0.9658    | 0.9641 | 0.9643   | 0.9970  |
| XGBoost       | 0.9737   | 0.9740    | 0.9737 | 0.9738   | 0.9986  |
| LightGBM      | 0.9737   | 0.9735    | 0.9737 | 0.9735   | 0.9991  |

Fournis :
1. Une comparaison claire des trois modèles
2. Une justification du choix du meilleur modèle
3. Des recommandations pour améliorer encore les performances
Utilise un langage technique mais accessible, et structure ta réponse avec des titres.
```

**Résultat obtenu :**
La réponse était très complète : comparaison détaillée, justification argumentée du choix de LightGBM, et recommandations concrètes (hyperparameter tuning, cross-validation, feature engineering). Le format structuré avec titres rendait la réponse directement réutilisable dans la documentation du projet.

---

### Tableau récapitulatif des prompts

| Version | Qualité du prompt | Qualité de la réponse | Utilisabilité |
|---------|-------------------|-----------------------|---------------|
| Prompt 1 (basique) | Faible | Générique et vague | Faible |
| Prompt 2 (avec contexte) | Moyenne | Pertinente et structurée | Bonne |
| Prompt 3 (avec rôle + format) | Élevée | Complète et réutilisable | Excellente |

---

### Efficacité des prompts — Discussion

Les expériences montrent que :

- Les **prompts vagues** produisent des réponses génériques sans valeur ajoutée pour le projet
- L'ajout de **données concrètes** (métriques, tableau de résultats) améliore considérablement la pertinence des réponses
- Attribuer un **rôle explicite** au modèle ("Tu es un data scientist expert...") oriente le niveau de langage et la profondeur de l'analyse
- Spécifier le **format attendu** (liste numérotée, titres, tableau) rend la réponse directement exploitable dans la documentation

### Améliorations potentielles

Pour aller encore plus loin, les améliorations suivantes pourraient être envisagées :

- **Ajouter des exemples de réponses attendues** (few-shot prompting) pour guider encore mieux le modèle
- **Décomposer les tâches complexes** en plusieurs prompts successifs (chain-of-thought) plutôt qu'un seul prompt long
- **Itérer sur les prompts** en fonction des retours obtenus pour les affiner progressivement
- **Tester différents LLMs** (GPT-4, Copilot, Claude) et comparer la qualité des réponses pour la même tâche

### Conclusion

Le prompt engineering améliore significativement la qualité des analyses générées par les modèles de langage. Un prompt bien conçu — avec contexte, rôle, données et format attendu — permet d'obtenir des résultats directement utilisables dans la documentation du projet, réduisant ainsi le temps de rédaction et améliorant la précision des analyses.