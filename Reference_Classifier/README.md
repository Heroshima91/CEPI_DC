# Reference Classifier

Reference Classifier permet de séparer une banque de données d'images suivant les différents modèles qui existent. 

## Installation

Python version >= 3.7.4
```bash
  pip install opencv-python==3.4.5
  pip install opencv-contrib-python==3.4.5
```

## Usage

Créer un dossier nommé Template à la racine, et insérer tous les différents types de modèles et taper la commande

```python
python classifier.py
```

## Précisions

La ligne 123 peut être remplacer par d'autres détecteurs (SIFT, SURF)

```python
    detector = cv2.AKAZE_create(threshold = 0.01)
```

La ligne 88 contient une vérification de si deux images ont 45% de taux de ressemblance, ce taux a été déterminé de manière empirique et peut être modifié. 

```python
    if(max(ref_sim)>=0.45):
```

## Liens vers les tutoriels

Akaze : https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html
Sift/Surf : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
