# Reference Classifier

Reference Classifier permet de séparer une banque de données d'images suivant les différents modèles qui existent. 

## Installation

Python version >= 3.7.4
```bash
  pip install opencv-python==3.4.5
  pip install opencv-contrib-python==3.4.5
```

## Usage

Créer un dossier nommé Dataset à la racine, et insérer toutes les images
Dans le script sequence.py, dans la fonction checkWindow(image, fn), mettez les coordonées des centres de chaque case cochée.
Rajouter le fichier csv contenant la valeur de chaque case à la racine du dossier et dans la fonction load_y(fn) mettez le bon nom de fichier (ligne 220). 
Lancer le script avec

```python
python sequence.py
```

Vous aurez deux fichier svm_X.txt et svm_y.txt.
Lancer le script svm.py (rapide) ou dtwknn.py.
```python
python svm.py
```

## Liens vers les tutoriels

[Tslearn](https://tslearn.readthedocs.io/en/latest/auto_examples/plot_neighbors.html)<br/>
[Sklearn](https://scikit-learn.org/stable/)

## Contact 

anandaramane.candassamy@gmail.com
