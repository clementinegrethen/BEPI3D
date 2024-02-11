# Projet Partie BRDF - Modèle d'Oren-Nayar

## Auteur: @gos-alexis

## Organisation des dossiers 
Voici la partie du BE concernant l'implémentation du modèle d'Oren-Nayar pour modéliser la luminance de matériaux.
### Data
Le dossier data contient les données utilisées lors du BE:
- jpeg-exports: images de la pierre
- Sphere-RGB: données virutelles d'une sphère sous 70 éclairages et 3 matériaux
- Torus_3Multi: données virutelles d'un tore sous 70 éclairages avec différents albédos et 3 matériaux
- Toros_RGB_71lims:données virutelles d'un tore sous 70 éclairages avec 3 matériaux en RGB
- Toros_RRR:données virutelles d'un tore sous 70 éclairages avec 3 matériaux rouges



### Utils
- coefficients_classification: contient les coefficient générés pour chaque méthode et à classifier
    - HSH pour la classification basée sur les harmoniques sphériques
    - RTI pour la clafficiation basée sur l'interpolation polynomiale (coef de RTI)
    - VecteursLuminosité pour la classification basée sur les tranches de BRDF

- Lumières: fichiers txt des coordonnées de la lumière pour chaque éclairage
- masques: contient des masques pour chaque objet
- normales: contient les normales associées à chaque objet


## 


## Autre
$ python CreateMask.py $  permet de générer des masques npy pour un nouvel objet de la taille souhaitée (en adaptant le threshold).

$ python ACP.py $ permet d'appliquer une ACP sur les coefficients pour visualiser les n_components composantes principales et d'appliquer un kmeans.


