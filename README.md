# Projet Partie BRDF - Modèle d'Oren-Nayar

## Auteur: @gos-alexis

## Organisation des dossiers 
Voici la partie du BE concernant l'implémentation du modèle d'Oren-Nayar pour modéliser la luminance de matériaux.
### Data
Le RTI contient les données utilisées lors du BE:
- jpeg-exports : Images de la pierre
- assembly-files/lumiere.txt : Directions des lumières

### Utils
#### Fonctions et .mat
- mask.mat : contient un masque pour les images de la pierre (On pourrait prendre plusieurs masques pour plusieurs matériaux en modifiant légèrement le code pour itérer sur chaque masque)
- Calcul_mk1.m : Permet de calcul mk+1 (cf présentation dans la branche main)
- Xon.m : Calcul de Xon (cf présentation dans la branche main)

#### Code à lancer
- OrenNayar_model.m : Code main à lancer pour calculer le modèle d'Oren-Nayar. Il faut adapter les path des images/mask/lumiere. Structure du code :
                      1 - 89 : Initialisation des paramètres et import des données
                      91 - 94 : Modèle Lambertien
                      110 - 132 : Boucle principale sur sigma et nb_iterations
                      134 - 147 : Nuance de gris --> RGB
                      149 - 152 : Images vectorisées --> Taille initiale
                      154 - 166 : Enregistrement des données affichables dans un .mat
- ON_visualization.m : Visualisation de courbes et de résultats du modèle

## Références
https://oatao.univ-toulouse.fr/14684/1/YQUEAU.pdf (page 208-212)

