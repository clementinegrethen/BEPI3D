# Projet  Partie Classification de Matériaux

# Auteur: @clementinegrethen
# Auteur: @leomeissner00

## Organisation des dossiers 
Voici la partie du BE concernant la classification des matériaux constituant un objet
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


## Classification Interpolation Polynomiale

$ python coefficientRTI.py $ génère un fichier npy qui convient les coefficients associés à chaque pixel. 
Cette fonction nécessite d'adapter le file_path pour indiquer le chemin des lumières, le images_folder pour le chemin des images et mask_path pour lui donner un masque associé à l'objet.

Attention: pour nos données de synthèse la taille est de 540x540, mais concernant la pierre les données étant en 4K nous avons travaillé avec une taille 612x408 (les masques sont adaptés), il faut donc dé-commenter la ligne 30 pour la lecture des images.

### clustering Kmeans : non supervisée
$ python ClusteringKmeansspimple.py $ 
Cette fonction affiche une classification basée sur un kmeans avec n_cluster.
Veuillez à adapter le path des coefficients à classifier et le masque de l'objet.

### clustering SVM: supervisé:
$ python VSMClustering.py $ 

Cette fonction affiche une classification basée sur un svm.
Il faut adapter le coef_path, le mask_path et "image" (chemin vers une image de l'objet pour l'interface de supervision).
Pour superviser il faut selectionner une boundingbox dans chaque classe, indiquer le numéro de la classe sur le terminal et répéter autant de fois que nécessaire (l'interface n'est pas intuitive et fonctionne mal si elle ets mal utilisée). Une fois la supervision fait il faut appuyer sur "q" pour lancer la classification.


## Classification des Tranches de BRDF
Ici, nous examinons la classification des matériaux en utilisant les tranches de BRDF (Bidirectional Reflectance Distribution Function). 
$ python coefficientVecteurLuminosite.py $ génère un fichier npy qui convient les coefficients associés à chaque pixel. 
Cette fonction nécessite d'adapter le light_file_path pour indiquer le chemin des lumières, le images_folder pour le chemin des images ,normal_path pour lui donner les  normales associées à l'objet.

Nos normales sont des images png mais le code permet aussi de traiter des normales stockées dans un fichier .mat (il faut donc décommenter les lignes correspondantes).

Attention: pour nos données de synthèse la taille est de 540x540, mais concernant la pierre les données étant en 4K nous avons travaillé avec une taille 612x408 (les masques sont adaptés), il faut donc modifier la fonction load_images_from_folder pour (ne pas) inclure l'image redimensionnée.
### clustering Kmeans : non supervisée
$ python ClusteringKmeans.py $ 
Cette fonction affiche une classification basée sur un kmeans avec n_cluster.
Veuillez à adapter le path des coefficients à classifier et le masque de l'objet.
(Elle diffère de ClusteringKmeansSimple.py car elle permet de classifier des coefficients de cette dimension précisement.)

## Classification Interpolation par Harmonique Sphérique
Dans cette partie, nous explorons la classification des matériaux en utilisant l'interpolation par harmonique sphérique. Cette méthode repose sur la représentation des fonctions de réflectance des matériaux en termes de coefficients d'harmoniques sphériques.

$ python coefficientsHSH.py $ génère un fichier npy qui convient les coefficients associés à chaque pixel. 
Cette fonction nécessite d'adapter le file_path pour indiquer le chemin des lumières, le images_folder pour le chemin des images ,normal_path pour lui donner les  normales associées à l'objet et mask_path pour le masque.

Nos normales sont des images png mais le code permet aussi de traiter des normales stockées dans un fichier .mat (il faut donc décommenter les lignes correspondantes).

Attention: pour nos données de synthèse la taille est de 540x540, mais concernant la pierre les données étant en 4K nous avons travaillé avec une taille 612x408 (les masques sont adaptés), il faut donc modifier la fonction load_images_from_folder pour (ne pas) inclure l'image redimensionnée.

### clustering Kmeans : non supervisée
$ python ClusteringKmeansspimple.py $ 
Cette fonction affiche une classification basée sur un kmeans avec n_cluster.
Veuillez à adapter le path des coefficients à classifier et le masque de l'objet.

### clustering SVM: supervisé:
$ python VSMClustering.py $ 

Cette fonction affiche une classification basée sur un svm.
Il faut adapter le coef_path, le mask_path et "image" (chemin vers une image de l'objet pour l'interface de supervision).
Pour superviser il faut selectionner une boundingbox dans chaque classe, indiquer le numéro de la classe sur le terminal et répéter autant de fois que nécessaire (l'interface n'est pas intuitive et fonctionne mal si elle est mal utilisée). Une fois la supervision réalisée il faut appuyer sur "q" pour lancer la classification.


## Autre
$ python CreateMask.py $  permet de générer des masques npy pour un nouvel objet de la taille souhaitée (en adaptant le threshold).

$ python ACP.py $ permet d'appliquer une ACP sur les coefficients pour visualiser les n_components composantes principales et d'appliquer un kmeans.


