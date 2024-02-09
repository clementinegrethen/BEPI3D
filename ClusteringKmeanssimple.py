from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


coef_path="/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/coefficients_classification/tore /coefficientsRTI.npy"
coefficients = np.load(coef_path)

mask_path ="/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/masques/TORE/masquetore540.npy"
mask = np.load(mask_path)

def apply_kmeans_to_coefficients(coefficients, mask, n_clusters):
    mask = mask.astype(bool)

    # Remodeler la matrice des coefficients pour K-means
    reshaped_coeffs = coefficients.reshape(-1, coefficients.shape[2])
    print(reshaped_coeffs.shape)
    # Utiliser le masque pour exclure les pixels noirs (fond)
    reshaped_mask = mask.flatten()
    print(reshaped_mask.shape)
    coeffs_for_clustering = reshaped_coeffs[reshaped_mask]
    
    print("Nombre de pixels blancs dans le masque:", np.sum(reshaped_mask))
    print("Nombre de pixels inclus dans le clustering:", coeffs_for_clustering.shape[0])

   
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coeffs_for_clustering)

    # Obtenir les étiquettes de cluster
    labels = kmeans.labels_

    # Créer une image de clustering de taille originale
    cluster_map = np.full(reshaped_mask.shape, -1)  # -1 pour les pixels du fond
    cluster_map[reshaped_mask] = labels

    # Remodeler les étiquettes pour qu'elles correspondent aux dimensions de l'image
    cluster_map = cluster_map.reshape(coefficients.shape[0], coefficients.shape[1])

    return cluster_map

# Nombre de clusters
n_clusters = 2
# Appliquer K-means sur les coefficients
cluster_map = apply_kmeans_to_coefficients(coefficients, mask, n_clusters)

# Remplacer les valeurs -1 par 0 pour que le fond soit noir
cluster_map[cluster_map == -1] = n_clusters + 1  # On fait ceic pour ne pas avoir la meme couleur en fond que certaines cellules

# Affichage:
plt.imshow(cluster_map, cmap='jet')
plt.title('Résultat du Clustering')
plt.axis('off')
plt.show()