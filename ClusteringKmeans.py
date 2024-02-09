from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

coef_path="/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/coefficients_classification/tore /coefficientsVecteursLuminositéTore.npy"
coefficients = np.load(coef_path)

mask_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/masques/TORE/masquetore540.npy"
mask = np.load(mask_path)

def apply_kmeans_to_3d_clouds(coefficients, mask, n_clusters):
    mask = mask.astype(bool)
    print(coefficients.shape)
    h, w, n_points, _ = coefficients.shape
    
    # Préparer les données pour le clustering
    data_for_clustering = []
    compteurchat=0
    for i in range(h):
        for j in range(w):
            
            
            if mask[i, j]:
                # Ajouter le nuage de points 3D pour ce pixel
                data_for_clustering.append(coefficients[i, j].reshape(-1))


    # Convertir en array numpy
    data_for_clustering = np.array(data_for_clustering)

    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_for_clustering)

    # Obtenir les étiquettes de cluster
    labels = kmeans.labels_

    # Créer une carte de clustering
    cluster_map = np.full((h, w), -1)  # -1 pour les pixels du fond
    label_idx = 0
    print(compteurchat)
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                cluster_map[i, j] = labels[label_idx]
                label_idx += 1

    return cluster_map

# Appliquer la fonction modifiée
n_clusters = 2 # ou tout autre nombre souhaité
cluster_map = apply_kmeans_to_3d_clouds(coefficients, mask, n_clusters)

# Affichage
plt.imshow(cluster_map, cmap='jet')
plt.title('Résultat du Clustering')
plt.axis('off')
plt.show()
