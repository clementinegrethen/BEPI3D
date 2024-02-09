from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chemin vers le fichier des coefficients
coef_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/coefficients_classification/tore /coefficientsVecteursLuminositéTore.npy"

# Charger les données
coefficients = np.load(coef_path)

# Appliquer l'ACP sur les données pour réduire les dimensions
pca = PCA(n_components=2)  # Vous pouvez ajuster le nombre de composantes principales selon vos besoins
coefficients_reshaped = coefficients.reshape(-1, coefficients.shape[3])
pca_result = pca.fit_transform(coefficients_reshaped)

# Appliquer K-means sur les composantes principales
n_clusters = 3  # Ou tout autre nombre de clusters souhaité
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(pca_result)

# Visualiser les clusters
plt.figure(figsize=(8, 6))
for cluster_label in range(n_clusters):
    plt.scatter(pca_result[cluster_labels == cluster_label, 0], pca_result[cluster_labels == cluster_label, 1],
                label=f'Cluster {cluster_label}')

plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('Classification après ACP et K-means')
plt.legend()
plt.grid(True)
plt.show()
