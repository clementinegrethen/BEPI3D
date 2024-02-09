from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib

coef_path="/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/coefficients_classification/tore /coefficientsRTI.npy"
coefficients = np.load(coef_path)

mask_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/masques/TORE/masquetore540.npy"

mask = np.load(mask_path)
mask = mask.astype(bool)

# Remodeler la matrice des coefficients pour K-means
reshaped_coeffs = coefficients.reshape(-1, coefficients.shape[2])
print(reshaped_coeffs.shape)
# Utiliser le masque pour exclure les pixels noirs (fond)
reshaped_mask = mask.flatten()
print(reshaped_mask.shape)
coeffs_for_clustering = reshaped_coeffs[reshaped_mask]




# # Initialisation des variables globales
# drawing = False  # True si le dessin est en cours
# mode = True  # Si True, dessinez un rectangle. Appuyez sur 'm' pour basculer en mode cercle
# ix, iy = -1, -1
# labels = []

# # Fonction de rappel pour les événements de souris
# def interactive_drawing(event, x, y, flags, param):
#     global ix, iy, drawing, mode

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
#             else:
#                 cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
#         else:
#             cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
#         # Demander le label pour la région sélectionnée
#         label = input("Entrez le label pour la région sélectionnée (1, 2, ou 3): ")
#         labels.append((label, (ix, iy, x, y)))

# img = cv2.imread('/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/Torus_RGB_71ims/0033.png')
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', interactive_drawing)

# while(1):
#     cv2.imshow('image', img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:  # Appuyez sur ESC pour quitter
#         break

# cv2.destroyAllWindows()

# # Afficher les labels et les régions
# for label, region in labels:
#     print(f"Label: {label}, Région: {region}")
    
# print("ok")
# # Séparation en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(coeffs_for_clustering, labels, test_size=0.2, random_state=42)




import cv2
import numpy as np
from sklearn.svm import SVC

# Fonction pour collecter les données de la région sélectionnée
def collect_data_from_region(image, coefficients, mask):
    ix, iy = -1, -1  # Coordonnées initiales du clic de souris
    drawing = False  # Indicateur de dessin
    selected_regions = []  # Liste pour stocker les régions sélectionnées
    all_labels = []  # Liste pour stocker tous les labels

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 1)
            label = int(input("Entrez le label pour la région sélectionnée (1, 2, ou 3): "))
            selected_regions.append((ix, iy, x, y, label))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    data = []
    for region in selected_regions:
        x1, y1, x2, y2, label = region
        for i in range(y1, y2):
            for j in range(x1, x2):
                if mask[i, j]:
                    data.append(coefficients[i, j])
                    all_labels.append(label)

    return np.array(data), np.array(all_labels)





image = cv2.imread('/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/Data/Torus_RGB_71ims/0005.png')
#image = cv2.resize(image, (612, 408))

data, labels = collect_data_from_region(image, coefficients, mask)


# Entraînement du SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(data,labels)

# Classification de l'ensemble des pixels
all_predictions = svm_model.predict(coeffs_for_clustering)

# Remodelage pour l'affichage
predicted_map = np.full(mask.flatten().shape, -1)
predicted_map[mask.flatten()] = all_predictions
predicted_map = predicted_map.reshape(mask.shape)

# Affichage
plt.imshow(predicted_map, cmap='jet')
plt.title('Résultat de la Classification SVM')
plt.axis('off')
plt.show()
