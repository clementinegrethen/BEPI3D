import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_mask(image_path):
    # Lecture de l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(image)
    scale_factor=0.5
    width = 612#int(masked_img.shape[1] * scale_factor)
    height = 408#int(masked_img.shape[0] * scale_factor)
    dim = (width, height)
    #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # Seuillage pour isoler la pierre
    _, thresholded = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.erode(thresholded, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    mask = (thresholded // 255).astype(np.uint8)

    return image, mask

image_path = '/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/masques/sphere/Masque.png'

# Créer le masque
original_image, mask = create_mask(image_path)
mask_path='utils/masquetorek540.npy'
np.save(mask_path, mask)

# Affichage pour vérifier:
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Masque')
plt.axis('off')

plt.show()




