import cv2
import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    image_paths = []
    light_coords = []
    for line in lines:
        parts = line.split()
        image_paths.append(parts[0])
        light_coords.append((float(parts[1]), -float(parts[2]), -float(parts[3])))
    return image_paths, light_coords

# Charger les images
def load_images_from_folder(folder):
    images = []
    mask_path="/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/masques/TORE/masquetore540.npy"
    mask = np.load(mask_path)
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # img = cv2.resize(img, (612, 408))

                masked_img = cv2.bitwise_and(img, img, mask=mask)
                scale_factor=0.5
                width = int(masked_img.shape[1] * scale_factor)
                height = int(masked_img.shape[0] * scale_factor)
                dim = (width, height)
                resized_img = cv2.resize(img, (612, 408))
                images.append(masked_img)
    return images

def cart_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # colatitude
    phi = np.arctan2(y, x)    # azimuth
    return theta, phi

def polynomial_model(x, a00, a10, a01, a20, a11, a02):
    theta, phi = x
    return a00 + a10*theta + a01*phi + a20*theta**2 + a11*theta*phi + a02*phi**2

def fit_rti(images_folder, light_coords):
    # on charge les images et on convdertit les coordonnées cartésiennes en coordonnées sphériques
    images = load_images_from_folder(images_folder)
    print(images)
    light_directions = np.array([cart_to_spherical(*coord) for coord in light_coords]).T
    # on calcule les coefficients du modèle polynomial pour chaque pixel : matrice de taille (h, w, 6) avec h
  

    coefficients = np.zeros((images[0].shape[0], images[0].shape[1], 6)) # 6 coefficients pour le modèle polynomial
    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            # intensities contient les intensités des pixels (i, j) de toutes les images
            intensities = np.array([img[i, j] for img in images])
            params, _ = curve_fit(polynomial_model, light_directions, intensities)
            coefficients[i, j] = params
    return coefficients



file_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/lumières /lumiere_torusRGB_71ims.txt"
image_paths, light_coords = read_file(file_path)
images_folder = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/Data/Torus_RGB_71ims"
# Calcul des coefficients RTI
coefficients = fit_rti(images_folder, light_coords)
coef_path='coefficientsRTI.npy'
np.save(coef_path, coefficients)
