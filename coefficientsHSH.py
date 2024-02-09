import numpy as np
import cv2
import os
from scipy.special import sph_harm
from scipy.optimize import least_squares
import scipy.io
# Fonction pour lire les données du fichier
def read_file(file_path):
    light_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            light_coords.append([float(parts[1]), -float(parts[2]), -float(parts[3])])
    return np.array(light_coords)

# Convertir les coordonnées cartésiennes en coordonnées sphériques
def cart_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return theta, phi

def calculate_spherical_harmonics(light_directions, max_degree):
    theta, phi = cart_to_spherical(*np.array(light_directions).T)
    harmonics = []
    for degree in range(max_degree + 1):
        for order in range(-degree, degree + 1):
            harmonic = sph_harm(order, degree, phi, theta)
            harmonics.append(harmonic.real)  
    return np.array(harmonics)

def calculate_rotation_matrix(normal):
    axis_z = np.array([0, 0, 1])
    rotation_axis = np.cross(axis_z, normal)
    rotation_angle = np.arccos(np.dot(axis_z, normal) / (np.linalg.norm(axis_z) * np.linalg.norm(normal)))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
    return rotation_matrix
def rotate_light_directions(normals, light_coords):
    rotated_light_coords = np.zeros_like(light_coords)
    for i in range(normals.shape[0]):
        for j in range(normals.shape[1]):
            normal = normals[i, j]
            rotation_matrix = calculate_rotation_matrix(normal)
            for k in range(light_coords.shape[0]):
                rotated_light_coords[k] = rotation_matrix @ light_coords[k]
    return rotated_light_coords


# Fonction pour ajuster le modèle avec les harmoniques sphériques
def fit_model_with_spherical_harmonics(images, light_coords, max_degree, normals):
    total_harmonics = (max_degree + 1) ** 2
    coefficients = np.zeros((images[0].shape[0], images[0].shape[1], total_harmonics))

    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            normal = normals[i, j]
            # Rotation des directions de la lumière pour ce pixel
            rotated_light_coords = [calculate_rotation_matrix(normal) @ light_coord for light_coord in light_coords]
            intensities = np.array([img[i, j] for img in images])
            # Calcul des harmoniques sphériques pour les directions de lumière tournées
            harmonics = calculate_spherical_harmonics(rotated_light_coords, max_degree)

            # Ajustement du modèle HSH
            func = lambda params: np.dot(harmonics.T, params) - intensities
            res = least_squares(func, np.zeros(total_harmonics))
            coefficients[i, j] = res.x

    return coefficients



# Charger les images
def load_images_from_folder(folder):
    images = []
    mask_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/masques/TORE/masquetore540.npy"
    mask = np.load(mask_path)
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                #img = cv2.resize(img, (612, 408))

                masked_img = cv2.bitwise_and(img, img, mask=mask)
                scale_factor=0.5
                width = int(masked_img.shape[1] * scale_factor)
                height = int(masked_img.shape[0] * scale_factor)
                dim = (width, height)
                resized_img = cv2.resize(masked_img, (612, 408))
                images.append(masked_img)
    return images


# Exemple d'utilisation
file_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/lumières /lumiere_torusRGB_71ims.txt"
light_coords = read_file(file_path)
images_folder = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/Data/Torus_3Multi"
images = load_images_from_folder(images_folder)
normal_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/normales/Normal_1.png"
#normals = scipy.io.loadmat("/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/normales/N.mat")['N']
#normals = np.nan_to_num(normals)
def normalize_normals(normals):
    # Calcule la norme de chaque vecteur
    norms = np.linalg.norm(normals, axis=2)
    # Évite la division par zéro
    norms[norms == 0] = 1
    # Normalise chaque vecteur
    return normals / np.expand_dims(norms, axis=2)
normals = cv2.imread(normal_path, cv2.IMREAD_COLOR)
print(normals)
normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)
normals = normals / 255.0 * 2 - 1
#normals = normalize_normals(normals)

max_degree = 2

# Calcul des coefficients en utilisant les harmoniques sphériques
coefficients = fit_model_with_spherical_harmonics(images, light_coords, max_degree,normals)
np.save("coefficientsHSHmulti.npy", coefficients)  # Sauvegarder les coefficients
