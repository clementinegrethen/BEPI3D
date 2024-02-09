# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os
# import scipy.io as sio
# from scipy.ndimage import gaussian_filter

# def calculate_rotation_matrix(normal):
#     # Axe Z par défaut
#     axis_z = np.array([0, 0, 1])
#     # Calculer l'axe de rotation (produit vectoriel) et l'angle de rotation
#     rotation_axis = np.cross(axis_z, normal)
#     rotation_angle = np.arccos(np.dot(axis_z, normal) / (np.linalg.norm(axis_z) * np.linalg.norm(normal)))
#     # Normaliser l'axe de rotation
#     rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
#     # Utiliser la formule de Rodrigues pour calculer la matrice de rotation
#     K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
#                   [rotation_axis[2], 0, -rotation_axis[0]],
#                   [-rotation_axis[1], rotation_axis[0], 0]])
#     rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
#     return rotation_matrix

# def cart_to_spherical(x, y, z):
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z / r)
#     phi = np.arctan2(y, x)
#     return r, theta, phi

# def load_images_from_folder(folder):
#     images = []
#     for filename in sorted(os.listdir(folder)):
#         if filename.endswith(".JPG") or filename.endswith(".png"):
#             img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
#             #Resize image in 810*810
#             #img = cv2.resize(img, (810, 810))


#             if img is not None:
#                 images.append(img)
#     return images

# def read_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     light_coords = []
#     for line in lines:
#         parts = line.split()
#         light_coords.append((float(parts[1]), -float(parts[2]), -float(parts[3])))
#         #Changer la list en np.array
#     light_coords = np.array(light_coords)
#     return light_coords


# def sphere(images_folder, light_coords):
#     # on charge les images et on convdertit les coordonnées cartésiennes en coordonnées sphériques
#     images = load_images_from_folder(images_folder)
  
#     nuage_points = np.zeros((images[0].shape[0], images[0].shape[1], 71,3)) # 6 coefficients pour le modèle polynomial
#     for i in range(images[0].shape[0]):
#         for j in range(images[0].shape[1]):
#             # intensities contient les intensités des pixels (i, j) de toutes les images
#             gray_levels= np.array([img[i, j] for img in images])
#             vector_endpoints = light_coords[i,j] * gray_levels[:, np.newaxis]

#             nuage_points[i, j] = vector_endpoints
#     return nuage_points

# def compute_derivatives(normals):
#     # Calcul des dérivées premières (gradient) en utilisant un filtre gaussien
#     dx = gaussian_filter(normals, sigma=1, order=[1, 0, 0])
#     dy = gaussian_filter(normals, sigma=1, order=[0, 1, 0])

#     # Calcul des dérivées secondes
#     dxx = gaussian_filter(normals, sigma=1, order=[2, 0, 0])
#     dyy = gaussian_filter(normals, sigma=1, order=[0, 2, 0])
#     dxy = gaussian_filter(normals, sigma=1, order=[1, 1, 0])

#     return dx, dy, dxx, dyy, dxy

# def compute_principal_curvatures(normals):
#     dx, dy, dxx, dyy, dxy = compute_derivatives(normals)
#     H = (dxx + dyy) / 2   # Courbure moyenne
#     K = dxx * dyy - dxy ** 2  # Courbure gaussienne

#     # Calcul des courbures principales
#     discriminant = np.sqrt(H ** 2 - K)
#     k1 = H + discriminant
#     k2 = H - discriminant

#     return k1, k2

# image_path = "/Users/leomeissner/Desktop/pi3d/BEPI3D-main/Generation_blender/Torus_RGB_71ims" 
# images = load_images_from_folder(image_path)
# #normals = sio.loadmat('torus/N.mat')
# normal_path = "/Users/leomeissner/Desktop/Normal_1.png"
# normals = cv2.imread(normal_path, cv2.IMREAD_COLOR)
# normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)
# normals = normals / (255/2) - 1
# normals = normals.astype(np.float32)

# #normals = normals['N']

# #Verifier que les normales sont bien normalisées
# for i in range(normals.shape[0]):
#     for j in range(normals.shape[1]):
#         if np.linalg.norm(normals[i, j]) != 1:
#             print("Erreur")
#             normals[i, j] = normals[i, j] / np.linalg.norm(normals[i, j])


# image_path = "/Users/leomeissner/Desktop/pi3d/BEPI3D-main/Generation_blender/Torus_RGB_71ims" 
# images = load_images_from_folder(image_path)
# file_path = '/Users/leomeissner/Desktop/pi3d/BEPI3D-main/Generation_blender/lumiere_torusRGB_71ims.txt'
# light_coords = read_file(file_path)

# new_light_coords = np.zeros((images[0].shape[0], images[0].shape[1], light_coords.shape[0], 3))
# for i in range(images[0].shape[0]):
#     for j in range(images[0].shape[1]):
#         new_light_coords[i, j] = light_coords

# coord_sphere=sphere(image_path, new_light_coords)

# #Pour chaque pixel on calcul les coord des lights
# new_light_coords = np.zeros((images[0].shape[0], images[0].shape[1], light_coords.shape[0], 3))
# for i in range(images[0].shape[0]):
#     for j in range(images[0].shape[1]):
            
#         z_normal = normals[i, j]
#         vectors = coord_sphere[i,j]
#         #Prendre le vecteur avec la norme max
#         max_norm = 0
#         for k in range(vectors.shape[0]):
#             if np.linalg.norm(vectors[k]) > max_norm:
#                 max_norm = np.linalg.norm(vectors[k])
#                 m = vectors[k]

#         if max_norm == 0:
#             m = np.array([0, 0, 0])

#         x_normal = np.cross(z_normal, m)
#         y_normal = np.cross(z_normal, x_normal)

#         #Verifier que c'est une base direct
#         if np.dot(z_normal, np.cross(x_normal, y_normal)) < 0:
#             y_normal = -y_normal

#         transformation_matrix = np.column_stack((x_normal, y_normal, z_normal))
#         #transformation_matrix = calculate_rotation_matrix(normals[i, j])

#         for k in range(light_coords.shape[0]):
#             new_light_coords[i, j, k] = np.matmul(transformation_matrix, light_coords[k])


# coord_sphere=sphere(image_path, new_light_coords)
# coef_path='coefToreSphere.npy'
# np.save(coef_path, coord_sphere)







import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import scipy.io

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

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Redimensionnement de l'image
                resized_img = cv2.resize(img, (612, 408))
                images.append(img)
    return images

def read_file(file_path):
    light_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            light_coords.append([float(parts[1]), -float(parts[2]), -float(parts[3])])
    return np.array(light_coords)

def normalize_normals(normals):
    return normals / np.linalg.norm(normals, axis=2, keepdims=True)

image_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/Data/Torus_RGB_71ims"
normal_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/normales/Normal_1.png"
light_file_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/lumières /lumiere_torusRGB_71ims.txt"
images = load_images_from_folder(image_path)

normals_image = cv2.imread(normal_path, cv2.IMREAD_COLOR)
# print(normals_image)
normals_image = cv2.cvtColor(normals_image, cv2.COLOR_BGR2RGB)
def normalize_normals(normals):
    # Calcule la norme de chaque vecteur
    norms = np.linalg.norm(normals, axis=2)
    # Évite la division par zéro
    norms[norms == 0] = 1
    # Normalise chaque vecteur
    return normals / np.expand_dims(norms, axis=2)

# Chargez le fichier .mat
#normals = scipy.io.loadmat("/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/normales/N.mat")['N']

# Remplacez les NaN par zéro
#normals = np.nan_to_num(normals)




# Appliquez la transformation
normals = normals_image / 255.0 * 2 - 1

# Normalisez les vecteurs normaux
#normals = normalize_normals(normals)




light_coords = read_file(light_file_path)
new_light_coords = np.zeros((normals.shape[0], normals.shape[1], light_coords.shape[0], 3))

for i in range(normals.shape[0]):
    for j in range(normals.shape[1]):
        normal = normals[i, j]
        rotation_matrix = calculate_rotation_matrix(normal)
        for k in range(light_coords.shape[0]):
            new_light_coords[i, j, k] = rotation_matrix @ light_coords[k]


def sphere(images_folder, light_coords):
    # on charge les images et on convdertit les coordonnées cartésiennes en coordonnées sphériques
    images = load_images_from_folder(images_folder)
  
    nuage_points = np.zeros((images[0].shape[0], images[0].shape[1], 71,3)) # 6 coefficients pour le modèle polynomial
    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            # intensities contient les intensités des pixels (i, j) de toutes les images
            gray_levels= np.array([img[i, j] for img in images])
            vector_endpoints = light_coords[i,j] * gray_levels[:, np.newaxis]

            nuage_points[i, j] = vector_endpoints
    return nuage_points


coord_sphere=sphere(image_path, new_light_coords)
coef_path='coefficientsVecteursLuminositéTore.npy'
np.save(coef_path, coord_sphere)