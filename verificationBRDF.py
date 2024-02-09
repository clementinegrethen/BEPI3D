import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.spatial import ConvexHull, distance
from itertools import combinations

def cart_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    light_coords = []
    for line in lines:
        parts = line.split()
        light_coords.append((float(parts[1]), -float(parts[2]), -float(parts[3])))
    return light_coords

def plot_light_directions(images, light_coords, pixel_coords, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Remove grid lines
    ax.grid(False)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Vérifiez d'abord si les coordonnées du pixel sont valides pour toutes les images
    if any(pixel_coords[0] >= img.shape[0] or pixel_coords[1] >= img.shape[1] for img in images):
        print(f"Pixel coordinates for {title} out of bounds for one or more images")
        return

    # Créer un tableau NumPy pour les niveaux de gris du pixel pour toutes les images
    gray_levels = np.array([img[pixel_coords[0], pixel_coords[1]] for img in images])
    max_gray = np.max(np.max(gray_levels))
    gray_levels = gray_levels / max_gray

    vector_endpoints = light_coords * gray_levels[:, np.newaxis]

    # Affichage des vecteurs en couleur correspondante
    if title == "blue":
        color = 'blue'
    elif title == "green":
        color = 'green'
    else:
        color = 'red'

    ax.scatter(vector_endpoints[:, 0], vector_endpoints[:, 1], vector_endpoints[:, 2], color=color)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title(f'3D Light Direction Vectors for {title}')

    # Calculate and plot the convex hull
    convex_hull = ConvexHull(vector_endpoints)
    for simplex in convex_hull.simplices:
        ax.plot(vector_endpoints[simplex, 0], vector_endpoints[simplex, 1], vector_endpoints[simplex, 2], color=color, alpha=0.3)

    plt.legend()
    plt.show()

def find_nearest_sphere(light_coords, points):
    min_distance = float('inf')
    nearest_sphere_center = None

    for sphere_center in light_coords:
        distances = [distance.euclidean(sphere_center, point) for point in points]
        avg_distance = np.mean(distances)

        if avg_distance < min_distance:
            min_distance = avg_distance
            nearest_sphere_center = sphere_center

    return nearest_sphere_center, min_distance

pixel_coords_bleu = (404, 404)  # Coordonnées du pixel bleu
pixel_coords_vert = (140, 140) 
pixel_coords_rouge = (315, 92) 

image_path = "/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/Data/Torus_RGB_71ims"
images = load_images_from_folder(image_path)
file_path = '/Users/clementinegrethen/Documents/Enseeiht/enseeiht 3rd year/BEPI3D/BEPI3D/utils/lumières /lumiere_torusRGB_71ims.txt'
light_coords = read_file(file_path)

# Plot each point cloud and the nearest sphere separately
plot_light_directions(images, light_coords, pixel_coords_bleu, "blue")
plot_light_directions(images, light_coords, pixel_coords_vert, "green")
plot_light_directions(images, light_coords, pixel_coords_rouge, "red")





