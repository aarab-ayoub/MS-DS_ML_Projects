import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger et afficher une image du dataset
img = cv2.imread('../1.jpg')  # Remplace par le chemin réel
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Afficher les caractéristiques
print(f"Dimensions : {img.shape}")
print(f"Type : {img.dtype}")
print(f"Taille (nombre de pixels) : {img.size}")

# 3. Générer et interpréter l’histogramme des couleurs
colors = ('b', 'g', 'r')
plt.figure()
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('Histogramme des couleurs')
plt.xlabel('Intensité')
plt.ylabel('Nombre de pixels')
plt.show()

# 4. Normaliser l’image (valeurs entre 0 et 1)
img_norm = img.astype(np.float32) / 255.0

# 5. Modifier un ensemble de pixels (par exemple, mettre un carré rouge en haut à gauche)
img_mod = img.copy()
img_mod[0:50, 0:50] = [0, 0, 255]  # Rouge en BGR
cv2.imwrite('image_modifiee.jpg', img_mod)


# PART 2.3

# --- Conversion couleur → niveaux de gris
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('image_gris.jpg', img_gray)

# --- Seuillage et détection de contours
_, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(img_gray, 100, 200)
cv2.imwrite('image_seuil.jpg', img_thresh)
cv2.imwrite('image_contours.jpg', edges)

# --- Débruitage avec différents filtres
img_blur = cv2.GaussianBlur(img, (7, 7), 0)
img_median = cv2.medianBlur(img, 5)
cv2.imwrite('image_gaussienne.jpg', img_blur)
cv2.imwrite('image_mediane.jpg', img_median)

# --- Opérations morphologiques : dilatation et érosion
kernel = np.ones((5, 5), np.uint8)
img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
img_erode = cv2.erode(img_thresh, kernel, iterations=1)
cv2.imwrite('image_dilatee.jpg', img_dilate)
cv2.imwrite('image_erodee.jpg', img_erode)

# --- Compression avec/sans perte
cv2.imwrite('image_jpeg_qualite90.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # Avec perte
cv2.imwrite('image_png_sans_perte.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # Sans perte






# --- Ajouter du bruit Gaussien ---
# def add_gaussian_noise(image, mean=0, sigma=25):
#     gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
#     noisy = image.astype(np.float32) + gauss
#     noisy = np.clip(noisy, 0, 255).astype(np.uint8)
#     return noisy

# img_gauss_noise = add_gaussian_noise(img)
# cv2.imwrite('image_bruit_gaussien.jpg', img_gauss_noise)

# # --- Ajouter du bruit sel-et-poivre ---
# def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
#     noisy = image.copy()
#     num_salt = np.ceil(amount * image.size * salt_vs_pepper)
#     num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

#     # Sel (pixels blancs)
#     coords = [np.random.randint(0, i - 1, int(num_salt))
#               for i in image.shape[:2]]
#     noisy[coords[0], coords[1]] = 255

#     # Poivre (pixels noirs)
#     coords = [np.random.randint(0, i - 1, int(num_pepper))
#               for i in image.shape[:2]]
#     noisy[coords[0], coords[1]] = 0

#     return noisy

# img_sp_noise = add_salt_pepper_noise(img)
# cv2.imwrite('image_bruit_selpoivre.jpg', img_sp_noise)

# # --- Filtrage sur images bruitées ---
# img_gauss_blur = cv2.GaussianBlur(img_gauss_noise, (7, 7), 0)
# img_gauss_median = cv2.medianBlur(img_gauss_noise, 5)
# cv2.imwrite('image_bruit_gaussien_gaussienne.jpg', img_gauss_blur)
# cv2.imwrite('image_bruit_gaussien_mediane.jpg', img_gauss_median)

# img_sp_blur = cv2.GaussianBlur(img_sp_noise, (7, 7), 0)
# img_sp_median = cv2.medianBlur(img_sp_noise, 5)
# cv2.imwrite('image_bruit_selpoivre_gaussienne.jpg', img_sp_blur)
# cv2.imwrite('image_bruit_selpoivre_mediane.jpg', img_sp_median)
# # ...existing code...