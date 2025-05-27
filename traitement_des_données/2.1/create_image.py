import numpy as np
import cv2

# 1. Créer une image RGB depuis zéro (par exemple, 256x256, couleur bleue)
img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
img_rgb[:] = (255, 0, 0)  # Bleu en OpenCV (BGR)

cv2.imwrite('image_rgb.png', img_rgb)

# 2. Créer une image en niveaux de gris (par exemple, un dégradé horizontal)
img_gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
cv2.imwrite('image_gris.png', img_gray)

# 3. Créer une image binaire (par exemple, un carré blanc sur fond noir)
img_bin = np.zeros((256, 256), dtype=np.uint8)
img_bin[64:192, 64:192] = 255  # Carré blanc au centre
cv2.imwrite('image_binaire.png', img_bin)

