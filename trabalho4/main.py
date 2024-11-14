# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# processo: normalizacao local -> binarizacao -> segmentacao -> ver no que dá
import cv2, numpy as np, sys, os
from segmentation import *
from img_utils import *
from operations import *

assert len(sys.argv) >= 2, "Por favor, insira como argumento para o script um caminho para uma imagem"
img_path = sys.argv[1]
assert os.path.exists(img_path), "Por favor, insira um caminho válido"

img = open_image(img_path)
img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

binarized_img = normalize_locally_and_binarize_image(img)
non_noisy_img = supress_image_noise(binarized_img)
# erodida1 = erodida.copy().reshape((img.shape[0], img.shape[1], 1))
# erodida = cv2.erode(erodida, np.ones((3, 3)))
# erodida2 = erodida.copy().reshape((img.shape[0], img.shape[1], 1))
# erodida = cv2.erode(erodida, np.ones((3, 3)))
# # erodida = cv2.erode(erodida, np.ones((3, 3)))


# rotula
componentes = label(non_noisy_img, 0, 0, 0)
n_componentes = len(componentes)
print("%d componentes detectados." % n_componentes)

# Mostra os objetos encontrados.
for c in componentes:
    cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

n_pixels = [_['n_pixels'] for _ in componentes]
mean = np.mean(n_pixels)
std = np.std(n_pixels)
print(
    "max", 
    max(componentes, key=lambda componente: componente['n_pixels'])['n_pixels'], 
    'min', 
    min(componentes, key=lambda componente: componente['n_pixels'])['n_pixels'], 
    "media", mean, 
    "std", std,
    'std / mean', std / mean
)

cv2.imshow("original", img)
cv2.imshow("binarizada", binarized_img)
cv2.imshow("erodida", non_noisy_img)
# cv2.imshow("bordas", binaria - erodida1 - erodida2)
# cv2.imshow("blur", blurry)
# cv2.imshow("diff", diff)
cv2.imshow("saida", img_out)

cv2.waitKey()
cv2.destroyAllWindows()