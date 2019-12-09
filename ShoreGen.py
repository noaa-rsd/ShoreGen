# ShoreGen

from pathlib import Path
import matplotlib.pyplot as plt

from skimage import data, color, io, measure
from skimage.transform import rescale, resize, downscale_local_mean


img_path = Path(r'Z:\ShoreGen\US4AK4LF_POLY.tif')

print('reading image...')
img = io.imread(img_path)

rescale_factor = 0.1
print('rescaling image ({})...'.format(rescale_factor))
img_rescaled = rescale(img, rescale_factor, anti_aliasing=True)

print('creating contours...')
contours = measure.find_contours(img_rescaled, 0.0)

fig, axes = plt.subplots(nrows=1, ncols=2)

ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(img_rescaled, cmap='gray')
ax[1].set_title("Rescaled Image (0.1)")

n = 20000
e = 9000
s = 16000
w = 5000

ax[0].set_xlim(s, n)
ax[0].set_ylim(w, e)

n_r = 20000 * rescale_factor
e_r = 9000 * rescale_factor
s_r = 16000 * rescale_factor
w_r = 5000 * rescale_factor

ax[1].set_xlim(s_r, n_r)
ax[1].set_ylim(w_r, e_r)

for n, contour in enumerate(contours):
    ax[1].plot(contour[:, 1], contour[:, 0], linewidth=1, color='orange')

plt.tight_layout()
plt.show()
