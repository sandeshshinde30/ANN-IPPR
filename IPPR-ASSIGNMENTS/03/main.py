import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Translation
def translate_point(x, y, tx, ty):
    return x + tx, y + ty

# 2. Rotation (90° CCW)
def rotate_point_90_ccw(x, y):
    return -y, x

# 3. Scaling
def scale_point(x, y, sx, sy):
    return sx * x, sy * y

# 4. Shearing (x-axis)
def shear_point_x(x, y, shx):
    return x + shx * y, y

# --- Point Transformations ---
print("1. Translation:", translate_point(2, 3, 4, 5))
print("2. Rotation (90° CCW):", rotate_point_90_ccw(2, 3))
print("3. Scaling:", scale_point(3, 4, 2, 3))
print("4. Shearing (x-axis):", shear_point_x(2, 3, 2))

# --- Image Transformations ---
img = cv2.imread('img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[:2]

# Translation matrix: move right by 50, down by 30
M_translate = np.float32([[1, 0, 50], [0, 1, 30]])
img_translated = cv2.warpAffine(img, M_translate, (cols, rows))

# Rotation matrix: rotate 45° around center
M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
img_rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

# Scaling: scale by 1.5x in both directions
img_scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Shearing matrix (x-axis shear)
M_shear = np.float32([[1, 0.5, 0], [0, 1, 0]])
img_sheared = cv2.warpAffine(img, M_shear, (int(cols*1.5), rows))

# --- Display Results ---
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.title('Original')
plt.imshow(img)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Translated')
plt.imshow(img_translated)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Rotated')
plt.imshow(img_rotated)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Scaled')
plt.imshow(img_scaled)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Sheared')
plt.imshow(img_sheared)
plt.axis('off')

plt.tight_layout()
plt.show()