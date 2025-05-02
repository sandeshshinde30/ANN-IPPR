# 📁 image_processing.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 📸 Read image
img = cv2.imread('img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# 🌌 Spatial Domain Processing
# 1. Gaussian Blur (Smoothing)
gaussian = cv2.GaussianBlur(img, (5,5), 0)

# 2. Median Blur (De-noising)
median = cv2.medianBlur(img, 5)

# 3. Sharpening
kernel = np.array([[-1,-1,-1], 
                   [-1,9,-1], 
                   [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, kernel)

# 📡 Frequency Domain Processing
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 1. Fourier Transform
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

# 2. Create Low-pass Filter (Smoothing)
rows, cols = gray.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # 🌀 Low-pass mask
fshift_low = fshift * mask
low_pass = np.fft.ifft2(np.fft.ifftshift(fshift_low)).real

# 3. Create High-pass Filter (Sharpening)
mask = np.ones((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # 🏹 High-pass mask
fshift_high = fshift * mask
high_pass = np.fft.ifft2(np.fft.ifftshift(fshift_high)).real

# 🖼️ Display results
plt.figure(figsize=(18,10))

# Original
plt.subplot(2,3,1), plt.imshow(img)
plt.title('Original Image'), plt.axis('off')

# Spatial Domain
plt.subplot(2,3,2), plt.imshow(gaussian)
plt.title('Gaussian Blur'), plt.axis('off')

plt.subplot(2,3,3), plt.imshow(median)
plt.title('Median Blur'), plt.axis('off')

plt.subplot(2,3,4), plt.imshow(sharpened)
plt.title('Sharpened'), plt.axis('off')

# Frequency Domain
plt.subplot(2,3,5), plt.imshow(low_pass, cmap='gray')
plt.title('Low-pass Filter'), plt.axis('off')

plt.subplot(2,3,6), plt.imshow(high_pass, cmap='gray')
plt.title('High-pass Filter'), plt.axis('off')

plt.tight_layout()
plt.show()