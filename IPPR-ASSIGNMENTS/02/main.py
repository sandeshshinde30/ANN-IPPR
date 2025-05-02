import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Task A: 24-bit color negative
negative_color = 255 - img

# Task B: Convert to grayscale first
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
negative_gray = 255 - gray

# Display results
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title('Original Image (24-bit Color)')
plt.imshow(img)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Negative Image (24-bit Color)')
plt.imshow(negative_color)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Grayscale Image (8-bit)')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Negative Grayscale Image (8-bit)')
plt.imshow(negative_gray, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()