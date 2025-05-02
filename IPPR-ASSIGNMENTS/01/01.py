
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Split the RGB channels
r, g, b = cv2.split(img)

# Create CMY channels
c = 255 - r
m = 255 - g
y = 255 - b

# Display RGB channels
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Red Channel')
plt.imshow(r, cmap='Reds')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Green Channel')
plt.imshow(g, cmap='Greens')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Blue Channel')
plt.imshow(b, cmap='Blues')
plt.axis('off')
plt.tight_layout()
plt.show()

# Display CMY channels
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Cyan Channel')
plt.imshow(c, cmap='winter')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Magenta Channel')
plt.imshow(m, cmap='magma')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Yellow Channel')
plt.imshow(y, cmap='YlOrBr')
plt.axis('off')
plt.tight_layout()
plt.show()
