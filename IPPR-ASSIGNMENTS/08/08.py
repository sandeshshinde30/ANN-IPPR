import cv2
import numpy as np

# 📸 Step 1: Read the image
img = cv2.imread('img.jpg', 0)  # Load in grayscale

# 🧱 Step 2: Create a kernel (structuring element)
kernel = np.ones((5, 5), np.uint8)  # 5x5 square of 1s

# 🧼 Step 3: Apply Morphological Operations

# Erosion - removes small white noises
erosion = cv2.erode(img, kernel, iterations=1)

# Dilation - increases white region
dilation = cv2.dilate(img, kernel, iterations=1)

# Opening - erosion followed by dilation
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing - dilation followed by erosion
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 💾 Step 4: Save results
cv2.imwrite('erosion.jpg', erosion)
cv2.imwrite('dilation.jpg', dilation)
cv2.imwrite('opening.jpg', opening)
cv2.imwrite('closing.jpg', closing)

print("✅ Morphological operations applied and images saved!")
