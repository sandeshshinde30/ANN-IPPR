import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image properly
img = cv2.imread('img.jpg')
if img is None:
    raise FileNotFoundError("Could not load image 'img.jpg'")
    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel Edge Detection (already correct)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# Fixed Prewitt Implementation
prewitt_kernel_x = np.array([[-1, 0, 1], 
                            [-1, 0, 1], 
                            [-1, 0, 1]], dtype=np.float32)
                            
prewitt_kernel_y = np.array([[-1, -1, -1], 
                            [0, 0, 0], 
                            [1, 1, 1]], dtype=np.float32)

# Convert to float32 and specify ddepth
prewitt_x = cv2.filter2D(gray.astype(np.float32), -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(gray.astype(np.float32), -1, prewitt_kernel_y)
prewitt_edges = cv2.magnitude(prewitt_x, prewitt_y)

# Canny Edge Detection
canny_edges = cv2.Canny(gray, 100, 200)

# Visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(gray, cmap='gray'), plt.title('Original')
plt.subplot(2, 2, 2), plt.imshow(sobel_edges, cmap='viridis'), plt.title('Sobel')
plt.subplot(2, 2, 3), plt.imshow(prewitt_edges, cmap='plasma'), plt.title('Prewitt')
plt.subplot(2, 2, 4), plt.imshow(canny_edges, cmap='inferno'), plt.title('Canny')
plt.tight_layout()
plt.show()