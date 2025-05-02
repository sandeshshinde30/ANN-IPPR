import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from the specified path."""
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Could not find or open image at {image_path}")
    
    return img

def convert_to_grayscale(image):
    """Convert a color image to grayscale."""
    if len(image.shape) == 3:  # If it's a color image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_img
    return image  # Return as is if already grayscale

def calculate_histogram(image):
    """Calculate the histogram of a grayscale image."""
    # Calculate histogram using OpenCV
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def plot_image_and_histogram(image, hist, title="Image Histogram"):
    """Plot the image and its histogram side by side."""
    # Create figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the image
    if len(image.shape) == 3:
        # Convert BGR to RGB for display
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(display_img)
    else:
        ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot the histogram
    ax2.plot(hist)
    ax2.set_title('Histogram')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def equalize_histogram(image):
    """Apply histogram equalization to enhance image contrast."""
    equalized = cv2.equalizeHist(image)
    return equalized

def plot_histogram_comparison(original, equalized):
    """Plot original and equalized images with their histograms."""
    # Create figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot original histogram
    hist_original = cv2.calcHist([original], [0], None, [256], [0, 256])
    axes[0, 1].plot(hist_original, color='b')
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot equalized image
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title('Equalized Image')
    axes[1, 0].axis('off')
    
    # Plot equalized histogram
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    axes[1, 1].plot(hist_equalized, color='r')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Histogram Equalization Comparison')
    plt.tight_layout()
    return fig

def apply_threshold(image, threshold_value=127):
    """Apply binary thresholding to the image."""
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def apply_adaptive_threshold(image):
    """Apply adaptive thresholding to the image."""
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return adaptive_thresh

def plot_thresholding_comparison(original, binary_thresh, adaptive_thresh):
    """Plot original and thresholded images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot binary thresholded image
    axes[1].imshow(binary_thresh, cmap='gray')
    axes[1].set_title('Binary Threshold')
    axes[1].axis('off')
    
    # Plot adaptive thresholded image
    axes[2].imshow(adaptive_thresh, cmap='gray')
    axes[2].set_title('Adaptive Threshold')
    axes[2].axis('off')
    
    plt.suptitle('Thresholding Comparison')
    plt.tight_layout()
    return fig

def main():
    # Example usage with an image path
    try:
        # Path to your image
        image_path = "img.jpg"  # Replace with your image path
        
        # Load the image
        img = load_image(image_path)
        
        # Convert to grayscale
        gray_img = convert_to_grayscale(img)
        
        # Calculate histogram
        hist = calculate_histogram(gray_img)
        
        # Plot image and histogram
        fig1 = plot_image_and_histogram(img, hist)
        plt.savefig('original_histogram.png')
        
        # Apply histogram equalization
        equalized = equalize_histogram(gray_img)
        
        # Plot comparison between original and equalized
        fig2 = plot_histogram_comparison(gray_img, equalized)
        plt.savefig('equalization_comparison.png')
        
        # Apply thresholding
        binary_thresh = apply_threshold(gray_img)
        adaptive_thresh = apply_adaptive_threshold(gray_img)
        
        # Plot thresholding comparison
        fig3 = plot_thresholding_comparison(gray_img, binary_thresh, adaptive_thresh)
        plt.savefig('threshold_comparison.png')
        
        # Show all plots
        plt.show()
        
        print("Image histogram analysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()