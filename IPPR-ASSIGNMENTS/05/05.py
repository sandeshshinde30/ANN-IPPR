# 📁 image_transform_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pathlib import Path

# 🛠️ Utility Functions with Error Checking
def load_image(image_path):
    """Load grayscale image with proper error handling"""
    try:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file {image_path} not found")
            
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to read image file (might be corrupted)")
            
        return img
    except Exception as e:
        raise RuntimeError(f"Image loading failed: {str(e)}")

def resize_image(image, max_size=512):
    """Resize image with even dimensions for wavelet transform"""
    try:
        h, w = image.shape
        if max(h, w) > max_size:
            ratio = max_size / max(h, w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            # Ensure even dimensions for wavelet transforms
            new_h = new_h // 2 * 2
            new_w = new_w // 2 * 2
            resized = cv2.resize(image, (new_w, new_h))
        else:
            # Still ensure even dimensions
            new_h = h // 2 * 2
            new_w = w // 2 * 2
            resized = cv2.resize(image, (new_w, new_h))
            
        if resized.size == 0:
            raise ValueError("Resizing resulted in empty image")
        return resized
    except Exception as e:
        raise RuntimeError(f"Resizing failed: {str(e)}")

# 🌀 DFT Functions with Dimension Handling
def apply_dft(image):
    """Apply DFT with proper padding and dimension tracking"""
    try:
        original_h, original_w = image.shape
        optimal_h = cv2.getOptimalDFTSize(original_h)
        optimal_w = cv2.getOptimalDFTSize(original_w)
        
        padded = cv2.copyMakeBorder(image, 0, optimal_h - original_h, 0,
                                   optimal_w - original_w, cv2.BORDER_CONSTANT, value=0)
        
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude = np.log(1 + cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
        magnitude_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        return dft_shift, magnitude_norm, (original_h, original_w)
    except Exception as e:
        raise RuntimeError(f"DFT failed: {str(e)}")

def inverse_dft(dft_shift, original_shape):
    """Inverse DFT with cropping to original dimensions"""
    try:
        dft_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(dft_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        img_back = cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)
        return img_back[:original_shape[0], :original_shape[1]]
    except Exception as e:
        raise RuntimeError(f"Inverse DFT failed: {str(e)}")

# 🔄 DCT Functions
def apply_dct(image):
    """Apply DCT with error checking"""
    try:
        if image.dtype != np.float32:
            image = np.float32(image)
        dct = cv2.dct(image)
        log_dct = np.log(abs(dct) + 1)
        return dct, cv2.normalize(log_dct, None, 0, 1, cv2.NORM_MINMAX)
    except Exception as e:
        raise RuntimeError(f"DCT failed: {str(e)}")

def inverse_dct(dct):
    """Inverse DCT with normalization"""
    try:
        img_back = cv2.idct(dct)
        return cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)
    except Exception as e:
        raise RuntimeError(f"Inverse DCT failed: {str(e)}")

# 🌊 DWT Functions with Dimension Safety
def apply_dwt(image):
    """Apply Discrete Wavelet Transform with safety checks"""
    try:
        if image.shape[0] % 2 != 0 or image.shape[1] % 2 != 0:
            raise ValueError("Image dimensions must be even for DWT")
            
        coeffs = pywt.dwt2(image, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # Visualization normalization
        def normalize_band(band):
            return cv2.normalize(band, None, 0, 1, cv2.NORM_MINMAX)
            
        combined = np.zeros((2*LL.shape[0], 2*LL.shape[1]))
        combined[:LL.shape[0], :LL.shape[1]] = normalize_band(LL)
        combined[:LL.shape[0], LL.shape[1]:] = normalize_band(LH)
        combined[LL.shape[0]:, :LL.shape[1]] = normalize_band(HL)
        combined[LL.shape[0]:, LL.shape[1]:] = normalize_band(HH)
        
        return coeffs, combined
    except Exception as e:
        raise RuntimeError(f"DWT failed: {str(e)}")

def inverse_dwt(coeffs):
    """Inverse DWT with normalization"""
    try:
        img_back = pywt.idwt2(coeffs, 'haar')
        return cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)
    except Exception as e:
        raise RuntimeError(f"Inverse DWT failed: {str(e)}")

# 📈 Compression Functions with Validation
def validate_percentage(percentage):
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

def apply_compression_dft(dft_shift, percentage, original_shape):
    """DFT compression with validation"""
    try:
        validate_percentage(percentage)
        rows, cols = dft_shift.shape[:2]
        crow, ccol = rows//2, cols//2
        
        radius = int(min(rows, cols) * percentage / 200)
        mask = np.zeros((rows, cols, 2), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
        mask[mask_area] = 1
        
        return dft_shift * mask
    except Exception as e:
        raise RuntimeError(f"DFT compression failed: {str(e)}")

def apply_compression_dct(dct, percentage):
    """DCT compression with thresholding"""
    try:
        validate_percentage(percentage)
        threshold = np.percentile(abs(dct), 100 - percentage)
        dct_copy = dct.copy()
        dct_copy[abs(dct_copy) < threshold] = 0
        return dct_copy
    except Exception as e:
        raise RuntimeError(f"DCT compression failed: {str(e)}")

def apply_compression_dwt(coeffs, percentage):
    """DWT compression with validation"""
    try:
        validate_percentage(percentage)
        LL, (LH, HL, HH) = coeffs
        details = np.concatenate([LH.ravel(), HL.ravel(), HH.ravel()])
        threshold = np.percentile(abs(details), 100 - percentage)
        
        LH_comp = np.where(abs(LH) >= threshold, LH, 0)
        HL_comp = np.where(abs(HL) >= threshold, HL, 0)
        HH_comp = np.where(abs(HH) >= threshold, HH, 0)
        
        return (LL, (LH_comp, HL_comp, HH_comp))
    except Exception as e:
        raise RuntimeError(f"DWT compression failed: {str(e)}")

# 📊 Visualization and Metrics
def plot_transform_results(original, transform, reconstructed, title):
    """Safe plotting function"""
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(original, cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(transform, cmap='viridis')
        ax[1].set_title(f'{title} Transform')
        ax[2].imshow(reconstructed, cmap='gray')
        ax[2].set_title('Reconstructed')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Plotting failed: {str(e)}")
        return None

def calculate_metrics(original, processed):
    """Calculate MSE and PSNR safely"""
    try:
        mse = np.mean((original - processed)**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        return mse, psnr
    except Exception as e:
        print(f"Metric calculation failed: {str(e)}")
        return None, None

# 🏁 Main Workflow with Comprehensive Error Handling
def main():
    try:
        # Configuration
        IMAGE_PATH = 'img.jpg'  # 🖼️ Change to your image path
        COMPRESSION_LEVELS = [90, 70, 50, 30, 10]
        
        # Load and preprocess
        img = load_image(IMAGE_PATH)
        img = resize_image(img)
        img_normalized = img.astype(np.float32) / 255.0
        
        # DFT Analysis
        try:
            dft_shift, dft_vis, orig_shape = apply_dft(img)
            dft_recon = inverse_dft(dft_shift, orig_shape)
            plot_transform_results(img_normalized, dft_vis, dft_recon, 'DFT')
            plt.savefig('dft_results.png')
        except Exception as e:
            print(f"DFT processing failed: {str(e)}")
        
        # DCT Analysis
        try:
            dct_coeffs, dct_vis = apply_dct(img_normalized)
            dct_recon = inverse_dct(dct_coeffs)
            plot_transform_results(img_normalized, dct_vis, dct_recon, 'DCT')
            plt.savefig('dct_results.png')
        except Exception as e:
            print(f"DCT processing failed: {str(e)}")
        
        # DWT Analysis
        try:
            dwt_coeffs, dwt_vis = apply_dwt(img_normalized)
            dwt_recon = inverse_dwt(dwt_coeffs)
            plot_transform_results(img_normalized, dwt_vis, dwt_recon, 'DWT')
            plt.savefig('dwt_results.png')
        except Exception as e:
            print(f"DWT processing failed: {str(e)}")
        
        # Compression Comparison
        try:
            fig, ax = plt.subplots(3, len(COMPRESSION_LEVELS)+1, figsize=(20, 15))
            
            # Original column
            for row in range(3):
                ax[row,0].imshow(img_normalized, cmap='gray')
                ax[row,0].axis('off')
                if row == 0:
                    ax[row,0].set_title('Original')
            
            # Compression levels
            for col, percent in enumerate(COMPRESSION_LEVELS, start=1):
                # DFT
                try:
                    dft_comp = apply_compression_dft(dft_shift, percent, orig_shape)
                    dft_recon = inverse_dft(dft_comp, orig_shape)
                    ax[0,col].imshow(dft_recon, cmap='gray')
                except Exception as e:
                    ax[0,col].text(0.5, 0.5, f"Error: {str(e)}", ha='center')
                ax[0,col].set_title(f'DFT {percent}%')
                ax[0,col].axis('off')
                
                # DCT
                try:
                    dct_comp = apply_compression_dct(dct_coeffs, percent)
                    dct_recon = inverse_dct(dct_comp)
                    ax[1,col].imshow(dct_recon, cmap='gray')
                except Exception as e:
                    ax[1,col].text(0.5, 0.5, f"Error: {str(e)}", ha='center')
                ax[1,col].set_title(f'DCT {percent}%')
                ax[1,col].axis('off')
                
                # DWT
                try:
                    dwt_comp = apply_compression_dwt(dwt_coeffs, percent)
                    dwt_recon = inverse_dwt(dwt_comp)
                    ax[2,col].imshow(dwt_recon, cmap='gray')
                except Exception as e:
                    ax[2,col].text(0.5, 0.5, f"Error: {str(e)}", ha='center')
                ax[2,col].set_title(f'DWT {percent}%')
                ax[2,col].axis('off')
            
            plt.suptitle('Compression Comparison')
            plt.savefig('compression_comparison.png')
            plt.close()
        except Exception as e:
            print(f"Compression comparison failed: {str(e)}")
        
        plt.show()
        print("✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")

if __name__ == "__main__":
    main()