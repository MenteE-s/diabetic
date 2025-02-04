import cv2
import numpy as np

class RetinalVesselDetector:
    def __init__(self, image_path=None):
        if image_path:
            self.load_image(image_path)
        self.mask_cache = None
        
    def load_image(self, image_path):
        """Load and validate the input image"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
    def preprocess(self, image):
        """Preprocess the image for vessel detection"""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10)
        return denoised
        
    def detect_vessels(self):
        """Detect vessels using multiple techniques"""
        # Preprocess
        preprocessed = self.preprocess(self.image)
        
        # Create structuring elements
        line_kernel = np.ones((3,3), np.uint8)
        
        # Apply morphological operations
        gradient = cv2.morphologyEx(preprocessed, cv2.MORPH_GRADIENT, line_kernel)
        
        # Enhance vessel structures
        enhanced = cv2.addWeighted(preprocessed, 1, gradient, -0.5, 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            2
        )
        
        # Clean up noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        self.mask_cache = cleaned
        return cleaned
        
    def create_visualization(self):
        """Create colored visualization of detected vessels"""
        if self.mask_cache is None:
            self.detect_vessels()
            
        # Create visualization
        result = self.image.copy()
        result[self.mask_cache > 5] = [0, 0, 255]  # Red for vessels
        
        return result
        
    def process_image(self):
        """Main processing pipeline"""
        vessel_mask = self.detect_vessels()
        visualization = self.create_visualization()
        return visualization, vessel_mask

def main():
    # File path
    image_path = r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\raw\Diabetic retinopathy\Severe\0f96c358a250.png"
    
    try:
        # Initialize detector
        detector = RetinalVesselDetector(image_path)
        
        # Process image
        result, mask = detector.process_image()
        
        # Display results
        cv2.imshow('Original', detector.image)
        cv2.imshow('Vessel Detection', result)
        # cv2.imshow('Binary Mask', mask)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()