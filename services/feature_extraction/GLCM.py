import cv2
import mahotas  # For GLCM
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Applying_GLCM:
    def __init__(self):
        pass

    def extract_glcm_features(image):
        """Extract GLCM features from a grayscale image"""
        # Calculate GLCM matrix
        distances = 1
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = mahotas.features.haralick(image, distance=distances, return_mean=True)
        # Return first 4 features: Contrast, Correlation, Energy, Homogeneity
        return glcm[:4]

    def extract_grayscale_histogram(image, bins=32):
        """Extract histogram from a grayscale image"""
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
        return hist
