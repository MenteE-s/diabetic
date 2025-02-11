import pywt
import numpy as np

class WaveletFeatureExtractor:
    def __init__(self, wavelet="haar", level=2):
        """
        Initialize the wavelet feature extractor.
        
        :param wavelet: The type of wavelet to use (default: 'haar')
        :param level: Decomposition level (default: 2)
        """
        self.wavelet = wavelet
        self.level = level

    def extract_wavelet_features(self, image):
        """
        Extracts wavelet transform features from a grayscale image.
        
        :param image: Input grayscale image as a 2D NumPy array.
        :return: Flattened feature vector.
        """
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        features = self.compute_statistical_features(coeffs)
        return features

    def compute_statistical_features(self, coeffs):
        """
        Computes statistical features from wavelet coefficients.
        
        :param coeffs: Wavelet coefficients (tuple of arrays)
        :return: Flattened feature vector.
        """
        feature_vector = []

        for i, coeff in enumerate(coeffs):
            if isinstance(coeff, tuple):
                # Detail coefficients (horizontal, vertical, diagonal)
                for sub_band in coeff:
                    feature_vector.extend(self.compute_basic_stats(sub_band))
            else:
                # Approximation coefficients
                feature_vector.extend(self.compute_basic_stats(coeff))

        return np.array(feature_vector)

    def compute_basic_stats(self, coeff):
        """
        Computes basic statistical features for a given coefficient matrix.
        
        :param coeff: 2D array of wavelet coefficients.
        :return: List of statistical features (mean, std, energy, entropy).
        """
        mean = np.mean(coeff)
        std_dev = np.std(coeff)
        energy = np.sum(coeff**2)
        entropy = -np.sum(coeff * np.log2(np.abs(coeff) + 1e-10))  # Small epsilon to avoid log(0)

        return [mean, std_dev, energy, entropy]
