from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np


class Resnet_Extractor:
    def __init__(self):
        """Initialize ResNet50 model"""
        self.model = ResNet50(weights='imagenet', 
                            include_top=False, 
                            pooling='avg',
                            input_shape=(224, 224, 3))

    def extract_features(self, images):
        """
        Extract features using ResNet50
        Args:
            images: Path to image file or numpy array
        Returns:
            numpy array of features
        """
        try:
            img = image.load_img(images, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Use ResNet50 preprocessing

            features = self.model.predict(img_array, verbose=0)
            return features.flatten()  # Return flattened feature vector
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None