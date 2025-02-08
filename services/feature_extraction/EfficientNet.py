from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

class EfficientNetExtractor:
    def __init__(self):
        self.model = None
        self.features = None

    def load_model(self):
        """Load EfficientNetB0 without the fully connected top layers."""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        return self
    
    def extract_features(self, img_path):
        """Extract features from an image using EfficientNetB0.
        Image should be in RGB form.
        """
        img = image.load_img(img_path, target_size=(224, 224))  # Load image
        img_array = image.img_to_array(img)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
        img_array = preprocess_input(img_array)  # Preprocess for EfficientNet

        self.features = self.model.predict(img_array)  # Extract features
        return self.features  # Return feature vector

# Example Usage:
# extractor = EfficientNetExtractor().load_model()
# features = extractor.extract_features("path/to/image.jpg")
# print(features.shape)  # Check feature shape
