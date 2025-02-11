from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
import numpy as np


class Autoencoder:
    def __init__(self):
        self.autoencoder = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(256, 256, 3))
        encoder = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
        decoder = Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(encoder)
        return Model(input_layer, decoder)

    def reconstruct_image(self, image):
        return self.autoencoder.predict(np.expand_dims(image, axis=0))[0]



