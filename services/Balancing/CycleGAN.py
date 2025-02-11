import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model
import numpy as np

class CycleGAN:
    def __init__(self, img_shape=(256, 256, 3)):
        self.img_shape = img_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        inputs = Input(shape=self.img_shape)
        
        x = Conv2D(128, kernel_size=4, padding="same")(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, kernel_size=4, padding="same")(x)
        x = tf.keras.layers.ReLU()(x)
        x = BatchNormalization()(x)

        for _ in range(2):
            
            res = x

            x = Conv2D(128, kernel_size=3, padding="same")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)

            x = Conv2D(64, kernel_size=3, padding="same")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)

            x = tf.keras.layers.Add()([x, res])

        x = Conv2DTranspose(64, kernel_size=3, padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        outputs = Conv2D(3, kernel_size=4, padding="same", activation="tanh")(x)
        return Model(inputs, outputs, name="Generator")

    def build_discriminator(self):
        inputs = Input(shape=self.img_shape)
        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(inputs)
        x = LeakyReLU(alpha=0.2)(x)

        
        x = tf.keras.layers.Dropout(0.3)(x)

        x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        

        # x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.2)(x)

        # x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(1, kernel_size=4, padding="same", activation="sigmoid")(x)
        return Model(inputs, x, name="Discriminator")

    def generate_image(self, image):
        """Transforms a single image using the Generator"""
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        fake_image = self.generator.predict(image)
        return np.squeeze(fake_image, axis=0)  # Remove batch dimension

    def classify_real_or_fake(self, image):
        """Discriminator classifies if an image is real (1) or fake (0)"""
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.discriminator.predict(image)
        return prediction[0][0]
