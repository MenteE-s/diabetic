import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model
import numpy as np

class CycleGAN:
    def __init__(self, img_shape=(256, 256, 3)):
        self.img_shape = img_shape

        self.G_XtoY = self.build_generator()  # Healthy → Diseased
        self.G_YtoX = self.build_generator()  # Diseased → Healthy
        self.D_Y = self.build_discriminator()  # Discriminates Diseased images
        self.D_X = self.build_discriminator()  # Discriminates Healthy images

    def build_generator(self):
        inputs = Input(shape=self.img_shape)
        
        x = Conv2D(128, kernel_size=5, padding="same")(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, kernel_size=5, padding="same")(x)
        x = tf.keras.layers.ReLU()(x)
        x = BatchNormalization()(x)

        for _ in range(4):
            
            res = x

            x = Conv2D(128, kernel_size=5, padding="same")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)

            x = Conv2D(64, kernel_size=5, padding="same")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)

            x = tf.keras.layers.Add()([x, res])

        x = Conv2DTranspose(64, kernel_size=5, padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        outputs = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(x)
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

    def generate_diseased(self, image):
        """Use G_XtoY to create a synthetic diseased image"""
        image = np.expand_dims(image, axis=0)
        fake_diseased = self.G_XtoY.predict(image)
        return np.squeeze(fake_diseased, axis=0)

    def generate_healthy(self, image):
        """Use G_YtoX to create a synthetic healthy image"""
        image = np.expand_dims(image, axis=0)
        fake_healthy = self.G_YtoX.predict(image)
        return np.squeeze(fake_healthy, axis=0)

    def classify_real_or_fake(self, image, domain="diseased"):
        """Use the respective discriminator to classify real/fake"""
        image = np.expand_dims(image, axis=0)
        if domain == "diseased":
            return self.D_Y.predict(image)[0][0]
        else:
            return self.D_X.predict(image)[0][0]
