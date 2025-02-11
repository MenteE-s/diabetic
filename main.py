import cv2
import logging
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from services.Preprocess_Images import ImageProcessor
from services.feature_extraction.GLCM import Applying_GLCM
from services.feature_extraction.FDFE import WaveletFeatureExtractor
from services.feature_extraction.resnet import Resnet_Extractor
from services.feature_extraction.EfficientNet import EfficientNetExtractor
from services.Balancing.CycleGAN import CycleGAN

class Processing_and_logging:
    def __init__(self):
        pass

        # Setup logging
    def setup_logging(self):
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"logs.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),  # Use append mode
                logging.StreamHandler()
            ]
        )

        # Add extra line breaks to the log file
        with open(log_file, 'a') as f:
            f.write('\n\n\n')

        return logging.getLogger(__name__)



    def start_processing(self,image_path):
        try:
            logger.info("Starting image processing")

            # image_path = r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\raw\Diabetic retinopathy\Moderate\0c7e82daf5a0.png"
            logger.info(f"Processing image: {image_path}")

            processor = ImageProcessor(image=image_path)
            logger.info("ImageProcessor initialized")

            processor.load_image()
            logger.info("Image loaded successfully")

            processor.to_grayscale()
            logger.info("Converted to grayscale")

            processor.enhance_gray()
            logger.info("Applied CLAHE enhancement")
            return processor.enhanced

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            return None


class Extraction_Functions(Processing_and_logging):
    def __init__(self, image_folder, label):
        super().__init__()
        self.image_folder = image_folder
        self.label = label

    def feature_extraction_using_GLCM(self):
        features_list = []
        self.labels = []

        logger.info(f"Starting feature extraction from images in: {self.image_folder}")

        for filename in os.listdir(self.image_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(self.image_folder, filename)
    
                logger.info(f"Processing file: {filename}")
                image = pipeline_and_everything.start_processing(image_path=image_path)
                # cv2.imshow('Image', image)
    
                if image is not None:
                    # Extract GLCM and histogram features
                    glcm_features = Applying_GLCM.extract_glcm_features(image=image)
                    histogram_features = Applying_GLCM.extract_grayscale_histogram(image=image)
    
                    features = np.hstack([glcm_features, histogram_features])
                    features_list.append(features)
    
                    self.labels.append(lable)
                    logger.info(f"Features extracted for {filename} | labled as {lable}")
                    logger.info(f"Features: {features_list}  \n\n")
    
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        df["label"] = self.labels
    
        # Normalize features (excluding labels)
        scaler = MinMaxScaler()
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
        logger.info("Feature extraction and normalization completed successfully.")
        
        # Optionally, you can save the DataFrame here if everything looks good
        # df.to_csv("processed_features.csv", index=False)
    
        # Show the first few rows to verify
        logger.info(f"Processed data preview:\n{df.head()}")
    def feature_extraction_using_FDFE(self):

        features_list = []
        self.labels = []

        wavelet_extractor = WaveletFeatureExtractor()

        logger.info(f"Starting feature extraction from images in: {self.image_folder}")

        for filename in os.listdir(self.image_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(self.image_folder, filename)

                logger.info(f"Processing file: {filename}")
                image = pipeline_and_everything.start_processing(image_path=image_path)
                # cv2.imshow('Image', image)

                if image is not None:
                    # Extract wavelet features
                    wavelet_features = wavelet_extractor.extract_wavelet_features(image=image)

                    features_list.append(wavelet_features)

                    self.labels.append(lable)
                    logger.info(f"Features extracted for {filename} | labled as {lable}")
                    logger.info(f"Features: {features_list}  \n\n")

        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        df["label"] = self.labels

        # Normalize features (excluding labels)
        scaler = MinMaxScaler()
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

        logger.info("Feature extraction and normalization completed successfully.")

        # Optionally, you can save the DataFrame here if everything looks good
        # df.to_csv("processed_features.csv", index=False)

        # Show the first few rows to verify
        logger.info(f"Processed data preview:\n{df.head()}")
    def feature_extraction_using_resnet(self):

        features_list = []
        self.labels = []

        resnet_extractor = Resnet_Extractor()

        logger.info(f"Starting feature extraction from images in: {self.image_folder}")

        for filename in os.listdir(self.image_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(self.image_folder, filename)

                img = cv2.imread(image_path)
                # cv2.imshow('Image', image)

                if img is not None:
                    # Extract Resnet features
                    resnet_features = resnet_extractor.extract_features(images=image_path)

                    features_list.append(resnet_features)

                    self.labels.append(lable)
                    logger.info(f"Features extracted for {filename} | labled as {lable}")
                    logger.info(f"Features: {features_list}  \n\n")

        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        df["label"] = self.labels

        # Normalize features (excluding labels)
        scaler = MinMaxScaler()
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

        logger.info("Feature extraction and normalization completed successfully.")

        # Optionally, you can save the DataFrame here if everything looks good
        # df.to_csv("processed_features.csv", index=False)

        # Show the first few rows to verify
        logger.info(f"Processed data preview:\n{df.head()}")
    def feature_extraction_using_efficientnet(self):

        features_list = []
        self.labels = []

        efficientnet_extractor = EfficientNetExtractor()
        efficientnet_extractor.load_model()

        logger.info(f"Starting feature extraction from images in: {self.image_folder}")

        for filename in os.listdir(self.image_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(self.image_folder, filename)

                img = cv2.imread(image_path)
                # cv2.imshow('Image', image)

                if img is not None:
                    # Extract EfficientNet Features
                    FDFC_features = efficientnet_extractor.extract_features(img_path=image_path)

                    features_list.append(FDFC_features)

                    self.labels.append(lable)
                    logger.info(f"Features extracted for {filename} | labled as {lable}")
                    logger.info(f"Features: {features_list}  \n\n")

        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        df["label"] = self.labels

        # Normalize features (excluding labels)
        scaler = MinMaxScaler()
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

        logger.info("Feature extraction and normalization completed successfully.")

        # Optionally, you can save the DataFrame here if everything looks good
        # df.to_csv("processed_features.csv", index=False)

        # Show the first few rows to verify
        logger.info(f"Processed data preview:\n{df.head()}")


class Balancing_techniques(Processing_and_logging):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        

    def cycle_gan_Balancing(self):
        # Example usage
        cycle_gan = CycleGAN()
        logger.info(f"Starting feature extraction from images in: {self.image_path}")

        for filename in os.listdir(self.image_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(self.image_path, filename)

                # Load and preprocess image
                img = cv2.imread(image_path)
                resized = cv2.resize(img, (256, 256))

                # Normalize input to [-1, 1] range for generator
                image = resized.astype(np.float32) / 255 
                

                # Generate a diseased version of the image
                fake_diseased = cycle_gan.generate_image(image)

                # Normalize output from [-1,1] back to [0,255]
                # fake_diseased = np.clip((fake_diseased + 1) * 127.5, 0, 255).astype(np.uint8)
                fake_diseased = np.clip(fake_diseased * 255, 0, 255).astype(np.uint8)

                # Classify if it's real or fake
                classification = cycle_gan.classify_real_or_fake(fake_diseased)

                # Convert to BGR for OpenCV display
                if fake_diseased.shape[-1] == 3:
                    # fake_diseased = cv2.cvtColor(fake_diseased, cv2.COLOR_RGB2BGR)
                    fake_diseased = cv2.cvtColor(fake_diseased, cv2.COLOR_RGB2GRAY)
                    # fake_diseased = pipeline_and_everything.start_processing(fake_diseased)

                # Debugging Output
                # print("Min:", fake_diseased.min(), "Max:", fake_diseased.max())

                # Display image
                cv2.imshow("Generated Diseased Image", fake_diseased)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                
                logger.info(f"Generated Image Shape: {fake_diseased.shape}")
                logger.info(f"Discriminator Score: {classification}")  # Close to 1 = real, Close to 0 = fake
                        
                # img = cv2.imread(image_path)
                # resized = cv2.resize(img, (256, 256))
                # image = np.array(resized)

                # # Generate a diseased version of the image
                # fake_diseased = cycle_gan.generate_image(image)

                # # Normalize image from [-1, 1] to [0, 255]
                # fake_diseased = np.clip((fake_diseased + 1) * 127.5, 0, 255).astype(np.uint8)
                
                # # Ensure correct channel order (RGB to BGR) for OpenCV
                # if fake_diseased.shape[-1] == 3:  # Check if the image has 3 channels
                #     fake_diseased = cv2.cvtColor(fake_diseased, cv2.COLOR_RGB2BGR)
                
                # # Debug: Print image statistics to check if it's valid
                # print("Min:", fake_diseased.min(), "Max:", fake_diseased.max())
                
                # # Display the corrected image
                # cv2.imshow("Generated Diseased Image", fake_diseased)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # # Classify if it's real or fake
                # classification = cycle_gan.classify_real_or_fake(fake_diseased)

                # logger.info(f"Generated Image Shape: {fake_diseased.shape}")
                # logger.info(f"Discriminator Score: {classification}")  # Close to 1 = real, Close to 0 = fake
        
        
        




if __name__ == "__main__":

    pipeline_and_everything = Processing_and_logging()
    logger = pipeline_and_everything.setup_logging()
    image_folder = [
        r"dataset\raw\Diabetic retinopathy\No_DR",
        r"dataset\raw\Diabetic retinopathy\Mild",
        r"dataset\raw\Diabetic retinopathy\Moderate",
        r"dataset\raw\Diabetic retinopathy\Proliferate_DR",
        r"dataset\raw\Diabetic retinopathy\Severe"
    ]
    lable = ["No_DR","Mild", "Moderate", "Proliferate_DR", "Severe"]

    # feature extraction Example

    # extraction = Extraction_Functions(image_folder=image_folder[0], label=lable[0])
    # feature_extraction_using_GKCM(image_folder[0], lable[0])
    # extraction.feature_extraction_using_efficientnet()

    # Data Balancing Example

    balancing_techniques = Balancing_techniques(image_path=image_folder[4])
    balancing_techniques.cycle_gan_Balancing()


