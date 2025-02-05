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



class Pipeline_And_Everything:
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


def feature_extraction_using_GKCM(image_folder, lable):
    features_list = []
    labels = []

    logger.info(f"Starting feature extraction from images in: {image_folder}")

    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, filename)

            logger.info(f"Processing file: {filename}")
            image = pipeline_and_everything.start_processing(image_path=image_path)
            # cv2.imshow('Image', image)

            if image is not None:
                # Extract GLCM and histogram features
                glcm_features = Applying_GLCM.extract_glcm_features(image=image)
                histogram_features = Applying_GLCM.extract_grayscale_histogram(image=image)

                features = np.hstack([glcm_features, histogram_features])
                features_list.append(features)

                labels.append(lable)
                logger.info(f"Features extracted for {filename} | labled as {lable}")
                logger.info(f"Features: {features_list}  \n\n")

    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    df["label"] = labels

    # Normalize features (excluding labels)
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    logger.info("Feature extraction and normalization completed successfully.")
    
    # Optionally, you can save the DataFrame here if everything looks good
    # df.to_csv("processed_features.csv", index=False)

    # Show the first few rows to verify
    logger.info(f"Processed data preview:\n{df.head()}")
    



if __name__ == "__main__":

    pipeline_and_everything = Pipeline_And_Everything()
    logger = pipeline_and_everything.setup_logging()
    image_folder = [
        r"dataset\raw\Diabetic retinopathy\No_DR",
        r"dataset\raw\Diabetic retinopathy\Mild",
        r"dataset\raw\Diabetic retinopathy\Moderate",
        r"dataset\raw\Diabetic retinopathy\Proliferate_DR",
        r"dataset\raw\Diabetic retinopathy\Severe"
    ]
    lable = ["No_DR","Mild", "Moderate", "Proliferate_DR", "Severe"]

    feature_extraction_using_GKCM(image_folder[0], lable[0])
