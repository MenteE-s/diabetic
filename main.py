import cv2
import logging
from datetime import datetime
from pathlib import Path
from services.Preprocess_Images import ImageProcessor

# Setup logging
def setup_logging():
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

# Initialize logging
logger = setup_logging()

try:
    logger.info("Starting image processing")
    
    image_path = r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\raw\Diabetic retinopathy\Moderate\0c7e82daf5a0.png"
    logger.info(f"Processing image: {image_path}")
    
    processor = ImageProcessor(image=image_path)
    logger.info("ImageProcessor initialized")
    
    processor.load_image()
    logger.info("Image loaded successfully")
    
    processor.to_grayscale()
    logger.info("Converted to grayscale")
    
    processor.enhance_gray()
    logger.info("Applied CLAHE enhancement")
    
    # Show results
    cv2.imshow("Original", processor.image)
    cv2.imshow("Grayscale", processor.gray)
    cv2.imshow("Enhanced", processor.enhanced)
    logger.info("Displayed images")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    logger.info("Processing completed successfully")

except Exception as e:
    logger.error(f"Error during processing: {str(e)}", exc_info=True)