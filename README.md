# Diabetic Retinopathy Image Processing Tool

> ⚠️ **WARNING:** This project is still in development and not ready for production use.


Simple tool to process retinal images for diabetic retinopathy analysis. Currently handles basic image loading and enhancement operations.

## Project Structure
diabetes-mellitus/
│ ├── services/
│ └── Preprocess_Images.py # Image processing operations
│ ├── main.py # Example usage
├── requirements.txt # Package dependencies
└── README.md

## Required Packages
- opencv-python (4.11)
- pandas (2.2.3)

## Files Overview

### Preprocess_Images.py
- Main class for image processing
- Features:
  - Load images
  - Convert to grayscale
  - Apply CLAHE enhancement

### main.py
- Example script showing how to use the ImageProcessor
- Demonstrates loading and processing a sample image

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Basic usage:


```python
from services.Preprocess_Images import ImageProcessor

# Initialize with image path
processor = ImageProcessor(image="path/to/image.png")

# Process image
processor.load_image()\
        .to_grayscale()\
        .enhance_gray()
```