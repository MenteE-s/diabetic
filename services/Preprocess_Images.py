import cv2


class ImageProcessor:
    # Constructor, the image url/path shouuld be passed after initialization
    def __init__(self, image):
        self.image = image
        self.gray = None
        self.enhanced = None

    # this will load the image, read that using cv2.imread function, and if non
    # means the iamge was not passed during initilization then will through an error
    # of image nor found
    def load_image(self):
        """Load image from path"""
        self.image = cv2.imread(self.image)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at: {self.image}")
        return self
    
    # this will convert the image into gray using cv2
    # and again checing the image for if its not availble. 
    # you can remove this check also.
    def to_grayscale(self):
        """Convert image to grayscale"""
        if self.image is None:
            raise ValueError("No image loaded")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self
    
    # this function is enhamcing the image by applying CLAHE
    # CLEHE stand for Contrast Limited Adoptive Histogram Equiluization
    # what this functiuon actually does, is that, it divide iamfe to tiles of your own size
    # it may be 8*8 or may be 16*16 then it add enhancement to each tile
    # separatly, and you can also limit the contrast
    def enhance_gray(self):
        """Apply CLAHE enhancement to grayscale image"""
        if self.gray is None:
            raise ValueError("Convert to grayscale first")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.enhanced = clahe.apply(self.gray)
        return self

    