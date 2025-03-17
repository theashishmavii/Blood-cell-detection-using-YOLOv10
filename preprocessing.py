import cv2
import numpy as np
from PIL import Image
from albumentations import Compose, RandomCrop, HorizontalFlip, Rotate, Normalize
from albumentations.pytorch import ToTensorV2

# Define the preprocessing pipeline using Albumentations
def get_transform():
    return Compose([
        RandomCrop(width=640, height=640, p=1.0),  # Random crop to a fixed size
        HorizontalFlip(p=0.5),  # Horizontal flip with a 50% probability
        Rotate(limit=40, p=0.5),  # Random rotation between -40 to +40 degrees
        Normalize(mean=[0, 0, 0], std=[1, 1, 1], p=1.0),  # Normalize the image
        ToTensorV2()  # Convert to PyTorch tensor
    ])

def preprocess_image(image_path):
    # Load image using OpenCV or PIL
    img = Image.open(image_path)
    
    # Apply augmentation and transformation
    transform = get_transform()
    augmented = transform(image=np.array(img))  # Convert PIL to numpy for processing
    return augmented['image']

# Test preprocessing
if __name__ == "__main__":
    preprocessed_image = preprocess_image('path_to_image.jpg')
    print("Preprocessed image shape:", preprocessed_image.shape)
