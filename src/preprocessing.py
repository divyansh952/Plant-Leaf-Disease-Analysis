import cv2
import numpy as np


def load_image(image_path):
    """
    Load an image from the given path.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    return image


def resize_image(image, size=(224, 224)):
    """
    Resize the image to the required size.
    """
    return cv2.resize(image, size)


def normalize_image(image):
    """
    Normalize pixel values from 0-255 to 0-1.
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path, size=(224, 224)):
    """
    Complete preprocessing pipeline.
    """
    image = load_image(image_path)
    image = resize_image(image, size)
    image = normalize_image(image)

    return image
