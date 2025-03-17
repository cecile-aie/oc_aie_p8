import numpy as np
from PIL import Image

IMG_SIZE = (256, 256)

def preprocess_image(image: Image.Image):
    """Convertit une image PIL en tenseur normalisé pour le modèle"""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter batch dim
    return image_array
