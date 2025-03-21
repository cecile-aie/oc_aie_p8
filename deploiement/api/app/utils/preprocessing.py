import numpy as np
from PIL import Image

IMG_SIZE = (256, 256)

def validate_image(image: Image.Image):
    """Vérifie que l'image est bien RGB avec 3 canaux."""
    if image.mode != 'RGB':
        raise ValueError(f"L'image doit être en mode RGB. Mode actuel : {image.mode}")

    image_array = np.array(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"L'image doit avoir exactement 3 dimensions (H, W, C=3). Dimensions trouvées : {image_array.shape}")

def preprocess_image(image: Image.Image):
    """Convertit une image PIL en tenseur normalisé pour le modèle"""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter batch dim
    return image_array
