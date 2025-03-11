import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))  # Adapter à l'entrée du modèle
    image = np.array(image) / 255.0  # Normalisation
    return np.expand_dims(image, axis=0)  # Ajouter la dimension batch
