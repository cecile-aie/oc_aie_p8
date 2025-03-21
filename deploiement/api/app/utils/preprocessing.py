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

def validate_gt_mask(mask: np.ndarray, num_classes: int):
    """Valide que le masque est mono-canal et contient uniquement des classes valides."""
    if mask.ndim != 2:
        raise ValueError(f"Le masque GT doit être une image à un seul canal. Dimensions trouvées : {mask.shape}")

    unique_values = np.unique(mask)
    if not np.all((unique_values >= 0) & (unique_values < num_classes)):
        raise ValueError(f"Le masque GT contient des classes invalides : {unique_values}")

def preprocess_image(image: Image.Image):
    """Convertit une image PIL en tenseur normalisé pour le modèle"""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter batch dim
    return image_array
