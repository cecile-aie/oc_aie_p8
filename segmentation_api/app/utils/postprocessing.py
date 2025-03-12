import numpy as np
import base64
import io
from PIL import Image
from app.models.model_loader import CLASS_COLORS

def encode_image(image_bytes):
    """Encode une image en base64 (format JPEG ou PNG)"""
    return base64.b64encode(image_bytes).decode()

def encode_mask(mask: np.ndarray):
    """Encode un masque multi-classes en base64 pour un retour JSON"""
    # Vérifier si le masque est un tenseur (multi-classes)
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask_2d = mask.argmax(axis=-1).astype(np.uint8)  # Convertir en indices de classes
    else:
        mask_2d = mask.astype(np.uint8)  # Si c'est déjà une image 2D, on garde tel quel

    # Convertir en image
    mask_image = Image.fromarray(mask_2d)

    # Sauvegarde en mémoire
    img_io = io.BytesIO()
    mask_image.save(img_io, format="PNG")
    img_io.seek(0)

    # Encodage base64
    return base64.b64encode(img_io.read()).decode("utf-8")

def encode_colored_mask(mask: np.ndarray):
    """Transforme un masque de segmentation en image couleur (RGB)"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == class_idx] = color  # Appliquer la couleur

    # Convertir en image
    mask_image = Image.fromarray(color_mask)

    # Sauvegarde en mémoire
    img_io = io.BytesIO()
    mask_image.save(img_io, format="PNG")
    img_io.seek(0)

    # Encodage base64
    return base64.b64encode(img_io.read()).decode()
