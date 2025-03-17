from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import time
from PIL import Image
import io
import os

from app.models.model_loader import model, CLASS_INFO, CLASS_COLORS
from app.utils.preprocessing import preprocess_image
from app.utils.postprocessing import encode_mask, encode_colored_mask

router = APIRouter()

# Dossier de stockage temporaire des prédictions
DATA_DIR = "data"
PREDICT_MASK_PATH = os.path.join(DATA_DIR, "predict_mask.png")
COLORED_MASK_PATH = os.path.join(DATA_DIR, "colored_predict_mask.png")

# Fonction de prédiction
def predict(image: Image.Image):
    """Effectue la prédiction sur une image donnée."""
    image_array = preprocess_image(image)

    start_time = time.time()
    prediction = model.predict(image_array)[0]  # Prédiction (output shape: (256,256,8))
    end_time = time.time()

    elapsed_time_ms = (end_time - start_time) * 1000  # Temps en ms

    # Conversion en masque de classes (0-8)
    mask = prediction.argmax(axis=-1).astype(np.uint8)

    # Sauvegarde des résultats
    mask_image = Image.fromarray(mask)
    mask_image.save(PREDICT_MASK_PATH)  # Masque en niveaux de gris

    # Création du masque coloré
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == class_idx] = color

    color_mask_image = Image.fromarray(color_mask)
    color_mask_image.save(COLORED_MASK_PATH)  # Masque coloré

    return mask, color_mask, elapsed_time_ms

# Route pour uploader une image et obtenir la prédiction
@router.post("/predict")
async def predict_uploaded(file: UploadFile = File(...)):
    """Prédiction sur une image uploadée par l'utilisateur."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    mask, color_mask, elapsed_time_ms = predict(image)

    return {
        "message": "Prédiction effectuée avec succès",
        "elapsed_time_ms": elapsed_time_ms,
        "legend": CLASS_INFO,
        "grayscale_mask": encode_mask(mask),
        "colored_mask": encode_colored_mask(mask),
    }

@router.get("/predict/mask")
async def get_predicted_mask():
    """Retourne le masque prédit en niveaux de gris."""
    mask_path = os.path.join("data", "predict_mask.png")
    return FileResponse(mask_path, media_type="image/png")

@router.get("/predict/colored_mask")
async def get_colored_mask():
    """Retourne le masque coloré."""
    color_mask_path = os.path.join("data", "colored_predict_mask.png")
    return FileResponse(color_mask_path, media_type="image/png")