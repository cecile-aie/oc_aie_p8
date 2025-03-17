from fastapi import APIRouter, UploadFile, File
import numpy as np
import time
from PIL import Image
import io
import os
from typing import List

from app.models.model_loader import model, CLASS_INFO, CLASS_COLORS
from app.utils.preprocessing import preprocess_image
from app.utils.postprocessing import encode_mask, encode_colored_mask

router = APIRouter()

# Dossier temporaire pour stocker les prédictions batch
DATA_DIR = "data"
BATCH_PREDICTIONS_DIR = os.path.join(DATA_DIR, "batch_predictions")

# S'assurer que le dossier existe
os.makedirs(BATCH_PREDICTIONS_DIR, exist_ok=True)

# Fonction de prédiction batch avec temps par image
def batch_predict(images: List[Image.Image]):
    """Effectue la prédiction sur plusieurs images et retourne les résultats indexés avec leur temps de calcul."""
    results = []
    total_start_time = time.time()  # Début du temps total d'inférence

    for idx, image in enumerate(images):
        image_array = preprocess_image(image)

        # Mesure du temps d'inférence individuel
        start_time = time.time()
        prediction = model.predict(image_array)[0]  # Prédiction (output shape: (256,256,8))
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000  # Temps en millisecondes

        mask = prediction.argmax(axis=-1).astype(np.uint8)

        # Sauvegarde des fichiers
        mask_path = os.path.join(BATCH_PREDICTIONS_DIR, f"predict_mask_{idx}.png")
        color_mask_path = os.path.join(BATCH_PREDICTIONS_DIR, f"colored_predict_mask_{idx}.png")

        Image.fromarray(mask).save(mask_path)  # Masque en niveaux de gris

        # Création du masque coloré
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in enumerate(CLASS_COLORS):
            color_mask[mask == class_idx] = color
        Image.fromarray(color_mask).save(color_mask_path)  # Masque coloré

        results.append({
            "index": idx,
            "elapsed_time_ms": elapsed_time_ms,  # Temps d'inférence individuel
            "grayscale_mask": encode_mask(mask),
            "colored_mask": encode_colored_mask(mask)
        })

    total_elapsed_time_ms = (time.time() - total_start_time) * 1000  # Temps total d'inférence

    return {
        "message": "Prédictions effectuées avec succès",
        "total_elapsed_time_ms": total_elapsed_time_ms,
        "legend": CLASS_INFO,
        "results": results
    }

# Route pour traiter plusieurs images
@router.post("/batch_predict")
async def batch_predict_uploaded(files: List[UploadFile] = File(...)):
    """Prédiction sur un lot d'images envoyées par l'utilisateur."""
    images = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        images.append(image)

    if not images:
        return {"error": "Aucune image reçue"}

    return batch_predict(images)
