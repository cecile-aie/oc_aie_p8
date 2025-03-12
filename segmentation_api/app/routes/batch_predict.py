from fastapi import APIRouter, File, UploadFile
from typing import List
import time
import io
import numpy as np
from PIL import Image
from app.utils.preprocessing import preprocess_image
from app.utils.postprocessing import encode_mask
from app.models.model_loader import model, CLASS_INFO

router = APIRouter()

@router.post("/batch_predict")
async def batch_predict(files: List[UploadFile]):
    """Effectue une prédiction en batch sur plusieurs images"""
    
    start_time = time.time()
    
    images = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        images.append(preprocess_image(image))  # Prétraitement

    if not images:
        return {"error": "Aucune image reçue."}

    # Convertir la liste en un tableau numpy (batching)
    batch_input = np.vstack(images)  

    # Prédiction en batch (optimisé pour TensorFlow)
    masks = model.predict(batch_input)

    # Temps total d'inférence
    inference_time = (time.time() - start_time) * 1000  # ms

    # Post-traitement des masques
    encoded_masks = [encode_mask(mask) for mask in masks]

    results = []
    for encoded_mask in encoded_masks:
        results.append({
            "class_info": CLASS_INFO,
            "prediction": encoded_mask
        })

    return {
        "batch_size": len(files),
        "inference_time": inference_time,  # Temps d'inférence total
        "results": results
    }
