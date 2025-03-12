from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import time
import io
import os
import base64
import numpy as np
from PIL import Image
from fastapi.responses import FileResponse
from app.utils.preprocessing import preprocess_image
from app.utils.postprocessing import encode_colored_mask
from app.models.model_loader import model, CLASS_INFO

router = APIRouter()

@router.post("/batch_predict")
async def batch_predict(files: List[UploadFile]):
    """Effectue une prédiction en batch sur plusieurs images et retourne les URLs des masques générés"""
    
    start_time = time.time()
    
    if not files:
        raise HTTPException(status_code=400, detail="Aucune image reçue.")

    results = []

    for idx, file in enumerate(files):
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
            input_data = preprocess_image(image)

            # Prédiction
            mask = model.predict(input_data)[0]
            inference_time = (time.time() - start_time) * 1000  # ms

            # 🔥 Génération du masque en niveaux de gris
            mask_2d = mask.argmax(axis=-1).astype(np.uint8)
            mask_image = Image.fromarray(mask_2d)
            mask_path = f"data/prediction_mask_{idx}.png"
            mask_image.save(mask_path)

            # 🔥 Génération et sauvegarde du masque coloré via `encode_colored_mask()`
            encoded_colored = encode_colored_mask(mask_2d)
            mask_colored_path = f"data/prediction_mask_colored_{idx}.png"
            with open(mask_colored_path, "wb") as f:
                f.write(base64.b64decode(encoded_colored))

            results.append({
                "mask_url": f"/api/prediction_mask_{idx}",  # URL du masque en niveaux de gris
                "mask_colored_url": f"/api/prediction_mask_colored_{idx}"  # URL du masque coloré
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "batch_size": len(files),
        "inference_time": time.time() - start_time,
        "results": results
    }

@router.get("/prediction_mask_{idx:int}")
async def get_prediction_mask(idx: int):
    """Retourne le masque de segmentation en niveaux de gris pour une image spécifique"""
    mask_path = f"data/prediction_mask_{idx}.png"
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Aucun masque généré")
    return FileResponse(mask_path, media_type="image/png")

@router.get("/prediction_mask_colored_{idx:int}")
async def get_prediction_mask_colored(idx: int):
    """Retourne le masque coloré pour une image spécifique"""
    mask_colored_path = f"data/prediction_mask_colored_{idx}.png"
    if not os.path.exists(mask_colored_path):
        raise HTTPException(status_code=404, detail="Aucun masque coloré généré")
    return FileResponse(mask_colored_path, media_type="image/png")
