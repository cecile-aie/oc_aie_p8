from fastapi import APIRouter, File, UploadFile, HTTPException
import time
import numpy as np
import io
import os
from PIL import Image
import base64
from fastapi.responses import FileResponse
from app.utils.preprocessing import preprocess_image
from ..utils.postprocessing import encode_mask, encode_colored_mask
from app.models.model_loader import model, CLASS_INFO

router = APIRouter()

@router.post("/predict", summary="Effectue une prédiction sur une image",
             response_description="Masque de segmentation encodé en base64 et images accessibles via URL")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_data = preprocess_image(image)
        
        start_time = time.time()
        mask = model.predict(input_data)[0]  # Prédiction du modèle
        inference_time = (time.time() - start_time) * 1000  # en millisecondes

        # 🔥 Vérifier les classes détectées
        unique_classes = np.unique(mask.argmax(axis=-1))
        print(f"Classes détectées dans la prédiction : {unique_classes}")

        # 🔥 Génération du masque en niveaux de gris
        mask_2d = mask.argmax(axis=-1).astype(np.uint8)
        mask_image = Image.fromarray(mask_2d)
        mask_path = "data/prediction_mask.png"
        mask_image.save(mask_path)

        # 🔥 Génération et sauvegarde du masque coloré via `encode_colored_mask()`
        encoded_colored = encode_colored_mask(mask_2d)  # Encode en base64
        mask_colored_path = "data/prediction_mask_colored.png"

        # 🔥 Sauvegarde du masque coloré depuis `encode_colored_mask()`
        with open(mask_colored_path, "wb") as f:
            f.write(base64.b64decode(encoded_colored))

        # Encodage base64 du masque en niveaux de gris
        encoded_mask = encode_mask(mask)

        return {
            "inference_time": inference_time,
            "class_info": CLASS_INFO,
            "prediction": encoded_mask,  # Version base64 (gris)
            "mask_url": "/api/prediction_mask",  # URL du masque en niveaux de gris
            "mask_colored_url": "/api/prediction_mask_colored"  # URL du masque coloré
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction_mask")
async def get_prediction_mask():
    """Retourne le masque de segmentation prédit en niveaux de gris"""
    mask_path = "data/prediction_mask.png"
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Aucun masque généré")
    return FileResponse(mask_path, media_type="image/png")

@router.get("/prediction_mask_colored")
async def get_prediction_mask_colored():
    """Retourne le masque de segmentation prédit en couleur"""
    mask_colored_path = "data/prediction_mask_colored.png"
    if not os.path.exists(mask_colored_path):
        raise HTTPException(status_code=404, detail="Aucun masque coloré généré")
    return FileResponse(mask_colored_path, media_type="image/png")
