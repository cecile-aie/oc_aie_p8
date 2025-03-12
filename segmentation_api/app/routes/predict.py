from fastapi import APIRouter, File, UploadFile, HTTPException
import time
import numpy as np
import io
from PIL import Image
from app.utils.preprocessing import preprocess_image
from ..utils.postprocessing import encode_mask, encode_colored_mask
from app.models.model_loader import model, CLASS_INFO

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_data = preprocess_image(image)
        
        start_time = time.time()
        mask = model.predict(input_data)[0]
        inference_time = (time.time() - start_time) * 1000  # en millisecondes
        
        encoded_mask = encode_mask(mask)
        
        return {
            "inference_time": inference_time,
            "class_info": CLASS_INFO,
            "prediction": encoded_mask
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
