from fastapi import APIRouter, File, UploadFile
from typing import List
import time
import io
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
from app.utils.preprocessing import preprocess_image
from app.utils.postprocessing import encode_mask
from app.models.model_loader import model, CLASS_INFO

router = APIRouter()

@router.post("/batch_predict")
async def batch_predict(files: List[UploadFile]):
    results = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_data = preprocess_image(image)
        start_time = time.time()
        mask = model.predict(input_data)[0]
        inference_time = (time.time() - start_time) * 1000  # en ms
        encoded_mask = encode_mask(mask)
        results.append({
            "inference_time": inference_time,
            "class_info": CLASS_INFO,
            "prediction": encoded_mask
        })
    return results
