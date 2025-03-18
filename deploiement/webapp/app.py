from flask import Flask, render_template, request
import requests
import base64
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import sys
import io

app = Flask(__name__)

API_URL = os.getenv("API_URL", "http://api:8000/predict")

def apply_color_map(mask, legend):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, class_info in legend.items():
        color_mask[mask == int(class_id)] = class_info["color"]
    return Image.fromarray(color_mask)

def is_valid_mask(mask_path):
    try:
        mask = np.array(Image.open(mask_path))
        unique_values = np.unique(mask)
        print(f"[DEBUG] Unique values in ground truth: {unique_values}", file=sys.stderr)
        return any(val in range(8) for val in unique_values)
    except Exception as e:
        print(f"[ERROR] Unable to load mask: {e}", file=sys.stderr)
        return False

def find_ground_truth(image_path):
    image_filename = os.path.basename(image_path)
    name_base = "_".join(image_filename.split("_")[:3])
    gt_filename = f"{name_base}_gtFine_labelTrainIds.png"
    gt_root = "/app/data/gtFine"

    if os.path.exists(gt_root):
        for root, _, files in os.walk(gt_root):
            if gt_filename in files:
                gt_path = os.path.join(root, gt_filename)
                if is_valid_mask(gt_path):
                    return gt_path
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    result, error, gt_image = None, None, None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            files = {"file": (filename, open(file_path, "rb"), file.mimetype)}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                gt_path = find_ground_truth(file_path)
                if gt_path:
                    gt_mask = np.array(Image.open(gt_path))
                    colored_gt_mask = apply_color_map(gt_mask, result["legend"])

                    buffered = io.BytesIO()
                    colored_gt_mask.save(buffered, format="PNG")
                    gt_image = base64.b64encode(buffered.getvalue()).decode()
                else:
                    gt_image = None

                with open(file_path, "rb") as img_file:
                    encoded_original_image = base64.b64encode(img_file.read()).decode()
                result["original_image"] = encoded_original_image
            else:
                error = "Erreur lors de la segmentation"

            return render_template("index.html", result=result, error=error, gt_image=gt_image)

    return render_template("index.html", result=None, error=None, gt_image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
