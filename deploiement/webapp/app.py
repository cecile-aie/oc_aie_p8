from flask import Flask, render_template, request
import requests
import base64
import io
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import sys

app = Flask(__name__)

# 🔹 Détection automatique du bon environnement (Docker Compose ou test en local)
API_URL = os.getenv("API_URL", "http://api:8000/predict")  # Utilisation de l’alias réseau Docker

def is_valid_mask(mask_path):
    """
    Vérifie si l'image contient des valeurs entre 0 et 7.
    """
    try:
        mask = np.array(Image.open(mask_path))
        unique_values = np.unique(mask)
        print(f"[DEBUG] Unique values in ground truth: {unique_values}", file=sys.stderr)
        return any(val in range(8) for val in unique_values)
    except Exception as e:
        print(f"[ERROR] Unable to load mask: {e}", file=sys.stderr)
        return False

def find_ground_truth(image_path):
    """
    Recherche le fichier ground truth dans /app/data/gtFine/.
    """
    image_filename = os.path.basename(image_path)
    name_base = "_".join(image_filename.split("_")[:3])
    gt_filename = f"{name_base}_gtFine_labelTrainIds.png"
    gt_root = "/app/data/gtFine"
    
    print(f"[DEBUG] Searching in mounted gtFine: {gt_root}", file=sys.stderr)
    print(f"[DEBUG] Expected filename: {gt_filename}", file=sys.stderr)
    
    if os.path.exists(gt_root):
        for root, _, files in os.walk(gt_root):
            print(f"[DEBUG] Searching in: {root}", file=sys.stderr)
            print(f"[DEBUG] Files in directory: {files}", file=sys.stderr)
            if gt_filename in files:
                gt_path = os.path.join(root, gt_filename)
                print(f"[SUCCESS] Found ground truth file: {gt_path}", file=sys.stderr)
                if is_valid_mask(gt_path):
                    return gt_path
                else:
                    print("[ERROR] Ground truth mask contains no valid annotations!", file=sys.stderr)
                    return None
    else:
        print("[ERROR] gtFine directory not found in /app/data!", file=sys.stderr)
    
    print("[ERROR] Ground truth file not found!", file=sys.stderr)
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    gt_image = None  # Ajout de la variable pour stocker le masque ground truth
    gt_valid = True

    if request.method == "POST":
        if "file" not in request.files:
            error = "Aucun fichier sélectionné"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Fichier vide"
            else:
                try:
                    # 🔹 Sauvegarde temporaire de l'image
                    upload_folder = "static/uploads"
                    os.makedirs(upload_folder, exist_ok=True)
                    file_path = os.path.join(upload_folder, secure_filename(file.filename))
                    file.save(file_path)
                    
                    # 🔹 Recherche du ground truth
                    gt_path = find_ground_truth(file_path)
                    if gt_path:
                        with open(gt_path, "rb") as gt_file:
                            gt_bytes = gt_file.read()
                        encoded_gt_image = base64.b64encode(gt_bytes).decode()
                        gt_image = encoded_gt_image
                    else:
                        gt_valid = False
                        print("[ERROR] No valid ground truth image found!", file=sys.stderr)

                    # 🔹 Envoyer l'image à l'API FastAPI (locale ou AWS)
                    files = {"file": (secure_filename(file.filename), open(file_path, "rb"), file.mimetype)}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                    else:
                        error = "Erreur lors de la segmentation"

                    # 🔹 Encodage de l'image originale pour affichage
                    with open(file_path, "rb") as img_file:
                        image_bytes = img_file.read()
                    encoded_original_image = base64.b64encode(image_bytes).decode()
                    
                    if result:
                        result["original_image"] = encoded_original_image  # Ajout de l'image originale
                except Exception as e:
                    error = f"Erreur interne : {str(e)}"

    return render_template("index.html", result=result, error=error, gt_image=gt_image, gt_valid=gt_valid)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
