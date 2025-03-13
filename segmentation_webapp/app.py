from flask import Flask, request, render_template, send_file, jsonify
import requests
import os
import zipfile
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

# Liens vers l'API déployée
# API_URL_SINGLE = "https://ton-api-fastapi.com/predict"
# API_URL_BATCH = "https://ton-api-fastapi.com/predict-batch"
# Liens vers l'API locale dans le docker
API_URL_SINGLE = "http://localhost:8000/predict"  
API_URL_BATCH = "http://localhost:8000/batch_predict"



UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/single", methods=["POST"])
def single_image():
    file = request.files["file"]
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        files = {"file": open(file_path, "rb")}
        response = requests.post(API_URL_SINGLE, files=files)
        result = response.json()
        
        mask_url = result.get("mask_url")  # L'URL du masque retournée par l'API
        metrics = result.get("metrics", {})
        
        return render_template("single_result.html", original=file.filename, mask_url=mask_url, metrics=metrics)
    return "Erreur lors du téléchargement"

@app.route("/download/<format>/<filename>")
def download_image(format, filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if format == "json":
        with open(file_path.replace(".png", ".json"), "r") as f:
            return jsonify(json.load(f))
    return send_file(file_path, as_attachment=True)

@app.route("/batch", methods=["POST"])
def batch_images():
    files = request.files.getlist("files")
    uploaded_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        uploaded_files.append((file.filename, open(file_path, "rb")))
    
    response = requests.post(API_URL_BATCH, files={"files": [f[1] for f in uploaded_files]})
    results = response.json()
    
    # Créer un fichier ZIP des résultats
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for result in results.get("images", []):
            mask_url = result.get("mask_url")
            metrics = result.get("metrics", {})
            
            # Télécharger le masque
            mask_response = requests.get(mask_url)
            if mask_response.status_code == 200:
                mask_filename = os.path.join(RESULT_FOLDER, result.get("filename"))
                with open(mask_filename, "wb") as f:
                    f.write(mask_response.content)
                zip_file.write(mask_filename, arcname=result.get("filename"))
                
                # Sauvegarder les métriques
                json_filename = mask_filename.replace(".png", ".json")
                with open(json_filename, "w") as f:
                    json.dump(metrics, f)
                zip_file.write(json_filename, arcname=os.path.basename(json_filename))
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name="segmentation_results.zip", mimetype="application/zip")

if __name__ == "__main__":
    app.run(debug=True)
