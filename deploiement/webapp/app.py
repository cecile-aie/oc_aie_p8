from flask import Flask, render_template, request
import requests
import base64
import io
import os
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 🔹 Détection de l'environnement : local ou déploiement (PythonAnywhere)
if "PYTHONANYWHERE_DOMAIN" in os.environ:
    API_URL = "http://54.173.58.174:8000/predict"  # 🔥 URL de l'API FastAPI sur AWS (remplace par ton IP EC2)
else:
    API_URL = os.getenv("API_URL", "http://api:8000/predict")  # 🔥 API locale pour test en local

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "Aucun fichier sélectionné"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Fichier vide"
            else:
                try:
                    # 🔹 Envoyer l'image à l'API FastAPI (locale ou AWS)
                    files = {"file": (secure_filename(file.filename), file.stream, file.mimetype)}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                    else:
                        error = "Erreur lors de la segmentation"

                    # 🔹 Encodage de l'image originale pour affichage
                    image = Image.open(file.stream).convert("RGB")  # Convertir en RGB (évite les problèmes transparence)
                    image = image.resize((256, 256))  # Redimensionner l'image originale

                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format="PNG")  # Sauvegarde en mémoire
                    encoded_original_image = base64.b64encode(image_bytes.getvalue()).decode()  # Encodage Base64

                    if result:
                        result["original_image"] = encoded_original_image  # Ajout de l'image originale
                except Exception as e:
                    error = f"Erreur interne : {str(e)}"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
