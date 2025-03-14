from flask import Flask, render_template, request
import requests
import base64
import io
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="Aucun fichier sélectionné")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Fichier vide")

        # Lire l'image et envoyer à l'API
        files = {"file": (secure_filename(file.filename), file.stream, file.mimetype)}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
        else:
            return render_template("index.html", error="Erreur lors de la segmentation")

        # 🔹 Encodage de l'image originale pour l'afficher dans le template
        image = Image.open(file.stream).convert("RGB")  # Convertir en RGB (évite problèmes transparence)
        image = image.resize((256, 256))  # Redimensionner l'image originale

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")  # Sauvegarde en mémoire
        encoded_original_image = base64.b64encode(image_bytes.getvalue()).decode()  # Encodage Base64

        result["original_image"] = encoded_original_image  # Ajout au résultat

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
