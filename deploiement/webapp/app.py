from flask import Flask, render_template, request, jsonify, session
import requests
import base64
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import sys
import io

app = Flask(__name__)
app.secret_key = "super-secret-key"  # Nécessaire pour utiliser session

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


@app.route("/check_mask", methods=["POST"])
def check_mask():
    image_name = request.form["image_name"]
    image_path = os.path.join("static/uploads", secure_filename(image_name))
    mask_path = find_ground_truth(image_path)
    return jsonify({"mask_exists": bool(mask_path)})


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, gt_image = None, None, None
    show_mask_input = False  # Valeur par défaut

    if request.method == "POST":
        image_name = request.form.get("image_name")
        file = request.files.get("file")
        user_mask = request.files.get("mask")

        # 🧠 Cas où aucun nouveau fichier image n’est fourni
        if not file or not file.filename:
            filename = session.get("last_image_name")
            file_path = session.get("last_image_path")
        else:
            filename = secure_filename(file.filename)
            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            session["last_image_path"] = file_path
            session["last_image_name"] = filename

        image_name = filename
        gt_path = find_ground_truth(file_path)

        # ✅ Dès qu’une image est en mémoire, on affiche le champ masque
        show_mask_input = session.get("last_image_path") is not None

        try:
            image_file = open(file_path, "rb")
        except Exception as e:
            error = "Aucune image n’a été chargée ou elle a été perdue. Veuillez en sélectionner une."
            return render_template("index.html", result=None, error=error, gt_image=None,
                                   image_name=None, show_mask_input=show_mask_input)

        files = {"file": (filename, image_file, "image/jpeg")}

        if gt_path:
            files["gt_file"] = (os.path.basename(gt_path), open(gt_path, "rb"), "image/png")
        elif user_mask and user_mask.filename:
            user_mask_filename = secure_filename(user_mask.filename)
            mask_folder = "static/masks"
            os.makedirs(mask_folder, exist_ok=True)
            user_mask_path = os.path.join(mask_folder, user_mask_filename)
            user_mask.save(user_mask_path)
            files["gt_file"] = (user_mask_filename, open(user_mask_path, "rb"), "image/png")

        response = requests.post(API_URL, files=files)
        try:
            api_response = response.json()
        except Exception:
            api_response = {}

        if response.status_code == 200:
            result = api_response

            if "error" in api_response:
                if gt_path or (user_mask and user_mask.filename):
                    error = api_response["error"]
                show_mask_input = True  # ✅ En cas d’erreur masque, toujours laisser visible

            with open(file_path, "rb") as img_file:
                encoded_original_image = base64.b64encode(img_file.read()).decode()
            result["original_image"] = encoded_original_image

            gt_image = None
            if not error:
                final_mask_path = gt_path if gt_path else (user_mask_path if user_mask and user_mask.filename else None)
                if final_mask_path:
                    try:
                        gt_mask = np.array(Image.open(final_mask_path))
                        colored_gt_mask = apply_color_map(gt_mask, result["legend"])
                        buffered = io.BytesIO()
                        colored_gt_mask.save(buffered, format="PNG")
                        gt_image = base64.b64encode(buffered.getvalue()).decode()
                    except Exception as e:
                        print(f"[ERROR] Échec lors de l'application de la colormap : {e}", file=sys.stderr)
                        error = "Masque GT non exploitable pour l'affichage. Il est peut-être invalide."
                        show_mask_input = True  # ✅ Erreur d'affichage → on laisse la saisie possible

        else:
            error = api_response.get("error", "Erreur lors de la segmentation")
        # ✅ le champ masque reste visible
        show_mask_input = session.get("last_image_path") is not None

        return render_template("index.html", result=result, error=error, gt_image=gt_image,
                               image_name=image_name, show_mask_input=show_mask_input)

    return render_template("index.html", result=None, error=None, gt_image=None,
                           image_name=None, show_mask_input=False)



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
