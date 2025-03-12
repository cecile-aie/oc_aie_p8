from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
import os
import numpy as np
from PIL import Image
from app.models.model_loader import CLASS_COLORS

router = APIRouter()

@router.get("/overview", response_class=HTMLResponse)
async def overview():
    """Affiche directement les images dans Swagger UI"""
    example_image_path = "data/example_image.png"
    example_mask_path = "data/example_mask.png"
    colored_mask_path = "data/example_mask_colored.png"  # Nouveau fichier colorisé

    # Vérifier si les fichiers existent
    if not os.path.exists(example_image_path) or not os.path.exists(example_mask_path):
        return "<h3>Erreur : Les fichiers d'exemple ne sont pas trouvés.</h3>"

    # Convertir le masque en couleur si le fichier colorisé n'existe pas encore
    if not os.path.exists(colored_mask_path):
        mask_array = np.array(Image.open(example_mask_path))  # Charger le masque brut
        color_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

        # Appliquer les couleurs à chaque classe
        for class_idx, color in enumerate(CLASS_COLORS):
            color_mask[mask_array == class_idx] = color

        # Sauvegarder le masque coloré
        color_image = Image.fromarray(color_mask)
        color_image.save(colored_mask_path)

    # Retourner la réponse HTML avec les images
    return f"""
    <html>
        <body>
            <h2>Exemple d'image originale</h2>
            <img src="/api/example_image" width="300"/>
            
            <h2>Exemple de masque coloré</h2>
            <img src="/api/example_mask_colored" width="300"/>
        </body>
    </html>
    """

@router.get("/example_image")
async def get_example_image():
    """Retourne l'image d'exemple en tant que fichier"""
    return FileResponse("data/example_image.png", media_type="image/png")

@router.get("/example_mask_colored")
async def get_colored_mask():
    """Retourne le masque coloré en tant que fichier"""
    return FileResponse("data/example_mask_colored.png", media_type="image/png")
