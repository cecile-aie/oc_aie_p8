from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.routes import predict, batch_predict
from app.models.model_loader import model
import numpy as np
from PIL import UnidentifiedImageError

app = FastAPI()

# Lecture de la version au démarrage
try:
    with open("/app/VERSION.txt", "r") as f:
        APP_VERSION = f.read().strip()
except FileNotFoundError:
    APP_VERSION = "Version inconnue"
    
@app.get("/version")
def get_version():
    return {"version": APP_VERSION}

# Inclusion des routes
app.include_router(predict.router)
app.include_router(batch_predict.router)

@app.on_event("startup")
def warm_up_model():
    """Effectue une prédiction au démarrage pour charger le modèle en mémoire."""
    print("⏳ Warm-up du modèle en cours...")
    try:
        dummy_input = np.zeros((1, 256, 256, 3))  # Conforme aux dimensions d'entrée pour le modèle
        _ = model.predict(dummy_input)  # Première prédiction
        print("✅ Modèle chargé en mémoire (warm-up terminé)")
    except Exception as e:
        print(f"❌ Erreur lors du warm-up du modèle : {e}")

# Gestion personnalisée des exceptions HTTP
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Gestion des erreurs liées aux images invalides
@app.exception_handler(UnidentifiedImageError)
async def image_exception_handler(request: Request, exc: UnidentifiedImageError):
    return JSONResponse(
        status_code=400,
        content={"error": "L'image fournie est invalide ou corrompue. Veuillez fournir un fichier image RGB valide."}
    )

# Gestion globale des exceptions inattendues
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Une erreur inattendue est survenue."}
    )
# Remontée de l'erreur de validate_image
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}  # <-- message explicite depuis validate_image
    )
