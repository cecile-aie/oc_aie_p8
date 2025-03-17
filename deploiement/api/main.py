from fastapi import FastAPI
from app.routes import predict, batch_predict
from app.models.model_loader import model
import numpy as np

app = FastAPI()

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