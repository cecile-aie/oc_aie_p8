from fastapi import FastAPI
import uvicorn
from fastapi.openapi.utils import get_openapi
from app.routes import predict, batch_predict, overview

# Initialiser FastAPI
app = FastAPI()

# Inclure les routes
app.include_router(predict.router, prefix="/api")
app.include_router(batch_predict.router, prefix="/api")
app.include_router(overview.router, prefix="/api")

# 🎨 Personnalisation de Swagger UI pour afficher les images
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Segmentation API",
        version="1.0.0",
        description="API pour la segmentation sémantique des images.",
        routes=app.routes,
    )

    # Modifier la documentation de `/api/overview` pour afficher les liens vers les images
    if "/api/overview" in openapi_schema["paths"]:
        openapi_schema["paths"]["/api/overview"]["get"]["responses"]["200"] = {
            "description": "Liens vers l'image d'exemple et le masque coloré",
            "content": {
                "application/json": {
                    "example": {
                        "example_input_url": "<a href='/api/example_image' target='_blank'>Voir l'image originale</a>",
                        "example_prediction_url": "<a href='/api/example_mask_colored' target='_blank'>Voir le masque coloré</a>"
                    }
                }
            }
        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Appliquer la personnalisation de Swagger UI
app.openapi = custom_openapi

# Lancer l'application avec Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
