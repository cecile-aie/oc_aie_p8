from fastapi import FastAPI
import uvicorn
from app.routes import predict, batch_predict, overview

# Initialiser FastAPI
app = FastAPI()

# Inclure les routes
app.include_router(predict.router, prefix="/api")
app.include_router(batch_predict.router, prefix="/api")
app.include_router(overview.router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
