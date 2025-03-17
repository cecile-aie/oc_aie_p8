from fastapi import FastAPI
from app.routes import predict, batch_predict

app = FastAPI()

app.include_router(predict.router)
app.include_router(batch_predict.router)
