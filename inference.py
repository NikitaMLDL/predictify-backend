from fastapi import FastAPI, HTTPException, Request, Form
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import time


app = FastAPI(title="Iris Classifier Inference")

instrumentator = Instrumentator()

PREDICTION_LATENCY = Gauge(
    "model_prediction_latency_seconds",
    "Time spent performing model inference (seconds)"
)


instrumentator.instrument(app).expose(app, endpoint="/metrics")
templates = Jinja2Templates(directory="templates")


class Model:
    def __init__(self):
        self.model = None

    def load(self, experiment_name: str, alias: str):
        self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{experiment_name}@{alias}")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)


iris_model = Model()
iris_model.load("IrisClassifier", "staging")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/", response_class=HTMLResponse)
def predict_form(
    request: Request,
    features: IrisFeatures = Form(...)
):
    try:
        df = pd.DataFrame([{
            "SepalLengthCm": features.sepal_length,
            "SepalWidthCm": features.sepal_width,
            "PetalLengthCm": features.petal_length,
            "PetalWidthCm": features.petal_width
        }])
        df = df.drop_duplicates()

        df['sepal_ratio'] = df['SepalLengthCm'] / df['SepalWidthCm']
        df['petal_ratio'] = df['PetalLengthCm'] / df['PetalWidthCm']

        start_time = time.time()
        prediction = iris_model.predict(df)[0]
        elapsed = time.time() - start_time

        PREDICTION_LATENCY.set(elapsed)
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/")
async def receive_webhook(req: Request):
    payload = await req.json()
    print("Before load:", id(iris_model.model))
    if payload.get("entity") == "model_version_alias" and payload.get("action") == "created":
        data = payload.get("data", {})
        if data.get("alias") == "prod":
            iris_model.load(data.get("name"), "prod")
            print("After load:", id(iris_model.model))
            print(f"Model '{data.get('name')}' loaded with alias 'prod'")
    return {"status": "success"}

# @app.post("/webhook/")
# async def receive_webhook(req: Request):
#     data = await req.json()
#     print("Webhook received:", data)
#     return {"status": "success"}