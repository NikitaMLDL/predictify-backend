from fastapi import FastAPI, HTTPException, Request, Form
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd


app = FastAPI(title="Iris Classifier Inference")

instrumentator = Instrumentator()

PREDICTION_LATENCY = Gauge(
    "model_prediction_latency_seconds",
    "Time spent performing model inference (seconds)"
)


@instrumentator.add
def add_custom_metrics(instrumentator: Instrumentator):
    return instrumentator


instrumentator.instrument(app).expose(app, endpoint="/metrics")
templates = Jinja2Templates(directory="templates")


class Model:
    def __init__(self, experiment_name: str, stage: str = "Staging"):

        self.model = mlflow.pyfunc.load_model(f"models:/{experiment_name}/{stage}")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)


iris_model = Model(experiment_name="IrisClassifier", stage="Staging")


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

        prediction = iris_model.predict(df)[0]
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
