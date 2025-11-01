from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd


templates = Jinja2Templates(directory="templates")


class Model:
    def __init__(self, experiment_name: str, stage: str = "Staging", tracking_uri: str = None):

        self.model = mlflow.pyfunc.load_model(f"models:/{experiment_name}/{stage}")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)


app = FastAPI(title="Iris Classifier Inference")

iris_model = Model(experiment_name="IrisClassifier", stage="Staging")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


@app.post("/", response_class=HTMLResponse)
def predict_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    try:
        df = pd.DataFrame([{
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }])
        prediction = iris_model.predict(df)[0]
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict_api(features: IrisFeatures):
    try:
        df = pd.DataFrame([{
            "SepalLengthCm": features.sepal_length,
            "SepalWidthCm": features.sepal_width,
            "PetalLengthCm": features.petal_length,
            "PetalWidthCm": features.petal_width
        }])
        prediction = iris_model.predict(df)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
