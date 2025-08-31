from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from typing import Literal
from contextlib import asynccontextmanager
import os
from models import SetModel, LoadFeatures


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model(name: Literal['xgboost', 'lgbm'] = 'lgbm'):
    model_name = name or os.getenv("MODEL_NAME", name)
    try:
        with open(os.path.join(MODEL_DIR, f"{model_name}_model.pkl"), "rb") as f:
            model = pickle.load(f)
        app.state.model = model
        app.state.model_name = model_name
        return model
    except FileNotFoundError:
        print("ERROR: model not found")
        print(f"GIVEN file name: {name}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(name='lgbm')
    
    yield
    app.state.model = None # before shutdown
    app.state.model_name = None

## fastapi configs
origins = [
    "http://localhost:3000",
    "https://power-load-forcasting.vercel.app/"
]
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.model = None
app.state.model_name = None


@app.get("/health")
async def get_health():
    return {"status": "healthy", "using": app.state.model_name}


@app.post("/set-model")
async def set_model(model: SetModel):
    model_name = model.name
    load_model(model_name)
    return {"status": "model set", "model": model_name}


@app.post("/predict")
async def predict(features: LoadFeatures):
    model = app.state.model
    if model is None:
        return {"error": "Model not loaded"}
    
    # sklearn expects [n_samples, n_features]
    input_data = [features.get_features]
    
    pred = model.predict(input_data)[0]
    
    # do predict_proba (tested on lgbm & xgboost)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0].tolist()
    else:
        proba = None
    
    rev_load_map = {
        0: 'Light Load',
        1: 'Medium Load',
        2: 'Maximum Load'
    }

    return {
        "prediction": str(rev_load_map.get(pred, "Unknown")),
        "probabilities": proba
    }
