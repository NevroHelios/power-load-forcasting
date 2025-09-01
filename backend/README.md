# Backend - FastAPI ML Inference API

This backend provides a high-performance API for power load forecasting using pre-trained ML models (LightGBM and XGBoost).

## Features
- Async FastAPI with ORJSON for optimized JSON responses
- Model caching for fast switching between LightGBM and XGBoost
- CORS enabled for frontend integration
- Health check and model selection endpoints

## Prerequisites
```bash
pip install fastapi uvicorn orjson pickle5 scikit-learn xgboost lightgbm
```

## Local Development
1. Ensure models are in `models/` directory (e.g., `lgbm_model.pkl`, `xgboost_model.pkl`).
2. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```
3. API will be available at `http://localhost:8080`.

## API Endpoints
- `GET /health`: Check API status and active model.
- `POST /set-model`: Switch models (body: `{"name": "lgbm" | "xgboost"}`).
- `POST /predict`: Make predictions (body: LoadFeatures with `get_features` array).

## Model Details
- Supports LightGBM and XGBoost classifiers.
- Expects 7 features: [Usage_kWh, Lagging_Reactive, Leading_Reactive, CO2, Lagging_PF, Leading_PF, NSM].
- Returns prediction (Light/Medium/Maximum Load) and probabilities.

### re-deployment
 UPDATE: git `ci/cd` enabled 
```bash
gcloud run deploy fastapi-ml-app \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10
```

### docker test
-   build
```
docker build -t ml-app .
```
-   run
```
docker run -p ml-app 
```

### load test
-   locust
```
uv run locust -f locustfile.py --host http://localhost:8080
```
-   uvicorn
```
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```