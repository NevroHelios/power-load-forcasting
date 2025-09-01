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