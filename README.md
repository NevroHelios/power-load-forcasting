# Power Load Forecasting Project

## Overview

This project implements a machine learning-based power load classification system that predicts electrical load types based on power consumption patterns. The system analyzes electrical grid data to classify loads into three categories: Light Load, Medium Load, and Maximum Load.

## Dataset

The dataset contains power consumption data from 2018 with measurements collected every 15 minutes, resulting in 96 readings per day and over 35,000 total records. 

### Key Features:
- **Date_Time**: Timestamp of the measurement
- **Usage_kWh**: Power consumption in kilowatt-hours
- **Lagging_Current_Reactive.Power_kVarh**: Reactive power (lagging)
- **Leading_Current_Reactive_Power_kVarh**: Reactive power (leading)
- **CO2(tCO2)**: Carbon dioxide emissions
- **Lagging_Current_Power_Factor**: Power factor (lagging current)
- **Leading_Current_Power_Factor**: Power factor (leading current)
- **Load_Type**: Target variable (Light_Load, Medium_Load, Maximum_Load)

## Methodology

### 1. Data Preprocessing
- **Time Feature Engineering**: Extracted temporal features (hour, day, month, weekday)
- **Cyclical Features**: Created sinusoidal transformations for temporal patterns
- **Holiday Detection**: Incorporated holiday information for better seasonality modeling
- **Missing Value Imputation**: Used monthly and load-type-specific means for imputation
- **Feature Engineering**: Created additional features like total reactive power, power factor differences, and usage ratios

### 2. Model Development
Three state-of-the-art machine learning classifiers were trained and evaluated:

- **Gradient Boosting Classifier**: Ensemble method with boosting
- **XGBoost Classifier**: Optimized gradient boosting with advanced regularization
- **LightGBM Classifier**: Fast gradient boosting with optimal performance

### 3. Model Performance
The models achieved excellent performance metrics:

- **Best Model**: LightGBM Classifier
- **Accuracy**: 92%
- **F1 Score**: 0.91 macro-F1
- **Training Period**: January-November 2018
- **Testing Period**: December 2018

## Key Achievements

✅ **High Accuracy**: Achieved 92% classification accuracy across three load types  
✅ **Robust Feature Engineering**: Comprehensive temporal and domain-specific features  
✅ **Model Comparison**: Systematic evaluation of multiple ML algorithms  
✅ **Real-world Application**: 15-minute interval predictions for grid management  
✅ **Deployment Success**: Built and deployed a load forecasting ML model achieving 92% accuracy and 0.91 macro-F1 on real-world energy datasets  
✅ **API Optimization**: Engineered an async FastAPI inference API with ORJSON and caching, sustaining 430 RPS throughput under 500 concurrent users  
✅ **Latency Improvement**: Reduced 95th percentile latency from 3000 ms to 1100 ms (2.7× faster) via multi-worker scaling and optimized serialization  
✅ **Core Prediction Speed**: Optimized FastAPI inference pipeline to deliver <1 ms core prediction latency with 430 RPS throughput under 500 concurrent users  

## Business Impact

This power load forecasting system provides significant value for:

- **Grid Operators**: Better load planning and resource allocation
- **Energy Companies**: Improved demand forecasting and pricing strategies
- **Sustainability**: Optimized energy distribution reducing waste
- **Infrastructure Planning**: Data-driven decisions for grid expansion

## How to Run

### Prerequisites
```bash
cd backend
# Backend dependencies in /backend
pip install fastapi uvicorn orjson pickle5 scikit-learn xgboost lightgbm
# or
uv sync

# Frontend dependencies in /frontend
cd frontend
npm install next react react-dom recharts lucide-react
```

### Launch the Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

### Launch the Frontend
```bash
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## Project Structure
```
├── backend/
│   ├── main.py                    # FastAPI Application
│   ├── models.py                  # Pydantic Models
│   ├── models/                    # Trained ML Models
│   └── README.md                  # Backend Documentation
├── frontend/
│   ├── app/                       # Next.js App Directory
│   │   ├── page.tsx               # Main Dashboard Page
│   │   ├── layout.tsx             # App Layout
│   │   └── globals.css            # Global Styles
│   └── README.md                  # Frontend Documentation
├── data/
│   ├── load_data.csv              # Main dataset
│   ├── train.csv                  # Training subset
│   └── test.csv                   # Testing subset
├── eda.ipynb                      # Exploratory Data Analysis & Model Training
└── README.md                      # Project Documentation
```

## Technology Stack

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning framework
- **XGBoost & LightGBM**: Advanced gradient boosting
- **FastAPI**: High-performance API framework
- **Next.js**: React framework for the frontend
- **Recharts**: Data visualization library
- **Tailwind CSS**: Utility-first CSS framework

## Future Enhancements

- Real-time load prediction API
- Integration with IoT sensors
- Deep learning models for time series forecasting
- Multi-step ahead predictions
- Anomaly detection for grid failures

---

*This project demonstrates the application of machine learning in energy sector optimization and grid management.*
