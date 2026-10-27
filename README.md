# Air Quality Index (AQI) Prediction Using Environmental Data

**Caption:** Predicting air quality from environmental data using machine learning.

## Features
- Streamlit frontend with four top navigation tabs: Home Dashboard, AQI Prediction, Model Evaluation, Prediction History.
- Dashboard with dataset overview, AQI category distribution, average pollutant chart and preview.
- AQI prediction from PM2.5, PM10, NO2, SO2, CO, O3, temperature and humidity.
- Regression model comparison: Linear Regression, Ridge Regression, Random Forest, Extra Trees.
- Evaluation: MAE, RMSE, R², training time, actual-vs-predicted plot, residual distribution and feature importance where supported.
- FastAPI backend and SQLite prediction history with CSV export.

## Run on Windows CMD
Open CMD in this project directory and run:
```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
Keep that window open. Open a second CMD in the same folder:
```cmd
.venv\Scripts\activate
streamlit run app.py
```
Streamlit usually opens at http://localhost:8501. API docs: http://127.0.0.1:8000/docs.

## Dataset note
The app expects `data/aqi.csv` with these numeric columns:
`PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, AQI`.
If a compatible file is absent, the app creates a clearly documented synthetic demonstration dataset locally so the application can run. Synthetic data is for software demonstration and does not represent official measured AQI. Replace it with a verified environmental monitoring dataset for meaningful scientific conclusions.

## AQI category labels
The demo uses common broad AQI bands: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), Severe (above 400). AQI breakpoints differ between countries and pollutant standards; verify local official rules before public/health use.

## Evaluation methodology
Random 80/20 train-test split with fixed seed. Models are compared using held-out MAE, RMSE and R². The best model is selected by lowest held-out RMSE. For a real deployment, use time-aware and location-aware validation, verified measurement units, pollutant-specific official AQI computation, and external testing.

## Important
This is an educational prototype, not an official AQI reporting service or medical advice.
