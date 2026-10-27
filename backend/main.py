from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from ml.service import get_service

app = FastAPI(title="AQI Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: dict[str, float]

@app.get("/health")
def health():
    s = get_service()
    return {"status":"ok", "dataset_ready":True, "model_ready":True}

@app.get("/dataset/overview")
def overview():
    s = get_service()
    df = s.dataset
    counts = df["AQI_Category"].value_counts().reindex(
        ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"], fill_value=0
    )
    pollutants = [{"pollutant":c,"average":float(df[c].mean())} for c in s.feature_names if c in df.columns]
    return {
        "records":int(len(df)), "mean_aqi":float(df["AQI"].mean()),
        "median_aqi":float(df["AQI"].median()), "pollutant_count":len(s.feature_names),
        "category_counts":[{"category":str(k),"count":int(v)} for k,v in counts.items()],
        "pollutant_averages":pollutants
    }

@app.get("/dataset/preview")
def preview():
    return {"rows":get_service().dataset.head(12).replace({np.nan:None}).to_dict("records")}

@app.get("/dataset/defaults")
def defaults():
    s=get_service()
    return {"feature_names":s.feature_names, "medians":s.dataset[s.feature_names].median(numeric_only=True).fillna(0).to_dict()}

@app.post("/predict")
def predict(req:PredictionRequest):
    s=get_service()
    missing=[f for f in s.feature_names if f not in req.features]
    if missing:
        raise HTTPException(422, "Missing features: "+", ".join(missing))
    X=pd.DataFrame([[req.features[f] for f in s.feature_names]],columns=s.feature_names)
    if not np.isfinite(X.to_numpy(dtype=float)).all() or (X < 0).any().any():
        raise HTTPException(422,"Measurements must be finite, non-negative numbers.")
    pred=float(s.model.predict(X)[0])
    pred=max(0.0,pred)
    category=s.aqi_category(pred)
    s.save_prediction(pred,category,req.features)
    return {"predicted_aqi":pred,"category":category,"model":s.model_name}

@app.get("/evaluation")
def evaluation():
    return get_service().evaluation

@app.get("/history")
def history():
    return {"rows":get_service().get_history()}
