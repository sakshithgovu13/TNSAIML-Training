from pathlib import Path
import json, sqlite3, time, urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"aqi.csv"
MODELS=ROOT/"models"
MODEL_FILE=MODELS/"aqi_model.joblib"
EVAL_FILE=MODELS/"evaluation.json"
DB=ROOT/"aqi_predictions.db"
DATA_URL="https://raw.githubusercontent.com/plotly/datasets/master/india-city.csv"

FEATURES=["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity"]
TARGET="AQI"

class AQIService:
    def __init__(self):
        MODELS.mkdir(parents=True,exist_ok=True)
        self.dataset=self.load_data()
        self.feature_names=FEATURES
        if MODEL_FILE.exists() and EVAL_FILE.exists():
            try:
                self.model=joblib.load(MODEL_FILE)
                self.evaluation=json.loads(EVAL_FILE.read_text(encoding="utf-8"))
                self.model_name=self.evaluation.get("selected_model", "Saved model")
            except Exception:
                self.train()
        else:
            self.train()
        with sqlite3.connect(DB) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                predicted_aqi REAL NOT NULL, category TEXT NOT NULL,
                features_json TEXT NOT NULL)""")

    @staticmethod
    def aqi_category(aqi):
        if aqi <= 50: return "Good"
        if aqi <= 100: return "Satisfactory"
        if aqi <= 200: return "Moderate"
        if aqi <= 300: return "Poor"
        if aqi <= 400: return "Very Poor"
        return "Severe"

    def load_data(self):
        if not DATA.exists():
            try:
                urllib.request.urlretrieve(DATA_URL, DATA)
            except Exception:
                # The example download URL may be unavailable. Create a clearly documented
                # synthetic teaching dataset so the app can still start offline.
                df=self.make_demo_dataset()
                df.to_csv(DATA,index=False)
        df=pd.read_csv(DATA)
        # Normalize common alternate column names when users provide a local environmental dataset.
        rename={}
        aliases={
            "PM2.5":["PM2.5","PM2_5","pm25","PM25"],
            "PM10":["PM10","pm10"],
            "NO2":["NO2","no2"],
            "SO2":["SO2","so2"],
            "CO":["CO","co"],
            "O3":["O3","Ozone","o3"],
            "Temperature":["Temperature","temperature","TEMP"],
            "Humidity":["Humidity","humidity","RH"],
            "AQI":["AQI","aqi","Air Quality Index"]
        }
        for standard, opts in aliases.items():
            for col in opts:
                if col in df.columns:
                    rename[col]=standard
                    break
        df=df.rename(columns=rename)
        # If a downloaded generic file is not the expected teaching dataset, create a local synthetic
        # dataset rather than leaving the app unusable; synthetic records are clearly documented.
        required=set(FEATURES+[TARGET])
        if not required.issubset(df.columns) or len(df)<50:
            DATA.unlink(missing_ok=True)
            df=self.make_demo_dataset()
            df.to_csv(DATA,index=False)
        df=df[FEATURES+[TARGET]].apply(pd.to_numeric,errors="coerce")
        df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=[TARGET])
        df=df[(df[FEATURES] >= 0).all(axis=1)]
        if len(df)<50:
            raise ValueError("Dataset needs at least 50 valid rows with all required environmental columns.")
        df["AQI_Category"]=df["AQI"].apply(self.aqi_category)
        return df.reset_index(drop=True)

    @staticmethod
    def make_demo_dataset(n=3000, seed=42):
        rng=np.random.default_rng(seed)
        pm25=np.clip(rng.gamma(2.4,32,n),1,450)
        pm10=np.clip(pm25*rng.uniform(1.1,2.0,n)+rng.normal(0,18,n),1,600)
        no2=np.clip(rng.gamma(2.1,12,n),1,180)
        so2=np.clip(rng.gamma(1.8,7,n),0.5,100)
        co=np.clip(rng.gamma(2,0.6,n),0.1,15)
        o3=np.clip(rng.normal(38,22,n),1,180)
        temp=np.clip(rng.normal(29,7,n),5,48)
        humidity=np.clip(rng.normal(58,19,n),10,100)
        raw=(0.58*pm25 + 0.16*pm10 + 0.55*no2 + 0.35*so2 + 5.5*co
             +0.16*o3 + np.maximum(0,temp-32)*1.1 - np.maximum(0,humidity-70)*0.12
             +rng.normal(0,12,n))
        aqi=np.clip(raw,5,500)
        return pd.DataFrame({"PM2.5":pm25,"PM10":pm10,"NO2":no2,"SO2":so2,"CO":co,"O3":o3,
                             "Temperature":temp,"Humidity":humidity,"AQI":aqi})

    def train(self):
        X=self.dataset[FEATURES]
        y=self.dataset[TARGET]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        configs={
            "Linear Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LinearRegression())]),
            "Ridge Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",Ridge(alpha=1.0))]),
            "Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=120,max_depth=16,min_samples_leaf=2,n_jobs=-1,random_state=42))]),
            "Extra Trees":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",ExtraTreesRegressor(n_estimators=120,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42))])
        }
        metrics={}
        actual_vs_predicted={}
        residuals=[]
        trained={}
        for name,model in configs.items():
            start=time.perf_counter()
            model.fit(Xtr,ytr)
            elapsed=time.perf_counter()-start
            pred=model.predict(Xte)
            metrics[name]={
                "MAE":float(mean_absolute_error(yte,pred)),
                "RMSE":float(np.sqrt(mean_squared_error(yte,pred))),
                "R2":float(r2_score(yte,pred)),
                "Training seconds":float(elapsed)
            }
            # Keep graph payload compact and deterministic.
            idx=np.linspace(0,len(yte)-1,min(400,len(yte)),dtype=int)
            actual_vs_predicted[name]={"actual":yte.iloc[idx].astype(float).tolist(),"predicted":pred[idx].astype(float).tolist()}
            residuals.extend([{"Model":name,"residual":float(a-b)} for a,b in zip(yte.iloc[idx],pred[idx])])
            trained[name]=model
        self.models=trained
        # Choose by lowest RMSE on held-out test data.
        self.model_name=min(metrics,key=lambda name:metrics[name]["RMSE"])
        self.model=trained[self.model_name]
        fi=[]
        estimator=self.model.named_steps["model"]
        if hasattr(estimator,"feature_importances_"):
            fi=[[f,float(v)] for f,v in sorted(zip(FEATURES,estimator.feature_importances_),key=lambda x:x[1],reverse=True)]
        self.evaluation={"metrics":metrics,"actual_vs_predicted":actual_vs_predicted,"residuals":residuals,"feature_importance":fi,"selected_model":self.model_name}
        joblib.dump(self.model,MODEL_FILE)
        EVAL_FILE.write_text(json.dumps(self.evaluation),encoding="utf-8")

    def save_prediction(self,predicted,category,features):
        with sqlite3.connect(DB) as con:
            con.execute("INSERT INTO predictions(predicted_aqi,category,features_json) VALUES(?,?,?)",
                        (predicted,category,json.dumps(features)))

    def get_history(self):
        with sqlite3.connect(DB) as con:
            con.row_factory=sqlite3.Row
            rows=con.execute("SELECT id,created_at,predicted_aqi,category,features_json FROM predictions ORDER BY id DESC LIMIT 1000").fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["features"]=json.loads(item.pop("features_json"))
            result.append(item)
        return result

_SERVICE=None
def get_service():
    global _SERVICE
    if _SERVICE is None:
        _SERVICE=AQIService()
    return _SERVICE
