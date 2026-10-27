import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="AQI Prediction", page_icon="🌫️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.7rem;max-width:1400px}
.subtle{color:#64748b;margin-top:-.6rem}
div[data-testid="stMetric"]{border:1px solid #d8dee8;padding:14px;border-radius:12px}
</style>""", unsafe_allow_html=True)

st.title("🌫️ Air Quality Index (AQI) Prediction")
st.markdown('<p class="subtle">Predicting air quality from environmental data using machine learning</p>', unsafe_allow_html=True)

def get(path, timeout=60):
    r = requests.get(API + path, timeout=timeout)
    r.raise_for_status()
    return r.json()

def post(path, **kwargs):
    r = requests.post(API + path, timeout=180, **kwargs)
    r.raise_for_status()
    return r.json()

try:
    get("/health")
except Exception as exc:
    st.error("The backend is not running or the dataset/model setup failed.")
    st.code("uvicorn backend.main:app --reload")
    st.caption(str(exc))
    st.stop()

dashboard, prediction, evaluation, history = st.tabs(
    ["Home Dashboard", "AQI Prediction", "Model Evaluation", "Prediction History"]
)

with dashboard:
    st.subheader("Dataset overview")
    info = get("/dataset/overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f'{info["records"]:,}')
    c2.metric("Average AQI", f'{info["mean_aqi"]:.1f}')
    c3.metric("Median AQI", f'{info["median_aqi"]:.1f}')
    c4.metric("Pollutants tracked", str(info["pollutant_count"]))

    left, right = st.columns(2)
    with left:
        cats = pd.DataFrame(info["category_counts"])
        st.plotly_chart(px.bar(cats, x="category", y="count", title="AQI category distribution",
                               labels={"category":"AQI category","count":"Records"}), use_container_width=True)
    with right:
        pollutants = pd.DataFrame(info["pollutant_averages"])
        st.plotly_chart(px.bar(pollutants, x="pollutant", y="average",
                               title="Average pollutant levels",
                               labels={"pollutant":"Pollutant","average":"Average concentration"}), use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(pd.DataFrame(get("/dataset/preview")["rows"]), use_container_width=True, hide_index=True)
    st.caption("The bundled dataset is a reproducible teaching dataset. Confirm local monitoring data and units before real-world use.")

with prediction:
    st.subheader("Predict AQI")
    defaults = get("/dataset/defaults")
    st.write("Enter pollutant and weather measurements to estimate the AQI.")
    with st.form("aqi_prediction_form"):
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(defaults["feature_names"]):
            with cols[i % 3]:
                values[feature] = st.number_input(
                    feature, min_value=0.0, value=float(defaults["medians"].get(feature, 0)),
                    format="%.3f", key="input_"+feature
                )
        submitted = st.form_submit_button("Predict AQI", type="primary", use_container_width=True)
    if submitted:
        try:
            result = post("/predict", json={"features": values})
            st.success(f'Estimated AQI: {result["predicted_aqi"]:.1f} — {result["category"]}')
            a, b, c = st.columns(3)
            a.metric("Predicted AQI", f'{result["predicted_aqi"]:.1f}')
            b.metric("AQI category", result["category"])
            c.metric("Model", result["model"])
            st.progress(min(max(result["predicted_aqi"] / 500, 0.0), 1.0))
            st.caption("This is a model estimate, not an official regulatory reading. Use a local monitoring agency for health decisions.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with evaluation:
    st.subheader("Model evaluation")
    try:
        ev = get("/evaluation", timeout=120)
        metrics = pd.DataFrame(ev["metrics"]).T.rename_axis("Model").reset_index()
        st.markdown("#### Regression metrics")
        st.dataframe(metrics.style.format({c:"{:.4f}" for c in metrics.columns if c != "Model"}),
                     use_container_width=True, hide_index=True)
        metric_options = [c for c in metrics.columns if c != "Model"]
        chosen = st.selectbox("Compare models by metric", metric_options, index=metric_options.index("R2") if "R2" in metric_options else 0)
        st.plotly_chart(px.bar(metrics, x="Model", y=chosen, title=f"Model comparison — {chosen}",
                               text_auto=".3f"), use_container_width=True)

        fig = go.Figure()
        for name, values in ev["actual_vs_predicted"].items():
            fig.add_trace(go.Scatter(x=values["actual"], y=values["predicted"], mode="markers",
                                     name=name, opacity=0.55))
        allvals = [v for curve in ev["actual_vs_predicted"].values() for v in curve["actual"] + curve["predicted"]]
        if allvals:
            lo, hi = min(allvals), max(allvals)
            fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Ideal prediction",
                                     line={"dash":"dash","color":"gray"}))
        fig.update_layout(title="Actual vs predicted AQI", xaxis_title="Actual AQI",
                          yaxis_title="Predicted AQI")
        st.plotly_chart(fig, use_container_width=True)

        resid = pd.DataFrame(ev["residuals"])
        st.plotly_chart(px.histogram(resid, x="residual", color="Model", barmode="overlay",
                                     opacity=0.65, title="Prediction residual distribution"),
                        use_container_width=True)
        if ev["feature_importance"]:
            fi = pd.DataFrame(ev["feature_importance"], columns=["Feature", "Importance"]).sort_values("Importance")
            st.plotly_chart(px.bar(fi, x="Importance", y="Feature", orientation="h",
                                   title="Random Forest feature importance"), use_container_width=True)
        st.markdown("#### What the metrics mean")
        st.markdown("""
- **MAE:** average absolute difference between actual and predicted AQI.
- **RMSE:** square root of the average squared prediction error; larger errors are penalized more.
- **R²:** how much variation in AQI the model explains compared with a mean-value baseline.
""")
    except Exception as exc:
        st.error(f"Could not load evaluation: {exc}")

with history:
    st.subheader("Prediction history")
    try:
        history_data = get("/history")
        df = pd.DataFrame(history_data["rows"])
        if df.empty:
            st.info("No predictions saved yet. Make an AQI prediction to create history.")
        else:
            a, b = st.columns(2)
            a.metric("Saved predictions", f"{len(df):,}")
            b.metric("Average predicted AQI", f'{df["predicted_aqi"].mean():.1f}')
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download history CSV", df.to_csv(index=False).encode("utf-8"),
                               "aqi_prediction_history.csv", "text/csv")
    except Exception as exc:
        st.error(f"Could not load prediction history: {exc}")
