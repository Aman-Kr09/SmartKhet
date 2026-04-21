import csv
import json
import os
import pickle
from statistics import mean

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="SmartKhet - Performance Analytics", layout="wide")
st.title("SmartKhet Performance Analytics")
st.caption("Interactive model-performance and system-flow dashboard")


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(ROOT_DIR, "pages")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compute_crop_metrics():
    crop_dict = {
        "rice": 0,
        "maize": 1,
        "chickpea": 2,
        "kidneybeans": 3,
        "pigeonpeas": 4,
        "mothbeans": 5,
        "mungbean": 6,
        "blackgram": 7,
        "lentil": 8,
        "pomegranate": 9,
        "banana": 10,
        "mango": 11,
        "grapes": 12,
        "watermelon": 13,
        "muskmelon": 14,
        "apple": 15,
        "orange": 16,
        "papaya": 17,
        "coconut": 18,
        "cotton": 19,
        "jute": 20,
        "coffee": 21,
    }

    data = load_csv(os.path.join(PAGES_DIR, "Crop_recommendation.csv"))
    features = []
    labels = []
    for row in data:
        features.append([
            float(row["N"]),
            float(row["P"]),
            float(row["K"]),
            float(row["temperature"]),
            float(row["humidity"]),
            float(row["ph"]),
            float(row["rainfall"]),
        ])
        labels.append(crop_dict[row["label"]])

    x_data = np.array(features, dtype=float)
    y_data = np.array(labels, dtype=int)
    _, x_test, _, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    with open(os.path.join(ROOT_DIR, "rf_model_compressed.pkl"), "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "samples": int(len(y_test)),
    }


def compute_fertilizer_metrics():
    fer_map = {
        "Urea": 0,
        "DAP": 1,
        "14-35-14": 2,
        "28-28": 3,
        "17-17-17": 4,
        "20-20": 5,
        "10-26-26": 6,
    }
    soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    crop_types = [
        "Maize",
        "Sugarcane",
        "Cotton",
        "Tobacco",
        "Paddy",
        "Barley",
        "Wheat",
        "Oil seeds",
        "Pulses",
        "Ground Nuts",
        "Millets",
    ]

    data = load_csv(os.path.join(PAGES_DIR, "Fertilizer Prediction.csv"))
    features = []
    labels = []
    for row in data:
        features.append([
            float(row["Temparature"]),
            float(row["Humidity "]),
            float(row["Moisture"]),
            soil_types.index(row["Soil Type"]),
            crop_types.index(row["Crop Type"]),
            float(row["Nitrogen"]),
            float(row["Potassium"]),
            float(row["Phosphorous"]),
        ])
        labels.append(fer_map[row["Fertilizer Name"]])

    x_data = np.array(features, dtype=float)
    y_data = np.array(labels, dtype=int)
    _, x_test, _, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    model = joblib.load(os.path.join(ROOT_DIR, "fer_model.pkl"))
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "samples": int(len(y_test)),
    }


def create_architecture_figure():
    labels = [
        "User Inputs",
        "Leaf Image",
        "Soil + Climate",
        "Soil + Crop + NPK",
        "State + City",
        "Disease Model",
        "Crop Model",
        "Fertilizer Model",
        "Weather API",
        "Advisory Rules",
        "SmartKhet UI",
        "Farmer Decisions",
    ]

    source = [0, 0, 0, 0, 1, 2, 3, 4, 8, 5, 6, 7, 9, 10]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    value = [35, 25, 25, 15, 35, 25, 25, 15, 15, 35, 25, 25, 15, 100]

    fig = go.Figure(
        go.Sankey(
            node={
                "label": labels,
                "pad": 18,
                "thickness": 17,
                "line": {"color": "#223", "width": 0.6},
                "color": [
                    "#2f7d32",
                    "#55a630",
                    "#80b918",
                    "#aacc00",
                    "#bfd200",
                    "#2a9d8f",
                    "#1d3557",
                    "#264653",
                    "#457b9d",
                    "#8ab17d",
                    "#3a5a40",
                    "#588157",
                ],
            },
            link={
                "source": source,
                "target": target,
                "value": value,
                "color": "rgba(54, 162, 89, 0.35)",
            },
            arrangement="snap",
        )
    )
    fig.update_layout(
        title="Interactive SmartKhet System Architecture",
        font={"size": 12},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


history = load_json(os.path.join(PAGES_DIR, "training_hist.json"))
epochs = list(range(1, len(history["accuracy"]) + 1))

crop_metrics = None
fertilizer_metrics = None
crop_metrics_error = ""
fertilizer_metrics_error = ""

try:
    crop_metrics = compute_crop_metrics()
except Exception as exc:
    crop_metrics_error = str(exc)

try:
    fertilizer_metrics = compute_fertilizer_metrics()
except Exception as exc:
    fertilizer_metrics_error = str(exc)

with st.sidebar:
    st.subheader("Interaction Controls")
    advisory_trigger_rate = st.slider("Advisory trigger rate (%)", 0, 100, 72)
    avg_tips_per_alert = st.slider("Average tips per advisory", 1, 7, 3)
    estimated_adoption = st.slider("Estimated farmer adoption (%)", 0, 100, 68)
    estimated_satisfaction = st.slider("Estimated usefulness rating (%)", 0, 100, 74)

st.subheader("1) Disease Detection Performance")
col_a, col_b = st.columns(2)

with col_a:
    fig_disease = go.Figure()
    fig_disease.add_trace(go.Scatter(x=epochs, y=history["accuracy"], mode="lines+markers", name="Train Accuracy"))
    fig_disease.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"], mode="lines+markers", name="Validation Accuracy"))
    fig_disease.update_layout(title="Accuracy over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy")
    st.plotly_chart(fig_disease, use_container_width=True)

with col_b:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines+markers", name="Train Loss"))
    fig_loss.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Validation Loss"))
    fig_loss.update_layout(title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

st.subheader("2) Disease Detection Accuracy Error Types")
final_train_acc = history["accuracy"][-1]
final_val_acc = history["val_accuracy"][-1]
generalization_gap = max(final_train_acc - final_val_acc, 0.0)
total_error = 1.0 - final_val_acc
residual_error = max(total_error - generalization_gap, 0.0)

fig_error = go.Figure(
    data=[
        go.Pie(
            labels=["Generalization gap", "Residual misclassification error"],
            values=[generalization_gap * 100.0, residual_error * 100.0],
            hole=0.48,
            textinfo="label+percent",
        )
    ]
)
fig_error.update_layout(title="Validation Error Breakdown (Disease Model)")
st.plotly_chart(fig_error, use_container_width=True)

st.subheader("3) Crop Recommendation and Fertilizer Suggestion Performance")
metric_option = st.radio("Choose metric", ["Accuracy", "Precision", "Recall", "F1"], horizontal=True)
metric_key = metric_option.lower() if metric_option != "F1" else "f1"

labels = []
values = []
notes = []

if crop_metrics is not None:
    labels.append("Crop Recommendation")
    values.append(crop_metrics[metric_key] * 100.0)
    notes.append(f"n={crop_metrics['samples']}")
else:
    labels.append("Crop Recommendation")
    values.append(0.0)
    notes.append("metric unavailable")

if fertilizer_metrics is not None:
    labels.append("Fertilizer Suggestion")
    values.append(fertilizer_metrics[metric_key] * 100.0)
    notes.append(f"n={fertilizer_metrics['samples']}")
else:
    labels.append("Fertilizer Suggestion")
    values.append(0.0)
    notes.append("metric unavailable")

fig_model_compare = go.Figure(
    data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f"{value:.2f}% ({note})" for value, note in zip(values, notes)],
            textposition="outside",
            marker_color=["#2a9d8f", "#e76f51"],
        )
    ]
)
fig_model_compare.update_layout(
    title=f"{metric_option} Comparison",
    yaxis_title="Percentage",
    yaxis_range=[0, 100],
)
st.plotly_chart(fig_model_compare, use_container_width=True)

if crop_metrics_error:
    st.warning(f"Crop metrics note: {crop_metrics_error}")
if fertilizer_metrics_error:
    st.warning(f"Fertilizer metrics note: {fertilizer_metrics_error}")

st.subheader("4) Fertilizer Effectiveness and Crop Coverage")
fert_data = load_csv(os.path.join(PAGES_DIR, "Fertilizer Prediction.csv"))
crop_data = load_csv(os.path.join(PAGES_DIR, "Crop_recommendation.csv"))

fert_count = {}
for row in fert_data:
    fert_name = row["Fertilizer Name"]
    fert_count[fert_name] = fert_count.get(fert_name, 0) + 1

crop_count = {}
for row in crop_data:
    crop_name = row["label"]
    crop_count[crop_name] = crop_count.get(crop_name, 0) + 1

top_fertilizers = sorted(fert_count.items(), key=lambda item: item[1], reverse=True)[:7]
top_crops = sorted(crop_count.items(), key=lambda item: item[1], reverse=True)[:10]

col_c, col_d = st.columns(2)
with col_c:
    fig_fert_dist = go.Figure(
        data=[go.Bar(x=[item[0] for item in top_fertilizers], y=[item[1] for item in top_fertilizers], marker_color="#588157")]
    )
    fig_fert_dist.update_layout(title="Fertilizer Recommendation Frequency", xaxis_title="Fertilizer", yaxis_title="Count")
    st.plotly_chart(fig_fert_dist, use_container_width=True)

with col_d:
    fig_crop_dist = go.Figure(
        data=[go.Bar(x=[item[0] for item in top_crops], y=[item[1] for item in top_crops], marker_color="#3a86ff")]
    )
    fig_crop_dist.update_layout(title="Crop Recommendation Label Distribution", xaxis_title="Crop", yaxis_title="Count")
    st.plotly_chart(fig_crop_dist, use_container_width=True)

st.subheader("5) Weather-Based Advisory Usefulness")
usefulness_score = min(
    100.0,
    (0.35 * advisory_trigger_rate)
    + (0.15 * (avg_tips_per_alert * 14.0))
    + (0.25 * estimated_adoption)
    + (0.25 * estimated_satisfaction),
)

fig_weather = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=usefulness_score,
        delta={"reference": 70},
        title={"text": "Weather Advisory Usefulness Index"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2a9d8f"},
            "steps": [
                {"range": [0, 40], "color": "#ffddd2"},
                {"range": [40, 70], "color": "#fec89a"},
                {"range": [70, 100], "color": "#caffbf"},
            ],
        },
    )
)
st.plotly_chart(fig_weather, use_container_width=True)

advice_components = {
    "Trigger Coverage": advisory_trigger_rate,
    "Tip Density": avg_tips_per_alert * 14.0,
    "Adoption": estimated_adoption,
    "Satisfaction": estimated_satisfaction,
}
fig_weather_components = go.Figure(
    data=[go.Bar(x=list(advice_components.keys()), y=list(advice_components.values()), marker_color="#84a98c")]
)
fig_weather_components.update_layout(title="Weather Advisory Component Scores", yaxis_title="Score (0-100)")
st.plotly_chart(fig_weather_components, use_container_width=True)

st.subheader("6) All Model Accuracy Snapshot")
disease_acc = final_val_acc * 100.0
crop_acc = (crop_metrics["accuracy"] * 100.0) if crop_metrics is not None else 0.0
fert_acc = (fertilizer_metrics["accuracy"] * 100.0) if fertilizer_metrics is not None else 0.0
weather_proxy = usefulness_score

fig_all_accuracy = go.Figure(
    data=[
        go.Scatterpolar(
            r=[disease_acc, crop_acc, fert_acc, weather_proxy],
            theta=["Disease Detection", "Crop Recommendation", "Fertilizer Suggestion", "Weather Advisory"],
            fill="toself",
            name="Model/Module Score",
        )
    ]
)
fig_all_accuracy.update_layout(
    polar={"radialaxis": {"visible": True, "range": [0, 100]}},
    title="Combined Accuracy and Usefulness View",
)
st.plotly_chart(fig_all_accuracy, use_container_width=True)

st.subheader("7) Interactive System Architecture Block Diagram")
st.plotly_chart(create_architecture_figure(), use_container_width=True)

st.info(
    "Notes: Disease metrics come from training history. Crop/Fertilizer metrics are computed from saved models and CSV test-splits. "
    "Weather usefulness is an interactive operational index based on configurable assumptions."
)
