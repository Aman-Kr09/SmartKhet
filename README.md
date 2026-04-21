# 🌾 SmartKhet – Smart Farming Assistant

**SmartKhet** is a smart farming web application designed to assist farmers and agri-enthusiasts by integrating Deep Learning and data-driven solutions. It provides intelligent support for identifying plant diseases, choosing the right crops and fertilizers, and offering real-time action advice based on weather conditions.

---

## 🚀 Features

- **🌱 Plant Disease Detection**  
  Upload an image of a crop leaf to detect diseases using a deep learning model trained with TensorFlow and Keras.

- **🌾 Crop Recommendation**  
  Suggests the best crop to grow based on soil nutrients, temperature, humidity, pH level, and rainfall using a classification model.

- **💊 Fertilizer Suggestion**  
  Based on the nutrient requirements of crops and current soil condition, the app recommends suitable fertilizers.

- **📈 Action Advisory**  
  Provides useful farming suggestions by analyzing **real-time weather data** from OpenWeatherMap API, based on selected city and state.

---

## 🧠 Model Overview

### 🔍 1. **Plant Disease Detection**
- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow & Keras
- **Input**: Image of a leaf
- **Output**: Predicted disease (or healthy)

### 🌾 2. **Crop Recommendation**
- **Model**: Classification Model (Random Forest or similar)
- **Input Features**:
  - Nitrogen (N), Phosphorus (P), Potassium (K)
  - Temperature, Humidity
  - pH level, Rainfall
- **Output**: Recommended crop (e.g., Rice, Wheat, Cotton)

### 💊 3. **Fertilizer Suggestion**
- **Approach**: Rule-based logic
- **Logic**:
  - Compares current NPK values with ideal values for the selected crop
  - Suggests fertilizers to balance soil nutrients

### 📈 4. **Action Advisory**
- **Approach**: No ML used
- **Data Source**: Real-time weather from OpenWeatherMap API
- **Working**:
  - User selects a city and state
  - App fetches temperature, humidity, wind, rainfall, etc.
  - Based on predefined rules, it gives useful suggestions (e.g., “Apply irrigation,” “Avoid pesticide spraying,” etc.)

---

## 🧩 Process Block Diagram (End-to-End)

```mermaid
flowchart TD
    A[User Opens SmartKhet App] --> B[Select Module]

    B --> C1[Plant Disease Detection]
    B --> C2[Crop Recommendation]
    B --> C3[Fertilizer Suggestion]
    B --> C4[Action Advisory]

    %% Disease Detection Flow
    C1 --> D1[Upload Leaf Image]
    D1 --> E1[Image Preprocessing]
    E1 --> F1[CNN/TFLite Model Inference]
    F1 --> G1[Predicted Disease or Healthy]
    G1 --> H1[Display Result to User]

    %% Crop Recommendation Flow
    C2 --> D2[Input Soil + Climate Values\nN, P, K, Temperature, Humidity, pH, Rainfall]
    D2 --> E2[Feature Processing]
    E2 --> F2[Classification Model Prediction]
    F2 --> G2[Recommended Crop]
    G2 --> H2[Display Result to User]

    %% Fertilizer Suggestion Flow
    C3 --> D3[Input Crop + Current Soil NPK]
    D3 --> E3[Compare with Ideal NPK]
    E3 --> F3[Rule-Based Decision Engine]
    F3 --> G3[Suggested Fertilizer Actions]
    G3 --> H3[Display Result to User]

    %% Action Advisory Flow
    C4 --> D4[Select State and City]
    D4 --> E4[Fetch Weather Data from OpenWeatherMap API]
    E4 --> F4[Rule-Based Advisory Logic]
    F4 --> G4[Farming Action Suggestions]
    G4 --> H4[Display Result to User]

    %% Common End
    H1 --> Z[Farmer Takes Informed Action]
    H2 --> Z
    H3 --> Z
    H4 --> Z
```

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend Models**: TensorFlow + Keras
- **APIs**: OpenWeatherMap
- **Languages**: Python, HTML (within markdown)

---

## 📦 Installation & Run Locally

```bash
git clone https://github.com/Aman-Kr09/smartkhet.git
cd smartkhet
pip install -r requirements.txt
streamlit run app.py

