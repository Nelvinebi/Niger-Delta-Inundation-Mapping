# Flood Risk Prediction in the Niger Delta Using Remote Sensing and Machine Learning

## 📌 Project Overview

Flooding is one of the most critical environmental challenges in the Niger Delta region of Nigeria, driven by high rainfall, low-lying terrain, proximity to rivers, and land-use changes. This project demonstrates how **machine learning** and **synthetic remote sensing–derived data** can be used to predict flood risk levels in the Niger Delta.

The project is designed for **academic research, demonstrations, and portfolio purposes**, showing a complete end-to-end ML workflow—from data generation to model evaluation.

---

## 🎯 Objectives

* Simulate realistic flood-related environmental variables for the Niger Delta
* Build a machine learning model to classify flood risk levels
* Analyze the importance of environmental factors influencing flood risk
* Provide a reproducible and extensible ML framework

---

## 🗂️ Project Structure

```
Flood-Risk-Prediction-Niger-Delta/
│
├── flood_risk_prediction_niger_delta_ml.py   # Main ML script
├── Niger_Delta_Flood_Risk_Synthetic_Data.xlsx # Synthetic dataset
├── README.md                                  # Project documentation
```

---

## 🛰️ Dataset Description

The dataset is **synthetically generated** to reflect realistic environmental conditions in the Niger Delta.

### Input Features

| Feature              | Description                     |
| -------------------- | ------------------------------- |
| Rainfall_mm          | Annual rainfall (mm)            |
| Elevation_m          | Surface elevation (meters)      |
| Slope_deg            | Terrain slope (degrees)         |
| Distance_to_River_km | Distance from major rivers (km) |
| Soil_Moisture        | Soil moisture fraction          |
| Land_Cover           | Encoded land cover classes      |

### Target Variable

* **Flood_Risk**

  * `0` → Low Risk
  * `1` → Moderate Risk
  * `2` → High Risk

---

## 🤖 Machine Learning Model

* **Algorithm:** Random Forest Classifier
* **Number of Trees:** 150
* **Max Depth:** 10
* **Train/Test Split:** 70% / 30%

Random Forest was chosen due to its robustness, ability to handle non-linear relationships, and interpretability via feature importance.

---

## 📊 Model Evaluation

The model performance is evaluated using:

* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix
* Feature Importance Analysis

A feature importance plot is generated to identify the most influential flood-driving factors, typically:

* Rainfall
* Elevation
* Distance to rivers
* Soil moisture

---

## 📈 Visual Outputs

* Flood risk classification performance metrics
* Feature importance bar chart highlighting dominant flood predictors

---

## 🧪 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2️⃣ Run the Script

```bash
python flood_risk_prediction_niger_delta_ml.py
```

The script will:

* Generate synthetic flood-risk data
* Train the ML model
* Print evaluation metrics
* Display feature importance visualization

---

## 🌍 Applications

* Flood risk assessment and planning
* Environmental impact studies
* Climate change adaptation research
* Academic demonstrations and ML portfolios

---

## ⚠️ Disclaimer

This project uses **synthetic data** for demonstration purposes. Results should not be used for real-world decision-making without validation using real satellite and field data.

---

## 🔮 Future Improvements

* Integrate real satellite data (Sentinel-1, Sentinel-2, MODIS)
* Add spatial coordinates and GIS-based flood risk mapping
* Extend to time-series flood prediction
* Compare multiple ML models (XGBoost, LSTM)

---

## 👤 Author

**Ebingiye Nelvin Agbozu**
Environmental Science & Machine Learning Research Enthusiast

---

## ⭐ Acknowledgment

Inspired by flood vulnerability challenges in the Niger Delta and the growing role of AI in environmental monitoring.

---

## 📜 License
This project is released under the MIT License. Feel free to use, modify, and share with attribution.



This project is released under the **MIT License**. Feel free to use, modify, and share with attribution.
