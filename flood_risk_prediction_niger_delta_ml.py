import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# Step 1: Generate Synthetic Flood Risk Dataset
# --------------------------------------------------

np.random.seed(42)

n_samples = 800  # Moderate dataset size

# Synthetic remote sensing & environmental variables
rainfall = np.random.normal(180, 40, n_samples)        # mm
elevation = np.random.normal(20, 10, n_samples)        # meters
slope = np.random.uniform(0, 5, n_samples)             # degrees
distance_to_river = np.random.uniform(0, 5, n_samples) # km
soil_moisture = np.random.uniform(0.2, 0.9, n_samples) # fraction
land_cover = np.random.randint(0, 4, n_samples)        # 0–3 classes

# Flood risk logic (realistic assumptions)
flood_risk = (
    (rainfall > 200).astype(int) +
    (elevation < 15).astype(int) +
    (distance_to_river < 2).astype(int) +
    (soil_moisture > 0.6).astype(int)
)

# Convert to categorical risk levels
flood_risk_class = np.where(
    flood_risk <= 1, 0,
    np.where(flood_risk <= 3, 1, 2)
)

# Create DataFrame
data = pd.DataFrame({
    "Rainfall_mm": rainfall,
    "Elevation_m": elevation,
    "Slope_deg": slope,
    "Distance_to_River_km": distance_to_river,
    "Soil_Moisture": soil_moisture,
    "Land_Cover": land_cover,
    "Flood_Risk": flood_risk_class
})

# --------------------------------------------------
# Step 2: Prepare ML Dataset
# --------------------------------------------------

X = data.drop("Flood_Risk", axis=1)
y = data["Flood_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# Step 3: Train Machine Learning Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Step 4: Model Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Step 5: Feature Importance Analysis
# --------------------------------------------------

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, importance)
plt.title("Feature Importance for Flood Risk Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
