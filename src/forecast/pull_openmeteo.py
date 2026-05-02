# src/forecast/pull_openmeteo.py
import requests
import pandas as pd
import numpy as np

def get_forecast(lat, lon, days=7):
    """
    Pull rainfall forecast from Open-Meteo (free, no registration).
    Returns daily precipitation forecast for a single point.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "forecast_days": days,
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "precipitation_mm": data["daily"]["precipitation_sum"]
    })
    
    return df

# Test: Port Harcourt (Niger Delta)
forecast = get_forecast(4.8, 7.0, days=7)
print(forecast)