# src/forecast/town_alerts.py
import requests
import pandas as pd
from datetime import datetime

TOWNS = {
    "Port Harcourt": (4.8156, 7.0498),
    "Yenagoa": (4.9267, 6.2676),
    "Warri": (5.5167, 5.7500),
    "Sapele": (5.8961, 5.6767),
    "Ughelli": (5.5000, 6.0333),
    "Asaba": (6.2000, 6.7333),
    "Aba": (5.1167, 7.3667),
    "Calabar": (4.9757, 8.3417)
}

def check_alerts(threshold_mm=50):
    """Check 3-day rainfall forecast for all towns. Alert if above threshold."""
    alerts = []
    
    for town, (lat, lon) in TOWNS.items():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "forecast_days": 3,
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        total_rain = sum(data["daily"]["precipitation_sum"])
        
        if total_rain > threshold_mm:
            alerts.append({
                "town": town,
                "3_day_total_mm": round(total_rain, 1),
                "risk_level": "HIGH" if total_rain > 100 else "MODERATE"
            })
    
    return pd.DataFrame(alerts)

if __name__ == "__main__":
    print(f"Forecast check: {datetime.now()}")
    alerts = check_alerts(threshold_mm=50)
    
    if len(alerts) > 0:
        print(f"\n⚠️  FLOOD RISK ALERTS ({len(alerts)} towns):")
        print(alerts.to_string(index=False))
    else:
        print("\n✅ No flood risk in next 3 days.")