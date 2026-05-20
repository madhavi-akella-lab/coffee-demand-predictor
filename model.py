import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error


def generate_training_data(n_days=365):
    """
    Generate realistic synthetic coffee demand training data.
    Incorporates weather, day of week, local events, and seasonality.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    data = []
    for date in dates:
        day_of_week = date.dayofweek        # 0=Mon, 6=Sun
        month = date.month
        is_weekend = 1 if day_of_week >= 5 else 0

        # Weather simulation
        temp = np.random.normal(65, 15)     # Fahrenheit
        is_rainy = 1 if np.random.random() < 0.25 else 0
        is_cold = 1 if temp < 50 else 0

        # Local event simulation
        is_event_day = 1 if np.random.random() < 0.1 else 0

        # Seasonality boost
        season_boost = 0
        if month in [11, 12, 1, 2]:        # Winter — hot drinks up
            season_boost = 15
        elif month in [6, 7, 8]:           # Summer — iced drinks up
            season_boost = 10

        # Base demand with realistic factors
        base = 120
        demand = (
            base
            + season_boost
            + (25 if is_weekend else 0)
            + (20 if is_cold else 0)
            + (15 if is_rainy else 0)
            + (30 if is_event_day else 0)
            - (0.3 * max(0, temp - 80))    # Hot days reduce hot coffee
            + np.random.normal(0, 8)        # Random noise
        )
        demand = max(50, int(demand))

        data.append({
            "date": date,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "temperature_f": round(temp, 1),
            "is_rainy": is_rainy,
            "is_cold": is_cold,
            "is_event_day": is_event_day,
            "season_boost": season_boost,
            "demand": demand,
        })

    return pd.DataFrame(data)


FEATURE_COLS = [
    "day_of_week", "month", "is_weekend",
    "temperature_f", "is_rainy", "is_cold",
    "is_event_day", "season_boost"
]


def train_model(st):
    @st.cache_resource
    def _train():
        df = generate_training_data(365)
        X = df[FEATURE_COLS]
        y = df["demand"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Compute accuracy on training data as a baseline
        preds = model.predict(X)
        mape = mean_absolute_percentage_error(y, preds)
        accuracy = round((1 - mape) * 100, 1)
        return model, accuracy

    return _train()


def predict_demand(model, day_of_week, month, temperature_f, is_rainy, is_event_day):
    """Predict coffee demand for given conditions."""
    is_weekend = 1 if day_of_week >= 5 else 0
    is_cold = 1 if temperature_f < 50 else 0
    season_boost = 0
    if month in [11, 12, 1, 2]:
        season_boost = 15
    elif month in [6, 7, 8]:
        season_boost = 10

    features = np.array([[
        day_of_week, month, is_weekend,
        temperature_f, is_rainy, is_cold,
        is_event_day, season_boost
    ]])

    prediction = model.predict(features)[0]
    return max(50, int(round(prediction)))


def calculate_savings(predicted_demand, actual_demand=None):
    """
    Calculate waste and stockout metrics.
    Returns a dict with business impact numbers.
    """
    # Industry average without ML: ~60% accuracy
    naive_demand = int(predicted_demand * 1.4)   # Over-ordering by 40%
    waste_without_ml = max(0, naive_demand - predicted_demand)
    waste_with_ml = int(predicted_demand * 0.05)  # 5% buffer

    cost_per_unit = 0.80  # Average ingredient cost per cup
    revenue_per_cup = 4.50

    daily_waste_savings = (waste_without_ml - waste_with_ml) * cost_per_unit
    annual_waste_savings = daily_waste_savings * 365

    # Stockout reduction
    stockout_reduction = 0.75   # 75% fewer stockouts
    daily_stockout_savings = predicted_demand * 0.05 * stockout_reduction * revenue_per_cup
    annual_stockout_savings = daily_stockout_savings * 365

    total_annual_savings = annual_waste_savings + annual_stockout_savings

    return {
        "predicted_demand": predicted_demand,
        "naive_demand": naive_demand,
        "waste_without_ml": waste_without_ml,
        "waste_with_ml": waste_with_ml,
        "daily_waste_savings": round(daily_waste_savings, 2),
        "annual_waste_savings": round(annual_waste_savings, 2),
        "annual_stockout_savings": round(annual_stockout_savings, 2),
        "total_annual_savings": round(total_annual_savings, 2),
    }
