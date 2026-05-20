import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from model import train_model, predict_demand, calculate_savings, generate_training_data, FEATURE_COLS

st.set_page_config(
    page_title="AI Coffee Demand Predictor",
    page_icon="☕",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #fff8f0;
        border: 1px solid #f0d9c0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-num {
        font-size: 42px;
        font-weight: 800;
        color: #6f3d1e;
        line-height: 1;
    }
    .metric-label {
        font-size: 13px;
        color: #8b6350;
        margin-top: 6px;
    }
    .savings-card {
        background: #e8f5e9;
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .savings-num {
        font-size: 38px;
        font-weight: 800;
        color: #1a7340;
        line-height: 1;
    }
    .savings-label {
        font-size: 13px;
        color: #2e7d32;
        margin-top: 6px;
    }
    .warning-card {
        background: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #2d7dd2;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("☕ AI-Powered Coffee Demand Predictor")
st.markdown("**Predict daily coffee demand using weather, events, and seasonality. Reduce waste and stockouts with ML.**")
st.markdown("*UC Berkeley Executive Education Capstone · Built by Madhavi Akella · 100% free, no API key needed*")
st.divider()

# ── Train model ───────────────────────────────────────────────────────────────
with st.spinner("Training ML model on historical data..."):
    model, accuracy = train_model(st)
st.success(f"✅ Model ready! Training accuracy: **{accuracy}%**")
st.divider()

# ── Input Panel ───────────────────────────────────────────────────────────────
st.subheader("📅 Enter Today's Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📆 Date & Day**")
    selected_date = st.date_input("Select date", value=date.today())
    day_of_week = selected_date.weekday()
    month = selected_date.month
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.info(f"📅 {day_names[day_of_week]} — {'Weekend 🎉' if day_of_week >= 5 else 'Weekday'}")

with col2:
    st.markdown("**🌤️ Weather**")
    temperature = st.slider("Temperature (°F)", min_value=20, max_value=100, value=65, step=1)
    is_rainy = st.checkbox("🌧️ Rainy day")
    if temperature < 50:
        st.info("🥶 Cold day — expect higher hot drink demand")
    elif temperature > 80:
        st.info("🌞 Hot day — expect higher iced drink demand")

with col3:
    st.markdown("**🎪 Local Events**")
    is_event_day = st.checkbox("🎉 Local event nearby (festival, game, concert)")
    st.markdown("**🎛️ Human Adjustment**")
    adjustment = st.slider("Manual adjustment (cups)", min_value=-30, max_value=30, value=0, step=5,
                           help="Override the prediction based on your local knowledge")
    if adjustment != 0:
        st.info(f"{'➕' if adjustment > 0 else '➖'} You've adjusted by {adjustment:+d} cups")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("☕ Predict Demand", type="primary", use_container_width=True):

    base_prediction = predict_demand(
        model, day_of_week, month, temperature, int(is_rainy), int(is_event_day)
    )
    final_prediction = max(50, base_prediction + adjustment)
    savings = calculate_savings(final_prediction)

    st.divider()
    st.subheader("📊 Prediction Results")

    # ── Main metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num">{final_prediction}</div>
            <div class="metric-label">☕ Predicted Cups Today</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="warning-card">
            <div class="metric-num" style="color:#e65100">{savings['naive_demand']}</div>
            <div class="metric-label">📦 Without ML (over-order estimate)</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-num">{savings['waste_without_ml'] - savings['waste_with_ml']}</div>
            <div class="savings-label">🗑️ Cups of Waste Saved Today</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-num">${savings['daily_waste_savings']}</div>
            <div class="savings-label">💰 Daily Ingredient Savings</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Annual savings ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💰 Projected Annual Business Impact")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-num">${savings['annual_waste_savings']:,.0f}</div>
            <div class="savings-label">Annual Ingredient Waste Savings</div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-num">${savings['annual_stockout_savings']:,.0f}</div>
            <div class="savings-label">Annual Stockout Revenue Recovery</div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-num">${savings['total_annual_savings']:,.0f}</div>
            <div class="savings-label">💡 Total Annual Savings Per Store</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 AI Insights")

    insights = []
    if day_of_week >= 5:
        insights.append("📅 Weekend detected — demand is typically 20–25% higher than weekdays.")
    if is_rainy:
        insights.append("🌧️ Rainy weather increases hot drink demand by ~15 cups on average.")
    if temperature < 50:
        insights.append("🥶 Cold temperature boosts hot coffee demand significantly.")
    if temperature > 80:
        insights.append("🌞 Hot day — consider stocking up on iced drink ingredients.")
    if is_event_day:
        insights.append("🎉 Local event nearby — expect a surge of ~30 additional customers.")
    if month in [11, 12, 1, 2]:
        insights.append("❄️ Winter season — hot drink demand is at its seasonal peak.")
    if month in [6, 7, 8]:
        insights.append("☀️ Summer season — cold brew and iced drinks will be popular.")
    if adjustment > 0:
        insights.append(f"🎛️ You added {adjustment} cups based on local knowledge — great human-in-the-loop adjustment!")
    if not insights:
        insights.append("📊 Typical conditions detected — prediction based on historical averages for this day and season.")

    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    # ── 7-day forecast ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 7-Day Forecast")

    forecast_data = []
    for i in range(7):
        future_date = selected_date + timedelta(days=i)
        future_dow = future_date.weekday()
        future_month = future_date.month
        # Use same weather for forecast (user can adjust)
        future_pred = predict_demand(model, future_dow, future_month, temperature, int(is_rainy), 0)
        forecast_data.append({
            "Date": future_date.strftime("%a %b %d"),
            "Predicted Cups": future_pred,
            "Day Type": "Weekend" if future_dow >= 5 else "Weekday"
        })

    forecast_df = pd.DataFrame(forecast_data)
    st.bar_chart(forecast_df.set_index("Date")["Predicted Cups"])
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        "Built by [Madhavi Akella](https://linkedin.com/in/madhavi-akella-2b8213114) · "
        "[GitHub](https://github.com/madhavi-akella-lab) · "
        "UC Berkeley Executive Education Capstone"
    )
