import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="H₂ Wettability Predictor", layout="centered")
st.title("🔬 Hydrogen Wettability Predictor")
st.markdown("Predict contact angle for H₂ on **coal, sandstone, or carbonate** using a Random Forest model trained on 78 literature data points (R² = 0.84).")

rock_type = st.selectbox("Rock Type", ['Coal', 'Sandstone', 'Carbonate'])
pressure = st.number_input("Pressure (MPa)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
temperature = st.number_input("Temperature (K)", min_value=273.0, max_value=400.0, value=313.0, step=1.0)

if st.button("Predict Contact Angle"):
    rock_code = {'Coal':0, 'Sandstone':1, 'Carbonate':2}[rock_type]
    X = pd.DataFrame([[pressure, temperature, rock_code]], columns=['Pressure (Mpa)', 'Temperature (K)', 'rock_code'])
    X_scaled = scaler.transform(X)
    angle = model.predict(X_scaled)[0]
    st.success(f"✅ Predicted Contact Angle: **{angle:.1f}°**")
    st.caption("Model MAE = 3.2° · Trained on 78 literature points")

st.markdown("---")
st.markdown("👉 [Get API access or consultation](https://shahbazumarsky1.gumroad.com/l/bobhmpp)")
