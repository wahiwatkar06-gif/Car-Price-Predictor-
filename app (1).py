
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Cars24 Price Predictor",
    page_icon="🚗",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "cars24_price_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "cars24_cleaned_data.csv")

model = load_model()
data = load_data()

st.title("🚗 Indian Used Car Price Predictor")
st.write("Estimate a used car's price from its available specifications.")
st.info("Mileage is not used as an input to this model.")

required = ["brand", "model", "year", "fuel", "transmission", "price"]
if not all(c in data.columns for c in required):
    st.error("The cleaned dataset is missing required columns.")
    st.stop()

for col in ["brand", "model", "fuel", "transmission"]:
    data[col] = data[col].astype("string").str.strip()

data["year"] = pd.to_numeric(data["year"], errors="coerce")
data["price"] = pd.to_numeric(data["price"], errors="coerce")
data = data.dropna(subset=required).copy()
data["year"] = data["year"].astype(int)

brands = sorted(data["brand"].unique().tolist())
brand = st.selectbox("1. Select brand", brands)

brand_data = data[data["brand"] == brand]
models = sorted(brand_data["model"].unique().tolist())
selected_model = st.selectbox("2. Select model / variant", models)

car_data = brand_data[brand_data["model"] == selected_model]
years = sorted(car_data["year"].unique().tolist(), reverse=True)
year = st.selectbox("3. Manufacturing year", years)

year_data = car_data[car_data["year"] == year]
fuels = sorted(year_data["fuel"].unique().tolist())
fuel = st.selectbox("4. Fuel type", fuels)

fuel_data = year_data[year_data["fuel"] == fuel]
transmissions = sorted(fuel_data["transmission"].unique().tolist())
transmission = st.selectbox(
    "5. Transmission / drive mode",
    transmissions
)

st.divider()
st.subheader("Selected car")
st.write(f"**Brand:** {brand}")
st.write(f"**Model:** {selected_model}")
st.write(f"**Year:** {year}")
st.write(f"**Fuel:** {fuel}")
st.write(f"**Transmission / drive mode:** {transmission}")

if st.button("Predict car price", type="primary"):
    input_data = pd.DataFrame([{
        "brand": brand,
        "model": selected_model,
        "year": int(year),
        "fuel": fuel,
        "transmission": transmission
    }])

    try:
        price = float(model.predict(input_data)[0])
        if not pd.notna(price) or price <= 0:
            st.error("The model returned an invalid price.")
        else:
            st.success(f"Estimated resale price: ₹{price:,.0f}")
            st.caption(
                "This is an estimate, not a guaranteed selling price. "
                "Actual prices depend on vehicle condition, location, "
                "ownership and market demand."
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
