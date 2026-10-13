from pathlib import Path
import os
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Cars24 Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ---------------- FILE PATHS ----------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd())

MODEL_PATH = BASE_DIR / "cars24_price_model.pkl"
DATA_PATH = BASE_DIR / "cars24_cleaned_data.csv"


# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# Check that the required files exist
missing_files = [
    path.name for path in [MODEL_PATH, DATA_PATH]
    if not path.exists()
]

if missing_files:
    st.error(
        "Required files are missing: "
        + ", ".join(missing_files)
        + ". Upload them to the same GitHub folder as app.py."
    )
    st.stop()

try:
    model = load_model()
    data = load_data()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()


# ---------------- VALIDATE DATA ----------------
required_columns = [
    "brand", "model", "year", "fuel", "transmission"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    st.error(
        "The cleaned dataset is missing columns: "
        + ", ".join(missing_columns)
        + ". Please check the data-preparation code."
    )
    st.stop()

# Clean fields used by the dropdowns
for col in ["brand", "model", "fuel", "transmission"]:
    data[col] = data[col].astype("string").str.strip()

data["year"] = pd.to_numeric(data["year"], errors="coerce")

data = data.dropna(
    subset=["brand", "model", "year", "fuel", "transmission"]
).copy()

data["year"] = data["year"].astype(int)


# ---------------- APP TITLE ----------------
st.title("🚗 Indian Used Car Price Predictor")

st.write(
    "Estimate a used car's price using its brand, model, "
    "manufacturing year, fuel type and available transmission "
    "or drive-mode information."
)

st.info(
    "Mileage is not used as an input. Predictions depend on "
    "the features and price records available in the dataset."
)


# ---------------- BRAND ----------------
brands = sorted(data["brand"].unique().tolist())

if not brands:
    st.warning("No car brands are available in the dataset.")
    st.stop()

brand = st.selectbox("1. Select car brand", brands)


# ---------------- MODEL ----------------
brand_data = data[data["brand"] == brand]

models = sorted(brand_data["model"].unique().tolist())

if not models:
    st.warning("No models are available for this brand.")
    st.stop()

selected_model = st.selectbox(
    "2. Select car model / variant",
    models
)


# ---------------- YEAR ----------------
car_data = brand_data[
    brand_data["model"] == selected_model
]

years = sorted(
    car_data["year"].unique().tolist(),
    reverse=True
)

if not years:
    st.warning("No manufacturing years are available.")
    st.stop()

year = st.selectbox("3. Manufacturing year", years)


# ---------------- FUEL ----------------
year_data = car_data[car_data["year"] == year]

fuels = sorted(year_data["fuel"].unique().tolist())

if not fuels:
    st.warning("No fuel options are available.")
    st.stop()

fuel = st.selectbox("4. Fuel type", fuels)


# ---------------- TRANSMISSION / DRIVE MODE ----------------
fuel_data = year_data[year_data["fuel"] == fuel]

transmissions = sorted(
    fuel_data["transmission"].unique().tolist()
)

if not transmissions:
    st.warning("No matching options are available.")
    st.stop()

transmission = st.selectbox(
    "5. Transmission / drive mode",
    transmissions
)


# ---------------- SELECTED CAR ----------------
st.divider()
st.subheader("Your selected car")

st.write(f"**Brand:** {brand}")
st.write(f"**Model / variant:** {selected_model}")
st.write(f"**Manufacturing year:** {year}")
st.write(f"**Fuel type:** {fuel}")
st.write(f"**Transmission / drive mode:** {transmission}")


# ---------------- PREDICTION ----------------
if st.button("Predict car price", type="primary"):
    input_data = pd.DataFrame([{
        "brand": brand,
        "model": selected_model,
        "year": int(year),
        "fuel": fuel,
        "transmission": transmission
    }])

    try:
        predicted_price = float(model.predict(input_data)[0])

        if not pd.notna(predicted_price) or predicted_price <= 0:
            st.error("The model returned an invalid price estimate.")
        else:
            st.success(
                f"Estimated resale price: ₹{predicted_price:,.0f}"
            )
            st.caption(
                "This is an estimate, not a guaranteed selling price. "
                "Actual prices may vary with condition, ownership "
                "history, location and market demand."
            )
    except Exception as error:
        st.error(f"Prediction failed: {error}")