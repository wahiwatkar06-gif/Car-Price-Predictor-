import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/rajtilakls2510/car_price_predictor/master/quikr_car.csv"
LOCAL_DATA = Path(__file__).with_name("quikr_car.csv")

@st.cache_data
def load_data():
    if LOCAL_DATA.exists():
        return pd.read_csv(LOCAL_DATA)
    return pd.read_csv(DATA_URL)

def clean_price(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", "").strip()
    if s.lower() in {"ask for price", "nan", "", "n/a"}:
        return np.nan
    # Dataset uses Indian-style lakh/crore values in some entries.
    multiplier = 1.0
    if "crore" in s.lower():
        multiplier = 100.0
    elif "lakh" in s.lower():
        multiplier = 1.0
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    return float(nums[0]) * multiplier if nums else np.nan

def clean_year(value):
    nums = re.findall(r"\d{4}", str(value))
    return int(nums[0]) if nums else np.nan

def clean_kms(value):
    if pd.isna(value):
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(nums[0]) if nums else np.nan

@st.cache_resource
def train_model():
    df = load_data().copy()
    # Accommodate common column names in the supplied Quikr dataset.
    df.columns = [str(c).strip().lower() for c in df.columns]
    expected = {"name", "company", "year", "kms_driven", "fuel_type", "price"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Dataset columns were not as expected. Found: {list(df.columns)}")

    df = df[["name", "company", "year", "kms_driven", "fuel_type", "price"]].copy()
    df["year"] = df["year"].apply(clean_year)
    df["kms_driven"] = df["kms_driven"].apply(clean_kms)
    df["price"] = df["price"].apply(clean_price)
    df["company"] = df["company"].astype(str).str.strip()
    df["fuel_type"] = df["fuel_type"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df = df.replace({"nan": np.nan, "": np.nan})
    df = df.dropna(subset=["name", "company", "year", "kms_driven", "fuel_type", "price"])
    df = df[(df["price"] > 0) & (df["year"].between(1990, 2026)) & (df["kms_driven"] >= 0)]
    # Remove extreme prices to reduce the effect of malformed source rows.
    if len(df) < 20:
        raise ValueError("Not enough valid rows remained after cleaning the dataset.")

    X = df[["name", "company", "year", "kms_driven", "fuel_type"]]
    y = df["price"]
    categorical = ["name", "company", "fuel_type"]
    numeric = ["year", "kms_driven"]

    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, df, mae, r2

st.title("🚗 Used Car Price Predictor")
st.write("Estimate a used car's resale price using the supplied Quikr car dataset and a machine-learning model.")
st.caption("Educational project only: prices are estimates, not an official valuation. Dataset quality and market conditions affect accuracy.")

try:
    model, df, mae, r2 = train_model()
except Exception as exc:
    st.error(f"Could not load or prepare the dataset: {exc}")
    st.info("Check your internet connection, or place quikr_car.csv in the same folder as app.py.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Cleaned training records", f"{len(df):,}")
m2.metric("Test MAE", f"₹{mae:,.0f}")
m3.metric("Test R²", f"{r2:.2f}")
st.caption("Prices are interpreted as Indian rupees after cleaning comma-separated values. Treat estimates cautiously because the source data is old and contains inconsistent listings.")

st.divider()
left, right = st.columns([1, 1])
with left:
    st.subheader("Enter car details")
    companies = sorted(df["company"].dropna().unique().tolist())
    company = st.selectbox("Manufacturer", companies)
    company_rows = df[df["company"] == company]
    models = sorted(company_rows["name"].dropna().unique().tolist())
    car_name = st.selectbox("Car model / variant", models)
    fuels = sorted(df["fuel_type"].dropna().unique().tolist())
    fuel = st.selectbox("Fuel type", fuels)
    min_year = int(max(1990, df["year"].min()))
    max_year = int(min(2026, df["year"].max()))
    year = st.slider("Manufacturing year", min_year, max_year, min(max_year, 2018))
    max_kms = int(max(10000, min(300000, np.nanpercentile(df["kms_driven"], 98))))
    kms = st.number_input("Kilometres driven", min_value=0, max_value=1000000, value=45000, step=5000)

    if st.button("Predict estimated price", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "name": car_name, "company": company, "year": year,
            "kms_driven": float(kms), "fuel_type": fuel
        }])
        estimate = float(model.predict(sample)[0])
        st.session_state["prediction"] = estimate
        st.session_state["prediction_input"] = (car_name, company, fuel, year, kms)

with right:
    st.subheader("Prediction")
    if "prediction" in st.session_state:
        value = st.session_state["prediction"]
        st.metric("Estimated price", f"₹{value:,.0f}")
        st.write("This estimate is in Indian rupees, based on the cleaned prices in the source dataset.")
        st.write("Use it as a learning/demo estimate. Confirm the source data's price units and compare current local listings before making a purchase.")
    else:
        st.info("Enter the car details and click **Predict estimated price** to see the estimate.")

st.divider()
with st.expander("Preview the dataset used"):
    st.dataframe(df.head(20), use_container_width=True)
with st.expander("About this project"):
    st.markdown("""
    - **Dataset:** Quikr used-car CSV linked in the project brief.
    - **Preprocessing:** extracts numeric year, kilometres driven and price; removes unusable rows.
    - **Model:** scikit-learn Random Forest regression with one-hot encoding for categorical fields.
    - **Evaluation:** hold-out test split with MAE and R².
    - **Limitations:** the original dataset may contain inconsistent price formats, outdated listings and missing values. This is a student project, not a commercial valuation tool.
    """)
