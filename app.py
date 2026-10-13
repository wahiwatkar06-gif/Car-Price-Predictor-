
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="AutoValue", page_icon="🚘", layout="wide")
st.title("🚘 AutoValue — Used Car Price Predictor")
st.caption("Machine Learning project | Extra Trees Regression | Prices in USD")

DATA_PATH = Path(__file__).resolve().parent / "used_cars.csv"

CATS = [
    "brand", "model", "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title"
]
NUMS = ["model_year", "milage"]
FEATURES = CATS + NUMS
COLS = FEATURES + ["price"]

def numeric(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else np.nan

@st.cache_data
def load_data(path, modified):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[COLS].copy()
    df["price"] = df["price"].apply(numeric)
    df["milage"] = df["milage"].apply(numeric)
    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    for c in CATS:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df = df[df["milage"].isna() | (df["milage"] >= 0)]
    df = df[df["model_year"].isna() | df["model_year"].between(1900, 2035)]
    return df.drop_duplicates().reset_index(drop=True)

@st.cache_resource
def build_model(path, modified):
    df = load_data(path, modified)
    X, y = df[FEATURES], df["price"]
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
        ]), CATS),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), NUMS)
    ])
    pipe = Pipeline([
        ("prep", pre),
        ("model", ExtraTreesRegressor(
            n_estimators=220, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R²": r2_score(y_test, pred),
        "train": len(X_train),
        "test": len(X_test)
    }
    compare = pd.DataFrame({"Actual price": y_test, "Predicted price": pred})
    return df, pipe, metrics, compare

if not DATA_PATH.exists():
    st.error("Place used_cars.csv in the same folder as app.py.")
    st.stop()

try:
    df, model, metrics, comparison = build_model(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
except Exception as e:
    st.error(f"Could not prepare the dataset/model: {e}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Listings", f"{len(df):,}")
m2.metric("Test MAE", f"${metrics["MAE"]:,.0f}")
m3.metric("Test R²", f"{metrics["R²"]:.3f}")
st.caption("The dataset uses US dollars and miles. Estimates are not guaranteed sale prices.")

predict_tab, analysis_tab, about_tab = st.tabs(
    ["💰 Predict price", "📊 Model evaluation", "ℹ️ About"]
)

with predict_tab:
    st.subheader("Vehicle details")
    left, right = st.columns(2)

    with left:
        brands = sorted(df["brand"].dropna().astype(str).unique())
        brand = st.selectbox("Brand", brands)
        models = sorted(df.loc[df["brand"].astype(str) == brand, "model"]
                        .dropna().astype(str).unique())
        model_name = st.selectbox("Model", models or sorted(df["model"].dropna().astype(str).unique()))
        years = df["model_year"].dropna()
        year_default = int(years.median()) if len(years) else 2020
        year = st.number_input("Model year", min_value=1900, max_value=2035,
                               value=max(1900, min(2035, year_default)))
        mile_vals = df["milage"].dropna()
        mile_default = int(mile_vals.median()) if len(mile_vals) else 50000
        mileage = st.number_input("Mileage (miles)", min_value=0,
                                  value=max(0, mile_default), step=1000)
        fuel = st.selectbox("Fuel type", sorted(df["fuel_type"].dropna().astype(str).unique()))

    with right:
        transmission = st.selectbox("Transmission", sorted(df["transmission"].dropna().astype(str).unique()))
        engine = st.selectbox("Engine", sorted(df["engine"].dropna().astype(str).unique()))
        ext = st.selectbox("Exterior colour", sorted(df["ext_col"].dropna().astype(str).unique()))
        interior = st.selectbox("Interior colour", sorted(df["int_col"].dropna().astype(str).unique()))
        accident = st.selectbox("Accident history", sorted(df["accident"].dropna().astype(str).unique()))
        title = st.selectbox("Clean title", sorted(df["clean_title"].dropna().astype(str).unique()))

    if st.button("Estimate car price", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "brand": brand, "model": model_name, "model_year": year,
            "milage": mileage, "fuel_type": fuel, "engine": engine,
            "transmission": transmission, "ext_col": ext, "int_col": interior,
            "accident": accident, "clean_title": title
        }], columns=FEATURES)
        estimate = max(0, float(model.predict(sample)[0]))
        st.success(f"Estimated listing price: **${estimate:,.0f} USD**")
        st.caption("Educational estimate only; actual prices vary by location, condition and market.")

with analysis_tab:
    a, b, c = st.columns(3)
    a.metric("MAE", f"${metrics["MAE"]:,.2f}")
    b.metric("RMSE", f"${metrics["RMSE"]:,.2f}")
    c.metric("R²", f"{metrics["R²"]:.4f}")
    st.write(f"Training records: {metrics["train"]:,} | Test records: {metrics["test"]:,}")
    st.subheader("Actual vs predicted prices")
    st.scatter_chart(comparison, x="Actual price", y="Predicted price")
    st.subheader("Price distribution")
    prices = df.loc[df["price"] <= df["price"].quantile(0.99), "price"]
    st.bar_chart(pd.cut(prices, bins=20).value_counts().sort_index())
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

with about_tab:
    st.markdown("""
    **Workflow**
    1. Load and clean the listings.
    2. Impute missing values and encode categorical features.
    3. Train an Extra Trees regression model.
    4. Evaluate using MAE, RMSE and R² on a held-out test split.
    5. Estimate a price from user-entered vehicle specifications.

    **Limitations:** Results reflect this dataset and its market. USD prices and miles
    should not be interpreted as Indian market values. Predictions are not appraisals.
    """)
