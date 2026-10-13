
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AutoValue",
    page_icon="🚘",
    layout="wide",
)

st.title("🚘 AutoValue — Used Car Price Predictor")
st.caption(
    "Machine Learning Project | Extra Trees Regression | Prices in USD"
)

DATA_PATH = Path(__file__).resolve().parent / "used_cars.csv"

CATS = [
    "brand",
    "model",
    "fuel_type",
    "engine",
    "transmission",
    "ext_col",
    "int_col",
    "accident",
    "clean_title",
]

NUMS = ["model_year", "milage"]
FEATURES = CATS + NUMS
COLS = FEATURES + ["price"]


# =====================================================
# 2. HELPER FUNCTION
# =====================================================

def numeric(value):
    """Convert prices and mileage strings to numeric values."""
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        str(value).replace(",", ""),
    )

    return float(match.group(0)) if match else np.nan


# =====================================================
# 3. LOAD AND CLEAN DATASET
# =====================================================

@st.cache_data
def load_data(path, modified):
    df = pd.read_csv(path)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    missing = [col for col in COLS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[COLS].copy()

    # Convert numeric columns
    df["price"] = df["price"].apply(numeric)
    df["milage"] = df["milage"].apply(numeric)

    df["model_year"] = pd.to_numeric(
        df["model_year"],
        errors="coerce",
    )

    # Safely clean categorical columns
    for col in CATS:
        df[col] = df[col].map(
            lambda value: (
                str(value).strip()
                if pd.notna(value) and str(value).strip()
                else np.nan
            )
        ).astype(object)

    # Remove records without valid target prices
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Remove invalid mileage
    df = df[
        df["milage"].isna()
        | (df["milage"] >= 0)
    ]

    # Remove invalid model years
    df = df[
        df["model_year"].isna()
        | df["model_year"].between(1900, 2035)
    ]

    df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid records remain after cleaning the dataset."
        )

    return df


# =====================================================
# 4. TRAIN MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def build_model(path, modified):
    df = load_data(path, modified)

    X = df[FEATURES].copy()
    y = df["price"].copy()

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
            ),
        ),
    ])

    numerical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "categorical",
            categorical_transformer,
            CATS,
        ),
        (
            "numerical",
            numerical_transformer,
            NUMS,
        ),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            ExtraTreesRegressor(
                n_estimators=220,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions),
        "train": len(X_train),
        "test": len(X_test),
    }

    comparison = pd.DataFrame({
        "Actual price": y_test.to_numpy(),
        "Predicted price": predictions,
    })

    return df, pipeline, metrics, comparison


# =====================================================
# 5. LOAD MODEL SAFELY
# =====================================================

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place used_cars.csv "
        "in the same folder as app.py."
    )
    st.stop()

try:
    df, model, metrics, comparison = build_model(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

except Exception as error:
    st.error(f"Could not prepare the dataset/model: {error}")
    st.stop()


# =====================================================
# 6. DASHBOARD SUMMARY
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Usable Listings",
    f"{len(df):,}",
)

col2.metric(
    "Test MAE",
    f"${metrics['MAE']:,.0f}",
)

col3.metric(
    "Test R²",
    f"{metrics['R2']:.3f}",
)

st.caption(
    "Dataset prices are in USD and mileage is in miles. "
    "Predictions are estimates, not guaranteed sale prices."
)

predict_tab, analysis_tab, about_tab = st.tabs([
    "💰 Predict Price",
    "📊 Model Evaluation",
    "ℹ️ About",
])


# =====================================================
# 7. PRICE PREDICTION INTERFACE
# =====================================================

with predict_tab:
    st.subheader("Enter Vehicle Details")

    left, right = st.columns(2)

    # Get valid dropdown options
    brands = sorted(
        df["brand"].dropna().astype(str).unique().tolist()
    )

    if not brands:
        st.error("No valid brands were found in the dataset.")
        st.stop()

    with left:
        brand = st.selectbox("Brand", brands)

        brand_rows = df[
            df["brand"].astype(str) == brand
        ]

        models = sorted(
            brand_rows["model"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not models:
            models = sorted(
                df["model"].dropna().astype(str).unique().tolist()
            )

        model_name = st.selectbox("Car Model", models)

        valid_years = df["model_year"].dropna()

        year_default = (
            int(valid_years.median())
            if not valid_years.empty
            else 2020
        )

        year = st.number_input(
            "Model Year",
            min_value=1900,
            max_value=2035,
            value=max(1900, min(2035, year_default)),
            step=1,
        )

        valid_mileage = df["milage"].dropna()

        mileage_default = (
            int(valid_mileage.median())
            if not valid_mileage.empty
            else 50000
        )

        mileage = st.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=2000000,
            value=max(0, min(2000000, mileage_default)),
            step=1000,
        )

        fuel_options = sorted(
            df["fuel_type"].dropna().astype(str).unique().tolist()
        )

        fuel = st.selectbox("Fuel Type", fuel_options)

        transmission_options = sorted(
            df["transmission"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        transmission = st.selectbox(
            "Transmission",
            transmission_options,
        )

    with right:
        engine_options = sorted(
            df["engine"].dropna().astype(str).unique().tolist()
        )

        engine = st.selectbox("Engine", engine_options)

        exterior_options = sorted(
            df["ext_col"].dropna().astype(str).unique().tolist()
        )

        exterior = st.selectbox(
            "Exterior Colour",
            exterior_options,
        )

        interior_options = sorted(
            df["int_col"].dropna().astype(str).unique().tolist()
        )

        interior = st.selectbox(
            "Interior Colour",
            interior_options,
        )

        accident_options = sorted(
            df["accident"].dropna().astype(str).unique().tolist()
        )

        accident = st.selectbox(
            "Accident History",
            accident_options,
        )

        title_options = sorted(
            df["clean_title"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        clean_title = st.selectbox(
            "Clean Title",
            title_options,
        )

    if st.button(
        "Estimate Car Price",
        type="primary",
        use_container_width=True,
    ):
        sample = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "model_year": year,
            "milage": mileage,
            "fuel_type": fuel,
            "engine": engine,
            "transmission": transmission,
            "ext_col": exterior,
            "int_col": interior,
            "accident": accident,
            "clean_title": clean_title,
        }], columns=FEATURES)

        try:
            estimate = max(
                0,
                float(model.predict(sample)[0]),
            )

            st.success(
                f"Estimated Listing Price: ${estimate:,.0f} USD"
            )

            st.caption(
                "This is an educational machine-learning estimate. "
                "Actual market prices vary with condition and location."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =====================================================
# 8. MODEL EVALUATION
# =====================================================

with analysis_tab:
    st.subheader("Model Performance")

    a, b, c = st.columns(3)

    a.metric(
        "MAE",
        f"${metrics['MAE']:,.2f}",
    )

    b.metric(
        "RMSE",
        f"${metrics['RMSE']:,.2f}",
    )

    c.metric(
        "R² Score",
        f"{metrics['R2']:.4f}",
    )

    st.write(
        f"Training records: {metrics['train']:,} | "
        f"Testing records: {metrics['test']:,}"
    )

    st.subheader("Actual vs Predicted Prices")

    st.scatter_chart(
        comparison,
        x="Actual price",
        y="Predicted price",
    )

    st.subheader("Price Distribution")

    price_limit = df["price"].quantile(0.99)

    prices = df.loc[
        df["price"] <= price_limit,
        "price",
    ]

    price_bins = pd.cut(
        prices,
        bins=20,
    ).value_counts().sort_index()

    st.bar_chart(price_bins)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )


# =====================================================
# 9. PROJECT INFORMATION
# =====================================================

with about_tab:
    st.subheader("About AutoValue")

    st.markdown("""
    **AutoValue** predicts used-car listing prices using machine learning.

    **Project workflow**
    1. Load the used-car dataset.
    2. Clean price, mileage and categorical values.
    3. Impute missing values and encode categories.
    4. Train an Extra Trees regression model.
    5. Evaluate the model using MAE, RMSE and R².
    6. Estimate a price from user-entered vehicle details.

    **Limitations**
    - Predictions depend on the quality and coverage of the dataset.
    - The dataset uses US dollars and miles.
    - Predictions may be less reliable for rare vehicle combinations.
    - This project is for educational use, not professional valuation.
    """)

st.markdown("---")
st.caption("AutoValue | Machine Learning Engineering Project")
