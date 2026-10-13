# AutoValue — Used Car Price Predictor

A Streamlit machine-learning app using the supplied `used_cars.csv` dataset.

## Features
- Data cleaning and preprocessing
- Extra Trees regression model
- Interactive used-car price prediction
- MAE, RMSE, and R² evaluation
- Actual-vs-predicted chart and dataset preview

## Dataset columns
`brand`, `model`, `model_year`, `milage`, `fuel_type`, `engine`, `transmission`,
`ext_col`, `int_col`, `accident`, `clean_title`, `price`.

**Units:** Prices are in US dollars (USD); mileage is in miles.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
