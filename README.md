# Used Car Price Predictor

A Streamlit web app that uses the Quikr used-car dataset to estimate a used car's price with a Random Forest regression pipeline.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

The app downloads the CSV from the source GitHub repository. For offline use, save the CSV as `quikr_car.csv` beside `app.py`.

## Deploy for a public link

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Community Cloud will provide a public URL.

## Notes

The original dataset contains inconsistent formats and may be outdated. The app cleans common numeric fields, but predictions are educational estimates. Confirm price units and compare current local listings before relying on a result.
