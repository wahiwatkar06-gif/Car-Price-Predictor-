import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Indian Fake News Detector",
    page_icon="📰",
    layout="wide"
)

MODEL_PATH = "fake_news_model.joblib"


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "fake_news_model.joblib is missing. "
            "Upload it beside app.py in your GitHub repository."
        )
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


model = bundle["best_model"]
model_name = bundle["best_model_name"]
metrics = pd.DataFrame(bundle["metrics"])
labels = bundle["labels"]

st.title("📰 Indian Fake News Detector")
st.write(
    "An educational machine-learning application that classifies "
    "Indian news text using a model trained on an Indian news dataset."
)

st.caption(f"Selected model: {model_name}")

st.warning(
    "This tool does not independently verify facts. A prediction is "
    "not proof that a story is true or false. Check the original source, "
    "date, evidence, and independent reporting before sharing news."
)

page = st.sidebar.radio(
    "Navigation",
    ["News Detector", "Model Comparison", "Batch Prediction", "About"]
)


def classify(text):
    predicted_label = int(model.predict([text])[0])
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(probabilities[predicted_label])

    return predicted_label, confidence


def extract_news_column(data):
    # Normalize column names for uploaded CSV files
    normalized = {
        str(col).strip().lower(): col
        for col in data.columns
    }

    for candidate in ["news", "statement", "text", "content", "headline", "title"]:
        if candidate in normalized:
            return data[normalized[candidate]].fillna("").astype(str)

    return None


# ---------------- NEWS DETECTOR ----------------

if page == "News Detector":

    st.header("Analyze Indian News")

    with st.form("news_form"):
        news_text = st.text_area(
            "Paste a headline or article",
            height=240,
            placeholder="Paste the news text here..."
        )

        submitted = st.form_submit_button(
            "Analyze News",
            use_container_width=True
        )

    if submitted:

        if len(news_text.strip()) < 15:
            st.error("Please enter a longer headline or article.")
        else:
            label, confidence = classify(news_text.strip())

            st.subheader("Prediction")

            if label == 0:
                st.error("Model prediction: FAKE")
            else:
                st.success("Model prediction: REAL")

            if confidence is not None:
                st.metric(
                    "Model confidence for predicted class",
                    f"{confidence * 100:.2f}%"
                )
                st.caption(
                    "This is the classifier's estimated class probability, "
                    "not a verified probability that the story is false or true."
                )
            else:
                st.info(
                    "The selected model does not provide class probabilities. "
                    "No confidence percentage is displayed."
                )

            st.write("**Text analyzed:**")
            st.write(news_text.strip())

            st.markdown(
                "**Verification tip:** Check the publisher, date, author, "
                "original documents, and independent credible sources."
            )


# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.header("Compare Four ML Algorithms")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    chart_columns = [
        col for col in [
            "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
        ]
        if col in metrics.columns
    ]

    if chart_columns:
        st.bar_chart(
            metrics.set_index("Model")[chart_columns]
        )

    st.success(f"Best model by F1-score: {model_name}")


# ---------------- BATCH PREDICTION ----------------

elif page == "Batch Prediction":

    st.header("Analyze Multiple News Articles")

    st.write(
        "Upload a CSV with a news text column such as Statement, "
        "text, news, title, or content."
    )

    uploaded_csv = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_csv is not None:

        try:
            data = pd.read_csv(uploaded_csv)
            text_series = extract_news_column(data)

            if text_series is None:
                st.error(
                    "No text column found. Expected Statement, text, "
                    "news, title, headline, or content."
                )
            else:
                st.dataframe(data.head(), use_container_width=True)

                if st.button("Predict All Articles"):

                    valid = text_series.str.strip().str.len() >= 15
                    output = data.loc[valid].copy()
                    texts = text_series.loc[valid].tolist()

                    if not texts:
                        st.error("No usable news articles found.")
                    else:
                        predictions = model.predict(texts)

                        output["Prediction"] = [
                            labels[int(value)]
                            for value in predictions
                        ]

                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(texts)
                            output["Model_Confidence_Percent"] = [
                                round(
                                    float(probs[i][int(predictions[i])]) * 100,
                                    2
                                )
                                for i in range(len(predictions))
                            ]

                        st.subheader("Prediction Results")
                        st.dataframe(output, use_container_width=True)

                        st.download_button(
                            "Download prediction results",
                            data=output.to_csv(index=False).encode("utf-8"),
                            file_name="indian_fake_news_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")


# ---------------- ABOUT ----------------

elif page == "About":

    st.header("About This Project")

    st.markdown("""
    **Project:** Indian Fake News Detector

    **Dataset:** Indian Fake News Dataset (IFND)

    **Algorithms compared:**
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Linear Support Vector Machine
    4. SGD Classifier

    **Text processing:** TF-IDF word and bigram features.

    **Technology:** Python, pandas, scikit-learn, joblib and Streamlit.

    **Limitations:** The model learns patterns in its training dataset.
    It cannot independently verify claims or guarantee truthfulness.
    """)

    st.write("Selected model:", model_name)


st.divider()
st.caption("Indian Fake News Detector | Academic ML Project")
