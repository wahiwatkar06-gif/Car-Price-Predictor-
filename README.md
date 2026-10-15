# Indian Fake News Detector

An educational NLP application trained on an Indian fake-news dataset.

## Dataset
Indian Fake News Dataset (IFND):
https://github.com/sonalgarg174/Dataset

Check the original repository for research-use conditions and attribution.

## Algorithms
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- SGD Classifier

## Features
- News headline/article classification
- Model comparison dashboard
- Batch CSV predictions
- Downloadable prediction results

## Technology
Python, pandas, scikit-learn, TF-IDF, joblib and Streamlit.

## Run locally
Install packages:

    pip install -r requirements.txt

Start the app:

    streamlit run app.py

Keep app.py and fake_news_model.joblib in the same directory.

## Limitations
The classifier detects learned text patterns. It does not independently
fact-check claims or guarantee that a news article is true or false.
