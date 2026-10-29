# 🛡️ Malicious URL Detector

ML web app that classifies URLs as benign, phishing, malware or defacement using lexical features.

**Live demo:** https://malicious-url-detector-bbi12.streamlit.app/

## Tech
Python, scikit-learn (HistGradientBoosting), Streamlit, pandas

## Run locally
pip install -r requirements.txt
python train.py        # needs data/malicious_phish.csv from Kaggle
streamlit run app.py

## Dataset
Malicious URLs dataset (Kaggle, sid321axn), ~650k URLs, 4 classes.

## Future work
Domain age / WHOIS features, character n-gram TF-IDF, FastAPI backend, browser extension.
