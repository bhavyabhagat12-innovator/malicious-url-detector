import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import extract_features

from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "instagram.com", "twitter.com",
    "x.com", "linkedin.com", "github.com", "microsoft.com", "apple.com",
    "amazon.com", "amazon.in", "wikipedia.org", "netflix.com", "whatsapp.com",
    "stackoverflow.com", "reddit.com", "yahoo.com", "bing.com", "openai.com",
    "anthropic.com", "kaggle.com", "iitm.ac.in", "nptel.ac.in", "flipkart.com",
}


def is_trusted(url: str) -> bool:
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if host.startswith("www."):
        host = host[4:]
    return host in TRUSTED_DOMAINS

st.set_page_config(page_title="Malicious URL Detector", page_icon="🛡️")


@st.cache_resource
def load_model():
    return joblib.load("model/url_model.joblib")


def predict_proba(urls):
    X = np.array([extract_features(u) for u in urls], dtype=np.float32)
    return model.predict_proba(X)


model = load_model()
classes = list(model.classes_)
benign_idx = classes.index("benign")

st.title("🛡️ Malicious URL Detector")
st.caption("ML-based URL analysis. The URL is never opened or visited.")

tab1, tab2, tab3 = st.tabs(["Single URL", "Batch (CSV)", "About"])

with tab1:
    url = st.text_input("Enter a URL", placeholder="http://example.com/login/verify")
    if st.button("Analyze", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            proba = predict_proba([url])[0]
            risk = 1 - proba[benign_idx]
            label = classes[int(np.argmax(proba))]
            if is_trusted(url):
                proba = np.zeros(len(classes))
                proba[benign_idx] = 1.0
                risk = 0.0
                label = "benign"
                st.info("This domain is on the trusted list, so the ML prediction was overridden.")

            if label == "benign":
                st.success(f"Looks safe (predicted: **{label}**)")
            else:
                st.error(f"⚠️ Likely malicious (predicted: **{label}**)")

            st.metric("Risk score", f"{risk * 100:.1f}%")
            st.bar_chart(pd.Series(proba, index=classes, name="Probability"))

            with st.expander("Extracted features"):
                from features import FEATURE_NAMES
                st.dataframe(
                    pd.DataFrame({"feature": FEATURE_NAMES, "value": extract_features(url)}),
                    use_container_width=True,
                )

with tab2:
    st.write("Upload a CSV with a column named **url**.")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        data = pd.read_csv(file)
        if "url" not in data.columns:
            st.error("CSV must contain a 'url' column.")
        else:
            data = data.head(5000)  # keep the free server responsive
            proba = predict_proba(data["url"].astype(str))
            data["prediction"] = [classes[i] for i in proba.argmax(axis=1)]
            data["risk_score"] = (1 - proba[:, benign_idx]).round(3)
            st.dataframe(data, use_container_width=True)
            st.download_button(
                "Download results",
                data.to_csv(index=False).encode("utf-8"),
                "results.csv",
                "text/csv",
            )

with tab3:
    st.markdown(
        """
**How it works:** each URL is converted into ~34 numeric features (length, special
characters, digit ratio, entropy, suspicious keywords, IP-in-host, TLD, etc.) and passed
to a Histogram Gradient Boosting classifier trained on ~650k labeled URLs.

**Limitations:** it only looks at the URL text, so it can produce false positives and
false negatives. Don't use it as your only security check.
        """
    )
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f:
            m = json.load(f)
        st.metric("Test accuracy", f"{m['accuracy'] * 100:.2f}%")
        st.json(m["report"])