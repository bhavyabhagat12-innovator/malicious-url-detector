import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_features

print("Loading data...")
df = pd.read_csv("data/malicious_phish.csv").dropna().drop_duplicates(subset="url")
print(df["type"].value_counts())

print("Extracting features (takes a minute or two)...")
X = np.array([extract_features(u) for u in df["url"]], dtype=np.float32)
y = df["type"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training...")
model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.1,
    max_leaf_nodes=63,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc:.4f}\n")
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred, labels=model.classes_))

joblib.dump(model, "model/url_model.joblib", compress=3)
report = classification_report(y_test, pred, output_dict=True)
with open("model/metrics.json", "w") as f:
    json.dump({"accuracy": acc, "report": report}, f, indent=2)
print("\nSaved model/url_model.joblib")