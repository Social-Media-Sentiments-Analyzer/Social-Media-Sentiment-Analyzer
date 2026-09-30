import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

file_path = "data/processed/cleaned_data.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total tweets:", len(df))


# ==========================================
# 2. SELECT INPUT AND TARGET
# ==========================================

X = df["clean_text"]
y = df["airline_sentiment"]


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining tweets:", len(X_train))
print("Testing tweets:", len(X_test))


# ==========================================
# 4. TF-IDF VECTORIZATION
# ==========================================

print("\nConverting text into numerical features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# ==========================================
# 5. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 50)

print("\nSaved files:")
print("models/sentiment_model.pkl")
print("models/tfidf_vectorizer.pkl")