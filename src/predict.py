import re
import joblib


# Load trained model and TF-IDF vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Clean text in the same way as our training data
def clean_text(text):
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("=" * 50)
print("SOCIAL MEDIA SENTIMENT ANALYZER")
print("=" * 50)

while True:

    tweet = input("\nEnter a tweet (or type 'exit' to quit): ")

    if tweet.lower() == "exit":
        print("\nProgram closed.")
        break

    # Clean tweet
    cleaned_tweet = clean_text(tweet)

    # Convert text to TF-IDF
    tweet_tfidf = vectorizer.transform([cleaned_tweet])

    # Predict sentiment
    prediction = model.predict(tweet_tfidf)[0]

    print("\nPredicted sentiment:", prediction.upper())