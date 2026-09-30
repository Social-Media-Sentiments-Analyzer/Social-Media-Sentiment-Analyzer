import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import re


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")


@st.cache_resource
def load_model():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


df = load_data()
model, vectorizer = load_model()


# =========================================================
# TEXT CLEANING FUNCTION
# =========================================================

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


# =========================================================
# TITLE
# =========================================================

st.title("📊 Social Media Sentiment Analysis")

st.markdown(
    """
    **Machine Learning powered dashboard for analyzing social media sentiment**
    
    This system classifies posts into **Positive, Neutral, or Negative**
    using TF-IDF and Logistic Regression.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard")

page = st.sidebar.radio(
    "Select a section:",
    [
        "Overview",
        "Sentiment Analysis",
        "Airline Analysis"
    ]
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.header("📈 Dataset Overview")

    total_tweets = len(df)

    positive = len(df[df["airline_sentiment"] == "positive"])
    neutral = len(df[df["airline_sentiment"] == "neutral"])
    negative = len(df[df["airline_sentiment"] == "negative"])

    positive_percent = positive / total_tweets * 100
    neutral_percent = neutral / total_tweets * 100
    negative_percent = negative / total_tweets * 100


    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Tweets",
        f"{total_tweets:,}"
    )

    col2.metric(
        "Positive",
        f"{positive_percent:.1f}%"
    )

    col3.metric(
        "Neutral",
        f"{neutral_percent:.1f}%"
    )

    col4.metric(
        "Negative",
        f"{negative_percent:.1f}%"
    )


    st.divider()


    # SENTIMENT CHART

    sentiment_counts = (
        df["airline_sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]


    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Overall Sentiment Distribution",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # BAR CHART

    fig2 = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        title="Number of Tweets by Sentiment",
        text="Count"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# SENTIMENT ANALYSIS PAGE
# =========================================================

elif page == "Sentiment Analysis":

    st.header("🤖 Analyze a Social Media Post")

    st.write(
        "Enter a social media post below and the machine-learning "
        "model will predict its sentiment."
    )


    tweet = st.text_area(
        "Enter your post:",
        placeholder="Example: I absolutely love this airline!"
    )


    if st.button("🔍 Analyze Sentiment"):

        if tweet.strip() == "":
            st.warning("Please enter some text first.")

        else:

            cleaned_tweet = clean_text(tweet)

            tweet_tfidf = vectorizer.transform(
                [cleaned_tweet]
            )

            prediction = model.predict(
                tweet_tfidf
            )[0]


            st.divider()

            st.subheader("Prediction")

            if prediction == "positive":

                st.success(
                    "😊 POSITIVE SENTIMENT"
                )

            elif prediction == "negative":

                st.error(
                    "😡 NEGATIVE SENTIMENT"
                )

            else:

                st.info(
                    "😐 NEUTRAL SENTIMENT"
                )


            # Probability

            probabilities = model.predict_proba(
                tweet_tfidf
            )[0]

            classes = model.classes_

            probability_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities
            })

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            )


            st.subheader("Prediction Confidence")

            fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Probability",
                text_auto=".1f",
                title="Model Prediction Probability (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# AIRLINE ANALYSIS PAGE
# =========================================================

elif page == "Airline Analysis":

    st.header("✈️ Sentiment by Airline")


    airline_sentiment = pd.crosstab(
        df["airline"],
        df["airline_sentiment"]
    )


    st.dataframe(
        airline_sentiment,
        use_container_width=True
    )


    # Stacked bar chart

    fig = px.bar(
        airline_sentiment,
        x=airline_sentiment.index,
        y=[
            "negative",
            "neutral",
            "positive"
        ],
        title="Sentiment Distribution by Airline",
        barmode="stack"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Airline selector

    selected_airline = st.selectbox(
        "Select an airline:",
        sorted(df["airline"].unique())
    )


    filtered = df[
        df["airline"] == selected_airline
    ]


    st.subheader(
        f"Analysis for {selected_airline}"
    )


    counts = (
        filtered["airline_sentiment"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Sentiment",
        "Count"
    ]


    fig2 = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        title=f"Sentiment Distribution - {selected_airline}"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Social Media Sentiment Analysis | "
    "BCA Final Year Project | "
    "Python + Machine Learning + Streamlit"
)