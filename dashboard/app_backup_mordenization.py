import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import re

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
) 

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/cleaned_data.csv"
    )


@st.cache_resource
def load_model():

    model = joblib.load(
        "models/sentiment_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


df = load_data()
model, vectorizer = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# HEADER
# =========================================================

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")
st.sidebar.caption("🎓 BCA Final Year Project")

page = st.sidebar.selectbox(
    "Choose a section:",
    [
        "🏠 Dashboard",
        "🤖 Sentiment Analyzer",
        "✈️ Airline Analysis",
        "📋 Dataset",
        "📤 Upload CSV",
        "ℹ️ About Project",
        "📊 Model Performance"
    ]
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("📊 Social Media Sentiment Analysis")

    st.write(
        "Interactive dashboard for analyzing social media sentiment "
        "using Machine Learning and Natural Language Processing (NLP)."
    )

    st.divider()

    st.header("📈 Dashboard Overview")

    total = len(df)

    positive = len(
        df[df["airline_sentiment"] == "positive"]
    )

    neutral = len(
        df[df["airline_sentiment"] == "neutral"]
    )

    negative = len(
        df[df["airline_sentiment"] == "negative"]
    )


    positive_percent = (
        positive / total * 100
    )

    neutral_percent = (
        neutral / total * 100
    )

    negative_percent = (
        negative / total * 100
    )


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📱 Total Tweets",
        f"{total:,}"
    )

    col2.metric(
        "😊 Positive",
        f"{positive_percent:.1f}%"
    )

    col3.metric(
        "😐 Neutral",
        f"{neutral_percent:.1f}%"
    )

    col4.metric(
        "😡 Negative",
        f"{negative_percent:.1f}%"
    )


    st.divider()


    # =====================================================
    # SENTIMENT DATA
    # =====================================================

    sentiment_counts = (
        df["airline_sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]


    # =====================================================
    # PIE CHART
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            title="Overall Sentiment Distribution",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # BAR CHART
    # =====================================================

    with col2:

        fig2 = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Tweets by Sentiment",
            text="Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )
    st.divider()

    st.subheader("✈️ Sentiment by Airline")

    airline_sentiment = (
        df.groupby(
            ["airline", "airline_sentiment"]
        )
        .size()
        .reset_index(name="Count")
    )

    fig3 = px.bar(
        airline_sentiment,
        x="airline",
        y="Count",
        color="airline_sentiment",
        barmode="group",
        title="Sentiment Distribution by Airline",
        text="Count"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )
    st.divider()

    st.subheader("💡 Key Insights")

    most_common_sentiment = (
        df["airline_sentiment"]
        .value_counts()
        .idxmax()
    )

    most_common_airline = (
        df["airline"]
        .value_counts()
        .idxmax()
    )

    st.write(
        f"• The most common sentiment in the dataset is "
        f"**{most_common_sentiment.capitalize()}**."
    )

    st.write(
        f"• **{most_common_airline}** has the highest number "
        f"of tweets in the dataset."
    )
# =========================================================
# SENTIMENT ANALYZER
# =========================================================

elif page == "🤖 Sentiment Analyzer":
    st.header("🤖 Analyze a Social Media Post")

    st.write(
        "Enter a tweet or social media post below."
    )

    tweet = st.text_area(
        "Your post:",
        placeholder="Example: I absolutely love this airline!",
        height=150
    )

    if st.button(
        "🔍 Analyze Sentiment",
        type="primary"
    ):

        if not tweet.strip():

            st.warning(
                "Please enter a post first."
            )

        else:

            cleaned = clean_text(tweet)

            # TF-IDF
            tweet_tfidf = vectorizer.transform(
                [cleaned]
            )

            # Prediction
            prediction = model.predict(
                tweet_tfidf
            )[0]

            # Probabilities
            probabilities = model.predict_proba(
                tweet_tfidf
            )[0]

            classes = model.classes_

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            # RESULT
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

            # CONFIDENCE
            confidence = max(probabilities) * 100

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # PROBABILITY CHART
            probability_df = pd.DataFrame(
                {
                    "Sentiment": classes,
                    "Probability": probabilities * 100
                }
            )

            fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Probability",
                title="Prediction Probability",
                text_auto=".1f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# AIRLINE ANALYSIS
# =========================================================

elif page == "✈️ Airline Analysis":

    st.header(
        "✈️ Sentiment Analysis by Airline"
    )


    # Crosstab

    airline_data = pd.crosstab(
        df["airline"],
        df["airline_sentiment"]
    )


    st.subheader(
        "Sentiment Counts by Airline"
    )


    st.dataframe(
        airline_data,
        use_container_width=True
    )


    # =====================================================
    # CHART
    # =====================================================

    fig = px.bar(
        airline_data,
        x=airline_data.index,
        y=[
            "negative",
            "neutral",
            "positive"
        ],
        title="Sentiment Distribution by Airline",
        barmode="group"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =====================================================
    # AIRLINE SELECTOR
    # =====================================================

    st.subheader(
        "🔎 Analyze Individual Airline"
    )


    airline = st.selectbox(
        "Select airline:",
        sorted(
            df["airline"].unique()
        )
    )


    selected = df[
        df["airline"] == airline
    ]


    selected_counts = (
        selected["airline_sentiment"]
        .value_counts()
        .reset_index()
    )


    selected_counts.columns = [
        "Sentiment",
        "Count"
    ]


    fig2 = px.pie(
        selected_counts,
        names="Sentiment",
        values="Count",
        title=f"Sentiment Distribution - {airline}"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# DATASET
# =========================================================

elif page == "📋 Dataset":

    st.header(
        "📋 Dataset Explorer"
    )


    st.write(
        f"Dataset contains **{len(df):,} tweets**."
    )


    # Search

    search = st.text_input(
        "🔎 Search tweets:"
    )


    if search:

        filtered = df[
            df["clean_text"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    else:

        filtered = df


    st.write(
        f"Showing {len(filtered):,} tweets"
    )


    st.dataframe(
        filtered,
        use_container_width=True,
        height=500
    )

elif page == "📤 Upload CSV":
    st.header("📤 Upload CSV for Sentiment Analysis")

    st.write(
        "Upload a CSV file containing social media posts "
        "and the model will automatically predict their sentiment."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully!")

        st.write(
            f"Rows found: {len(uploaded_df):,}"
        )

        st.subheader("Preview")

        st.dataframe(
            uploaded_df.head(10),
            use_container_width=True
        )

        text_column = st.selectbox(
            "Select the column containing the posts:",
            uploaded_df.columns
        )

        if st.button(
            "🤖 Analyze CSV",
            type="primary"
        ):

            texts = uploaded_df[text_column].fillna("").astype(str)

            cleaned_texts = texts.apply(clean_text)

            tfidf_data = vectorizer.transform(
                cleaned_texts
            )

            predictions = model.predict(
                tfidf_data
            )

            probabilities = model.predict_proba(
                tfidf_data
            )

            uploaded_df["predicted_sentiment"] = predictions

            uploaded_df["confidence"] = (
                probabilities.max(axis=1) * 100
            ).round(2)

            st.success(
                "Sentiment analysis completed!"
            )

            st.subheader(
                "📊 Analysis Results"
            )

            st.dataframe(
                uploaded_df,
                use_container_width=True
            )

            sentiment_counts = (
                uploaded_df["predicted_sentiment"]
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
                title="Predicted Sentiment Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            csv_data = uploaded_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Analyzed CSV",
                csv_data,
                "analyzed_sentiment.csv",
                "text/csv"
            )

# =========================================================
# FOOTER
# =========================================================
elif page == "ℹ️ About Project":
    st.header("ℹ️ About the Project")

    st.markdown("""
    ## 📊 Social Media Sentiment Analysis

    This project uses **Machine Learning and Natural Language Processing (NLP)**
    to analyze social media posts and classify them into:

    - 😊 **Positive**
    - 😐 **Neutral**
    - 😡 **Negative**

    ### 🎯 Project Objective

    The main objective is to automatically understand the sentiment expressed
    in social media posts and present the results through an interactive
    analytics dashboard.

    ### 🛠️ Technologies Used

    - **Python**
    - **Pandas** — Data processing
    - **Scikit-learn** — Machine Learning
    - **TF-IDF** — Text feature extraction
    - **Logistic Regression** — Sentiment classification
    - **Plotly** — Interactive visualizations
    - **Streamlit** — Web dashboard

    ### 🤖 Machine Learning Model

    The project uses:

    **TF-IDF + Logistic Regression**

    The dataset is divided into training and testing data using an
    **80:20 split**.

    ### 📈 Model Performance

    The trained model achieved approximately:

    **80% Accuracy**

    ### 📂 Dataset

    The project uses the **Twitter US Airline Sentiment** dataset,
    containing approximately **14,640 tweets**.

    ### 🚀 Features

    - Interactive sentiment dashboard
    - Individual tweet sentiment prediction
    - Airline-wise sentiment analysis
    - Dataset explorer
    - CSV upload and batch sentiment analysis
    - Prediction confidence
    - Downloadable analyzed CSV

    ### 🎓 Project Type

    **BCA Final Year Project**

    **Social Media Sentiment Analysis & Analytics Dashboard**
    """)

elif page == "📊 Model Performance":
    st.header("📊 Model Performance")

    st.write(
        "Performance metrics of the trained Logistic Regression "
        "sentiment classification model."
    )

    test_data = pd.read_csv(
        "data/processed/cleaned_data.csv"
    )

    X = test_data["clean_text"]
    y = test_data["airline_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_test_tfidf = vectorizer.transform(X_test)

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    st.subheader("📈 Classification Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1-Score",
        f"{f1 * 100:.2f}%"
    )

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "negative",
            "neutral",
            "positive"
        ]
    )

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted Negative",
            "Predicted Neutral",
            "Predicted Positive"
        ],
        y=[
            "Actual Negative",
            "Actual Neutral",
            "Actual Positive"
        ],
        title="Confusion Matrix",
        labels={
            "x": "Predicted Sentiment",
            "y": "Actual Sentiment",
            "color": "Number of Tweets"
        }
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )
    

st.divider()

st.caption(
    "Social Media Sentiment Analysis | "
    "BCA Final Year Project | "
    "Python • Machine Learning • Streamlit"
)