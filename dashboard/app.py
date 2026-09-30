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
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #0f172a 100%
        );
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid #1e293b;
    }

    /* Main title */
    h1 {
        font-weight: 800;
        letter-spacing: -1px;
    }

    /* Section headings */
    h2, h3 {
        font-weight: 700;
    }

    /* Premium KPI Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(30, 41, 59, 0.95),
        rgba(15, 23, 42, 0.95)
    );
    border: 1px solid rgba(148, 163, 184, 0.18);
    padding: 22px 20px;
    border-radius: 20px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(148, 163, 184, 0.35);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.38);
}

div[data-testid="stMetricLabel"] {
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
}

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: #334155;
    }
/* Premium Hero Card */
.hero-card {
    padding: 34px 38px;
    margin-bottom: 25px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.95)
        );
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.35);
}

.hero-content h1 {
    font-size: 42px;
    margin: 12px 0 8px 0;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.hero-content p {
    font-size: 17px;
    color: #94a3b8;
    max-width: 750px;
    line-height: 1.6;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
/* Modern Chart Containers */
div[data-testid="stPlotlyChart"] {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 20px;
    padding: 12px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    overflow: hidden;
}

/* Section spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
/* Premium Section Headings */
h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.4px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

/* Better section spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

/* Cleaner horizontal dividers */
hr {
    margin: 28px 0;
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
}
/* =====================================================
   PREMIUM SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0b1120 0%,
        #0f172a 100%
    ) !important;

    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

/* Sidebar title */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
    font-weight: 800;
}

/* Navigation selectbox */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    border-radius: 12px;
}

/* Selectbox itself */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
}

/* Sidebar text */
section[data-testid="stSidebar"] p {
    color: #94a3b8;
}
/* =========================================================
   21B-10 → 21B-12
   GLOBAL DASHBOARD POLISH
   ========================================================= */

/* ---------- 21B-10: Premium Dashboard Layout ---------- */

.block-container {
    max-width: 1450px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* Smooth overall appearance */
.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(59, 130, 246, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 30%,
            rgba(139, 92, 246, 0.06),
            transparent 30%
        ),
        #0f172a;
}

/* Cleaner text */
.stMarkdown,
.stText,
p {
    line-height: 1.6;
}

/* Better spacing between major blocks */
.stElementContainer {
    margin-bottom: 4px;
}

/* ---------- 21B-11: Premium Insight / Content Styling ---------- */

/* Markdown emphasis */
strong {
    color: #f8fafc;
}

/* Make informational boxes feel more premium */
div[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    background: rgba(30, 41, 59, 0.55) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    background: rgba(30, 41, 59, 0.8) !important;
    color: #f8fafc !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: rgba(148, 163, 184, 0.4) !important;
    background: rgba(51, 65, 85, 0.9) !important;
}

/* ---------- 21B-12: Premium Footer / Status Area ---------- */

/* Footer area */
footer {
    background: transparent !important;
}

/* Hide Streamlit's default footer text */
footer a {
    visibility: hidden;
}

/* Bottom spacing */
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}

/* Subtle horizontal separation */
hr {
    border: none !important;
    border-top: 1px solid rgba(148, 163, 184, 0.12) !important;
    margin: 30px 0 !important;
}
/* =========================================================
   21B-19 — FINAL VISUAL CONSISTENCY
   ========================================================= */

/* Consistent page width */
.block-container {
    max-width: 1450px !important;
}

/* Consistent headings */
h1 {
    font-weight: 800 !important;
    letter-spacing: -1px !important;
}

h2 {
    font-weight: 750 !important;
    letter-spacing: -0.5px !important;
}

h3 {
    font-weight: 700 !important;
}

/* Consistent cards */
div[data-testid="stMetric"],
div[data-testid="stAlert"] {
    border-radius: 18px !important;
}

/* Consistent inputs */
input,
textarea {
    border-radius: 12px !important;
}

/* Consistent selectboxes */
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
}

/* Consistent buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
}

/* Consistent data tables */
div[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Consistent chart containers */
div[data-testid="stPlotlyChart"] {
    border-radius: 20px !important;
    overflow: hidden !important;
}

/* Consistent dividers */
hr {
    border-top: 1px solid rgba(148, 163, 184, 0.12) !important;
}
</style>
""", unsafe_allow_html=True)
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

    st.markdown("""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-badge">● MODEL ONLINE</div>
            <h1>Social Media Sentiment Analyzer</h1>
            <p>
                AI-powered analysis of social media conversations
                using Natural Language Processing and Machine Learning.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    ## 📊 Social Media Sentiment Analyzer

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