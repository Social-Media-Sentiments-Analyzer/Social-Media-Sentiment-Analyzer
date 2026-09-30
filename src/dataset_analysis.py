import pandas as pd

# Load dataset
file_path = "data/raw/Tweets.csv"
df = pd.read_csv(file_path)

print("=" * 50)
print("SOCIAL MEDIA SENTIMENT DATASET ANALYSIS")
print("=" * 50)

# 1. Dataset size
print("\n1. DATASET SIZE")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# 2. Column names
print("\n2. COLUMNS")
print(df.columns.tolist())

# 3. Sentiment distribution
print("\n3. SENTIMENT DISTRIBUTION")
print(df["airline_sentiment"].value_counts())

# 4. Missing values
print("\n4. MISSING VALUES")
print(df[["text", "airline_sentiment"]].isnull().sum())

# 5. Duplicate tweets
print("\n5. DUPLICATE TWEETS")
print("Duplicate tweets:", df["text"].duplicated().sum())

# 6. Average tweet length
df["tweet_length"] = df["text"].astype(str).str.len()

print("\n6. TWEET LENGTH")
print("Average tweet length:", round(df["tweet_length"].mean(), 2))
print("Shortest tweet:", df["tweet_length"].min())
print("Longest tweet:", df["tweet_length"].max())

# 7. Airline distribution
print("\n7. AIRLINE DISTRIBUTION")
print(df["airline"].value_counts())

print("\nAnalysis completed successfully!")