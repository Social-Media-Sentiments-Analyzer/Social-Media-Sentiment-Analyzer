import pandas as pd
import re

# Load original dataset
file_path = "data/raw/Tweets.csv"
df = pd.read_csv(file_path)

print("Original dataset loaded.")
print("Number of tweets:", len(df))


# Function to clean tweet text
def clean_text(text):
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)


# Keep only the columns we need
cleaned_df = df[["clean_text", "airline_sentiment", "airline"]]


# Remove missing values
cleaned_df = cleaned_df.dropna()


# Remove empty tweets
cleaned_df = cleaned_df[cleaned_df["clean_text"].str.strip() != ""]


# Save processed dataset
output_path = "data/processed/cleaned_data.csv"
cleaned_df.to_csv(output_path, index=False)


print("\nCleaning completed successfully!")
print("Cleaned dataset saved to:", output_path)

print("\nFirst 5 cleaned tweets:")
print(cleaned_df.head())

print("\nFinal number of tweets:", len(cleaned_df))