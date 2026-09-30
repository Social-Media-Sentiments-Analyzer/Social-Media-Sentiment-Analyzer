import pandas as pd

file_path = "data/raw/Tweets.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print()

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print()

print("Column names:")
print(df.columns.tolist())

print()
print("First 5 rows:")
print(df.head())