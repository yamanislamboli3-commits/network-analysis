import pandas as pd
import numpy as np
import glob

# Read all CSV files
files = glob.glob("data/raw/*WorkingHours*.csv")

dfs = []

for file in files:
    print(f"Reading: {file}")
    dfs.append(pd.read_csv(file, low_memory=False))

# Merge all files
df = pd.concat(dfs, ignore_index=True)


# Remove spaces from column names
df.columns = df.columns.str.strip()

# Check missing values
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Replace infinite values with NaN then remove them
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Check duplicates
print("Duplicates:", df.duplicated().sum())

print("Duplicated rows:")
print(df[df.duplicated()])

# Remove duplicates
df = df.drop_duplicates()

# Convert labels to binary
df["Label"] = df["Label"].str.strip()
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# Print data types
print(df.dtypes)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Print class distribution
print(df["Label"].value_counts())

# Save processed dataset
df.to_csv("data/processed/cicids2017_processed.csv", index=False)

print("\nSaved successfully!")
print("Final Shape:", df.shape)