
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("../data/raw/hikari.csv",
                    "../data/processed/hikari_processed.csv")
