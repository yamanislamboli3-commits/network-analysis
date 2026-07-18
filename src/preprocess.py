import pandas as pd
from features import selected_features
df = pd.read_csv('data/raw/hikari2021.csv')
df = df[selected_features]
print(df.isnull().sum()) 
df = df.dropna()
print("Duplicates:", df.duplicated().sum())
print('Duplicated rows:', df[df.duplicated()])
df = df.drop_duplicates()
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
print(categorical_columns)
print(df.dtypes)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df["Label"].value_counts())
df.to_csv("data/processed/hikari_processed.csv", index=False)
