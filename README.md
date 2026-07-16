
# Network Analysis Project

## Project Structure

- data/raw: Original dataset (HIKARI-2021 CSV)
- data/processed: Cleaned dataset
- src/preprocess.py: Data preprocessing
- src/train.py: Train Random Forest model
- src/evaluate.py: Evaluate trained model
- models/: Saved trained model

## Usage

1. Install dependencies

   pip install -r requirements.txt

2. Put the dataset into:

   data/raw/hikari.csv

3. Run preprocessing

   python src/preprocess.py

4. Train model

   python src/train.py

5. Evaluate model

   python src/evaluate.py
