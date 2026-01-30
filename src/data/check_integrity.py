import pandas as pd
import os

FILE = os.path.join("data", "processed", "benchmark_dataset.csv")

def check_integrity():
    if not os.path.exists(FILE):
        print("File not found.")
        return

    df = pd.read_csv(FILE)
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    print("\nSample Data:")
    print(df.head())
    
    print("\nUnique Regions:", df['Region'].unique())
    
    # Check for duplicates
    dups = df.duplicated(subset=['Date', 'Region'])
    print(f"\nDuplicate Date-Region pairs: {dups.sum()}")
    
    if dups.sum() > 0:
        print("\nExample duplicates:")
        print(df[dups].head())
        print(df[df['Date'] == df[dups].iloc[0]['Date']])

if __name__ == "__main__":
    check_integrity()
