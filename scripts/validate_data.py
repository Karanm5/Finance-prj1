import pandas as pd


FILE_PATH = "data/spy.csv"

def validate_data():
    df = pd.read_csv(FILE_PATH)

    print("Intial shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df.dropna()

    df.to_csv(FILE_PATH, index = False)
    print("\nCleaned data saved.")
    print("Final Shape:", df.shape)

if __name__ == "__main__":
    validate_data()