import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data")

def download_spy():
    print("Downloading SPY historical data..")

    ticker = "SPY"
    df = yf.download(ticker, start = "2015-01-01", end= "2024-12-31", auto_adjust=False)

    df.reset_index(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    DATA_PATH.mkdir(exist_ok = True)
    file_path = DATA_PATH / "spy.csv"

    df.to_csv(file_path, index = False)
    print(f"Saved data to {file_path}")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    download_spy()