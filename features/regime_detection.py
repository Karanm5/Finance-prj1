import pandas as pd


INPUT_PATH = "data/spy_features.csv"
OUTPUT_PATH = "data/spy_features_regime.csv"


def detect_market_regime():
    df = pd.read_csv(INPUT_PATH)

    # 0 = Bear, 1 = Sideways , 2 =  Bull
# RULE LOGIC
# BULL if : SMA_50 > SMA_200

#Bear if : SMA_50 < SMA_200

#Sideways if Volatility < rolling threshold


    df["regime"] = 1 # default sideways

    # Bull market condition
    df.loc[df["sma_50"] > df["sma_200"], "regime"] = 2


    #Bear market condition

    df.loc[df["sma_50"] < df["sma_200"], "regime"] = 0


    # Sideways if volatility is low

    low_vol_threshold = df["volatility_20"].quantile(0.3)
    df.loc[df["volatility_20"] < low_vol_threshold, "regime"] = 1

    df.to_csv(OUTPUT_PATH, index = False)

    print("Market regime detection completed.")
    print(df["regime"].value_counts())

if __name__ == "__main__":
    detect_market_regime()



