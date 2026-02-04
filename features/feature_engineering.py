import pandas as pd
import numpy as np
import ta

INPUT_PATH = "data/spy.csv"
OUTPUT_PATH = "data/spy_features.csv"

def generate_features():
    df = pd.read_csv(INPUT_PATH)

    #Ensure datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    close = df["Close"]


    #1. Returns
    df['daily_return'] = close.pct_change()  #Percentage change
    df["log_return"] = np.log(close / close.shift(1))


    # Trend Indicators
    # calculate a 20-period Simple Moving Average (SMA)
    # Using a combination of Simple Moving Averages (SMAs) with 20, 50, and 200-day windows allows traders to analyze 
    #multiple timeframes simultaneously, filtering out short-term noise while confirming long-term trend direction. 
    #This approach helps identify support/resistance levels, entry points, and overall market sentiment, especially 
    #when used on instruments like the S&P 500. 
    df["sma_20"]  = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"]  = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["sma_200"]  = ta.trend.SMAIndicator(close, window=200).sma_indicator()

    # 3 Momentum Indicators
# RSI - The Relative Strength Index (RSI) is a momentum oscillator (0–100 scale) measuring the speed and change of price 
# movements to identify overbought (>70) or oversold (<30) conditions. It helps traders spot potential trend reversals, 
# divergences, and confirms momentum, usually over 14 periods. RSI is best used in range-bound markets to time entries 
# and exits. 

# MACD - The Moving Average Convergence Divergence (MACD) is a versatile trend-following momentum indicator used to identify trend 
# direction, strength, and reversals. It calculates the difference between the 12-period and 26-period Exponential Moving 
# Averages (EMA) (MACD line) and a 9-period signal line. Key uses include spotting signal line crossovers, zero-line crossovers
# , and divergence from price trends

    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["macd"] = ta.trend.MACD(close).macd()

    # 4. Volatility / Risk

#VOLATILITY - That line of code calculates the Rolling Standard Deviation over a 20-period window, which is the foundational 
#measurement for technical volatility. What this tells you
#High Values: Indicate the price is swinging widely (high volatility), often seen during news events or strong trend breakouts.
#Low Values: Indicate the price is consolidating or trading in a tight range (low volatility).

# ATR - Calculating the Average True Range (ATR) using the ta library is a major step up from a simple rolling standard deviation. 
#While your previous standard deviation code only looked at closing prices, ATR accounts for intraday price movement (highs and lows) 
# and overnight gaps, making it a more comprehensive measure of "true" volatility.


    df["volatility_20"] = close.rolling(20).std()
    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], close).average_true_range()

    # 5. Volume Signals
# By adding a 20-period Volume Moving Average, you are transitioning from looking at raw volume "spikes" to analyzing the liquidity trend. 
# This is a powerful filter for the MACD and ATR indicators you’ve already defined.

# Adding Percent Change in Volume captures the "shock" factor in the market. While your volume_ma_20 shows the trend, this volume_change 
# identifies the exact moment new players enter the trade.
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()
    df["volume_change"] = df["Volume"].pct_change()

    df = df.dropna()

    df.to_csv(OUTPUT_PATH, index = False)
    print(f"Feature dataset saved to {OUTPUT_PATH}")
    print("Final shape:", df.shape)


if __name__ == "__main__":
    generate_features()
