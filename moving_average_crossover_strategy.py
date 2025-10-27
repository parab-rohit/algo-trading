import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch historical data for Apple (AAPL) for the last 5 years
symbol = "AAPL"
data = yf.download(symbol, period="5y", interval="1d")


# Calculate indicators and backtest strategy
def calculate_and_backtest(data, short_window, long_window, initial_capital=10000):
    df = data.copy()

    # Calculate moving averages
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()

    # Generate buy and sell signals
    df['buy_signal'] = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
    df['sell_signal'] = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))

    # Position: 1 when in long, 0 when out
    df['Position'] = np.where(df['Short_MA'] > df['Long_MA'], 1, 0)

    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()

    # Strategy returns (apply position from previous day)
    df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)

    # Portfolio value
    df['Portfolio_Value'] = (1 + df['Strategy_Returns']).cumprod() * initial_capital

    return df


# Grid search for best parameters
def grid_search(data, short_range, long_range):
    best_short = None
    best_long = None
    best_value = -np.inf
    best_df = None

    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue  # skip invalid combinations

            df = calculate_and_backtest(data, short_window, long_window)
            final_value = df['Portfolio_Value'].iloc[-1]

            if final_value > best_value:
                best_value = final_value
                best_short = short_window
                best_long = long_window
                best_df = df.copy()

    print(f"✅ Best Parameters Found:")
    print(f"   Short Window = {best_short}")
    print(f"   Long Window  = {best_long}")
    print(f"   Final Portfolio Value = ${best_value:,.2f}")
    return best_df, best_short, best_long


# Run grid search
short_range = range(5, 30, 5)
long_range = range(20, 100, 10)
best_df, best_short, best_long = grid_search(data, short_range, long_range)

# Plot results
plt.figure(figsize=(14, 8))

# Price and moving averages
plt.subplot(2, 1, 1)
plt.plot(best_df.index, best_df['Close'], label='Close Price', color='black')
plt.plot(best_df.index, best_df['Short_MA'], label=f'Short MA ({best_short})', color='blue')
plt.plot(best_df.index, best_df['Long_MA'], label=f'Long MA ({best_long})', color='red')
plt.title(f'AAPL Moving Average Crossover (Short={best_short}, Long={best_long})')
plt.legend()

# Equity curve
plt.subplot(2, 1, 2)
plt.plot(best_df.index, best_df['Portfolio_Value'], label='Strategy Equity', color='green')
plt.title('Portfolio Value Over Time')
plt.ylabel('Portfolio ($)')
plt.legend()

plt.tight_layout()
plt.show()
