import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#fetch historical data
stock_data = yf.download("AAPL",start="2018-01-01", end="2024-12-31")

print(stock_data.head())

plt.figure(figsize=(12,6))
plt.plot(stock_data['Close'], label="AAPL Closing Price")
plt.title("Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()