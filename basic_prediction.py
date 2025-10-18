import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

#fetch stock data
stock_data = yf.download("AAPL", start="2018-01-01", end="2024-12-31")

#calculate moving averages
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

#plot the data
plt.figure(figsize = (12,6))
plt.plot(stock_data['Close'], label = "AAPl close price", alpha = 0.6)
plt.plot(stock_data['EMA_50'], label = "50 day EMA", linestyle = "--")
plt.plot(stock_data['SMA_50'], label = "50 day SMA", linestyle = "--")
plt.title("AAPL stock prices with moving averages")
plt.title("Stock prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()









































































































