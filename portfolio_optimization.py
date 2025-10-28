import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
import time
from os import getenv

#Alpha Vantage API key
API_KEY = getenv("API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

#Define stocks and initial weights
stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
initial_weights = np.array([0.2] * len(stocks))

# fetch one minute data for a single stock
def fetch_intraday_data(symbol, interval="1min", outputsize="full"):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    response_json = response.json()
    time_series_key = f"Time Series ({interval})"

    if time_series_key in response_json:
        data = response_json[time_series_key]
        df = pd.DataFrame.from_dict(data, orient="index",dtype=float)
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print(f"Error fetching data for {symbol}: {response_json['Error Message']}")
        return None

# Fetch data for all stocks
stock_data = {}
for stock in stocks:
    print(f"Fetching data for {stock}...")
    stock_data[stock] = fetch_intraday_data(stock)
    time.sleep(12) # Avoiding api rate limit

# Combine adjusted close prices into a single dataframe
prices = pd.DataFrame({stock: stock_data[stock]["close"] for stock in stocks})

# Calculate daily returns from one minute data
returns = prices.resample("1D").last().pct_change().dropna()

#Portfolio optimization
def portfolio_metrics(weights, returns):
    """"
    Calculate portfolio return, standard deviation, and sharpe ratio
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252 #Annulize returns
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))) #Annulize volatility
    portfolio_sharpe = portfolio_return / portfolio_std
    return portfolio_return, portfolio_std, portfolio_sharpe

def negative_sharpe_ratio(weights, returns):
    """
    The objective function to minimize. (negative sharpe ratio)
    """
    _,_,sharpe_ratio = portfolio_metrics(weights, returns)
    return -sharpe_ratio

def optimize_portfolio(returns):
    """
    Optimize portfolio weights to maximize the sharpe ratio
    """
    num_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets)) #Weights are between 0 and 1
    # initial_guess = np.array([1 / num_assets] * num_assets) #Initial guess for each asset
    result = minimize(negative_sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#Run Optimization
opt_result = optimize_portfolio(returns)

#Display results
optimal_weights = opt_result.x
portfolio_return, portfolio_std, sharpe_ratio = portfolio_metrics(optimal_weights, returns)

# Create dictionary of stock-weight pairs
stock_weights = dict(zip(stocks, optimal_weights))

print("\nOptimal Portfolio Weights:")
for stock, weight in stock_weights.items():
    print(f"{stock}: {weight:.2%}")

print("\nPortfolio Metrics:")
print(f"Portfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Standard Deviation: {portfolio_std:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"\nAnnualized Metrics:")
print(f"Expected Annual Return: {portfolio_return * 252:.2%}")
print(f"Expected Annual Volatility: {portfolio_std * np.sqrt(252):.2%}")
print(f"Expected Annual Sharpe Ratio: {sharpe_ratio * np.sqrt(252):.2f}")