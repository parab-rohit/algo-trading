import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Portfolio Configuration
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
initial_weights = np.array([0.2] * len(tickers))  # Equal weights
risk_tolerance = 0.25  # Maximum acceptable volatility


# Fetch real time data
def fetch_real_time_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data


# Calculate portfolio daily returns and volatility
def calculate_risk_metrics(data, weights):
    daily_returns = data.pct_change().dropna()
    covariance_matrix = daily_returns.cov()
    portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights.T)
    return daily_returns, portfolio_volatility


# Prepare data for LSTM
def prepare_lstm_data(data, lookback=30):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        Y.append(data[i])
    return np.array(X), np.array(Y)


# Build and train the LSTM model
def build_and_train_lstm(data, lookback=30):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    x, y = prepare_lstm_data(normalized_data, lookback)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=20, batch_size=16, verbose=1)

    return model, scaler


# Adjust portfolio weights dynamically based on risk management strategy
def adjust_weights(daily_returns, target_volatility):
    covariance_matrix = daily_returns.cov()
    asset_volatilities = np.sqrt(np.diag(covariance_matrix))
    inverse_volatility_weights = 1 / asset_volatilities
    raw_weights = inverse_volatility_weights / inverse_volatility_weights.sum()

    # Scale to target portfolio volatility
    portfolio_volatility = np.sqrt(
        raw_weights @ covariance_matrix @ raw_weights.T
    )
    scaled_weights = raw_weights * (target_volatility / portfolio_volatility)
    return scaled_weights


def plot_portfolio_weights(tickers, weights, adjusted_weights):
    plt.figure(figsize=(12, 6))
    x = range(len(tickers))
    width = 0.35

    plt.bar([i - width / 2 for i in x], weights, width, label='Initial Weights')
    plt.bar([i + width / 2 for i in x], adjusted_weights, width, label='Adjusted Weights')

    plt.xlabel('Stocks')
    plt.ylabel('Weight')
    plt.title('Portfolio Weights Comparison')
    plt.xticks(x, tickers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    # Define the time range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    print("Fetching stock data...")
    # Fetch data
    data = fetch_real_time_data(tickers, start_date, end_date)

    # Calculate risk metrics
    daily_returns, portfolio_volatility = calculate_risk_metrics(data, initial_weights)

    print(f"\nInitial Portfolio Volatility: {portfolio_volatility:.4f}")

    # Adjust portfolio weights based on risk tolerance
    adjusted_weights = adjust_weights(daily_returns, risk_tolerance)
    _, adjusted_volatility = calculate_risk_metrics(data, adjusted_weights)

    print(f"Adjusted Portfolio Volatility: {adjusted_volatility:.4f}")
    print(f"\nInitial weights: {initial_weights}")
    print(f"Adjusted weights: {adjusted_weights}")
    print(f"Sum of adjusted weights: {adjusted_weights.sum():.4f}")

    # LSTM Trend Prediction
    print("\nTraining LSTM model for risk trend prediction...")
    avg_returns = daily_returns.mean(axis=1).values
    lstm_model, scaler = build_and_train_lstm(avg_returns)

    lookback = 30
    x_test = scaler.transform(avg_returns[-lookback:].reshape(-1, 1))
    x_test = np.reshape(x_test, (1, lookback, 1))
    predicted_risk_trend = scaler.inverse_transform(lstm_model.predict(x_test, verbose=0))
    print(f"\nPredicted Risk Trend: {predicted_risk_trend[0][0]:.4f}")

    # Plot portfolio weights comparison
    print("\nGenerating portfolio weights comparison chart...")
    plot_portfolio_weights(tickers, initial_weights, adjusted_weights)


if __name__ == "__main__":
    main()