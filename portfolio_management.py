from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#define portfolio tickers
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
initial_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
risk_free_rate = 0.02  #Annualized risk free rate

#Fetch one year of daily close prices
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

#Calculate daily and cumulative returns
def calculate_returns(data):
    daily_returns = data.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    return daily_returns, cumulative_returns

#Compute Portfolio Metrics (eg. Sharpe ratio)
def calculate_sharpe_ratio(returns, weights):
    portfolio_return = returns.dot(weights).mean() * 252
    portfolio_volatility = np.sqrt(weights @ returns.cov() @ weights.T)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

#Build and train the LSTM model for trend prediction
def build_and_train_LSTM(data, lookback=30):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    x,y = [],[]
    for i in range(lookback, len(normalized_data)):
        x.append(normalized_data[i-lookback:i])
        y.append(normalized_data[i])

    x,y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=20, batch_size=16, verbose=1)
    return model, scaler


#perform PCA for the risk factor analysis
def perform_PCA(returns):
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    pca = PCA()
    pca.fit(scaled_returns)
    explained_variance = pca.explained_variance_ratio_
    return explained_variance, pca.components_

#Mock function for Gen-AI based recommandationsys
def generate_recommendation(weights, risk_factor):
    recommendations = []
    for i, weight in enumerate(weights):
        if weight > 0.25:
            recommendations.append(f"Reduce exposure to {tickers[i]} to minimize concentration risk.")
        elif risk_factor[i] > 0.2:
            recommendations.append(f"Consider diversifying holdings similar to {tickers[i]} to reduce risk.")
        else:
            recommendations.append(f"Maintain current weight for {tickers[i]}.")
    return recommendations

#Main function
def main():
    #define time range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    #Fetch and Process data
    data = fetch_data(tickers, start_date, end_date)
    daily_returns, cumulative_returns = calculate_returns(data)

    #Compute sharpe ratio and portfolio metrics
    portfolio_return, portfolio_volatility, sharpe_ratio = calculate_sharpe_ratio(daily_returns, initial_weights)
    print(f"Portfolio Return: {portfolio_return:.2f}")
    print(f"Portfolio Volatility: {portfolio_volatility:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    #Train LSTM model for trend prediction
    lstm_model, scaler = build_and_train_LSTM(cumulative_returns.mean(axis=1).values.reshape(-1, 1))
    lookback = 30
    x_test = scaler.transform(cumulative_returns.mean(axis=1).values[-lookback:].reshape(-1, 1))
    x_test = np.reshape(x_test, (1, lookback, 1))

    trend_prediction = scaler.inverse_transform(lstm_model.predict(x_test))
    print(f"Predicted Trend: {trend_prediction[0][0]:.2f}")

    #Perform PCA for risk factor analysis
    explained_variance, components = perform_PCA(daily_returns)
    print(f"Explained Variance: {explained_variance}")
    print(f"Principal Components: {components}")

    #Generate recommendations based on risk factor analysis
    recommandations = generate_recommendation(initial_weights, explained_variance)
    print("\nrecommandations:")
    for recommendation in recommandations:
        print(f"- {recommendation}")

    #Visualize Performance
    cumulative_returns.plot(figsize=(14, 7))
    plt.plot(cumulative_returns.index, cumulative_returns.dot(initial_weights), label='Portfolio Cumulative Return')
    plt.title('Portfolio Cumulative Performance')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    #Visualize PCA contributions
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(explained_variance)), explained_variance, color='teal')
    # plt.xticks(range(len(tickers)), tickers, rotation=45)
    plt.title('PCA Contribution to Risk', fontsize=14)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Variance Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()