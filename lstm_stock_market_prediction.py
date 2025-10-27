import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

stock_data = yf.download("AAPL", start="2018-01-01", end="2024-01-01")

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Create Training and test dataset
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

def create_dataset(data, step=50):
    X, Y = [], []
    for i in range(step, len(data)):  # Start from 'step' instead of 0
        X.append(data[i-step:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1),
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Get actual prices - reshape to match predictions
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Example: identify the actionable insights based on predictable trends
signal_threshold = 5
predicted_movements = predictions - y_test_actual

# Find predictions exceeding the threshold
actionable_insights = [
    (day, move) for day, move in enumerate(predicted_movements) if abs(move[0]) > signal_threshold
]

# Print actionable signals
print(f"\nActionable Insights (threshold: ${signal_threshold}):")
for signal in actionable_insights:
    print(f"Day {signal[0]}, Predicted Movement of {signal[1][0]:.2f} USD")

# Plot predictions alongside the actual data
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--')
plt.title('AAPL Stock Price Prediction vs Actual')
plt.xlabel('Time (Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()