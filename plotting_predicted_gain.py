import matplotlib.pyplot as plt

assets = ["AAPL","MSFT","GOOGL"]

predicted_gains = [50,30,-20]

plt.figure(figsize=(10,6))
plt.bar(assets, predicted_gains, color=["green" if gain > 0 else "red" for gain in predicted_gains])
plt.title("portfolio level predicted gains/losses")
plt.xlabel("Asset")
plt.ylabel("Predicted Gain/Loss (USD)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.show()
