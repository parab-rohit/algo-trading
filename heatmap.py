import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Simulated data : trend strength for various stocks
trend_data = {
    'Stock': ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    , 'Trend Strength': [0.8, 0.6, -0.2, 0.1, -0.5]
}
trend_df = pd.DataFrame(trend_data)

#reshape the data for the heatmap
trend_pivot = trend_df.set_index('Stock')

#Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(trend_pivot, annot=True, cmap='RdYlGn', center=0,cbar=True)
plt.title('Trend Strength Heatmap')
plt.show()