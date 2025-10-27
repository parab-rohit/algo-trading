import requests
import pandas as pd
from os import getenv

# Replace with your Alpha Vantage API key
API_KEY = getenv("API_KEY")

# Alpha Vantage endpoint for real-time quote
url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={API_KEY}"

# Fetch data from API
response = requests.get(url)
data = response.json()

# Extract the "Global Quote" data
quote_data = data.get("Global Quote", {})

# Convert to a pandas DataFrame
if quote_data:
    df = pd.DataFrame([quote_data])
    # Optional: Clean up column names (remove "01. ", "02. ", etc.)
    df.columns = [col.split(". ")[-1] for col in df.columns]
    print("\n📊 Real-Time Apple (AAPL) Data:")
    print(df)
else:
    print("⚠️ No data returned. Check your API key or rate limits.")
