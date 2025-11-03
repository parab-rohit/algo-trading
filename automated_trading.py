import os

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from transformers import pipeline
import time

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Fetch historical data for a specific stock
def fetch_data(stock_symbol):
    data = yf.download(stock_symbol, period="1y", interval="1d")
    return data

#Example: Got Tesla stocks data
tesla_data = fetch_data("TSLA")
print(tesla_data.head())

def moving_average_crossover(data, short_window=5, long_window=20):
    #calculate moving averages
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    #Generate trading signals
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1.0, 0.0)
    data['Position'] = data['Signal'].diff()
    return data

tesla_data = moving_average_crossover(tesla_data)
print(tesla_data.head())

#Sentiment analysis using hugging face
sentiment_analyzer = pipeline("sentiment-analysis")

def fetch_sentiments(news_data):
    url = f"https://newsapi.org/v2/everything?q={news_data}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    print(response.json())
    article_texts = []
    articles = response.json()['articles']
    for article in articles:
        text = f"{article.get('title')} {article.get('description')}"
        if text.strip():
            article_texts.append(text)
    if article_texts:
        sentiments = sentiment_analyzer(article_texts)
        return sentiments
    else:
        sentiments = []

tesla_sentiments = fetch_sentiments("Tesla")

#Get sentiments for Tesla news
if tesla_sentiments:
    positive_count = sum(1 for sentiment in tesla_sentiments if sentiment['label'] == 'POSITIVE')
    positive_sentiments = positive_count > len(tesla_sentiments) / 2
    print(f"Positive Sentiments: {positive_sentiments}")
    print(f"Positive articles: {positive_count}/{len(tesla_sentiments)}")
else:
    print("No sentiment data available")

def trading_decosion_with_sentiment(data, sentiment):
    if sentiment == "POSITIVE" and data['Position'].iloc[-1] == 1:
        print("Positive Sentiment and crossover detected. Buy!")
    elif sentiment == "NEGATIVE" and data['Position'].iloc[-1] == -1:
        print("Negative Sentiment and crossover detected. Sell!")
    else:
        print("No sentiment or no crossover detected.")

trading_decosion_with_sentiment(tesla_data, "NEGATIVE")